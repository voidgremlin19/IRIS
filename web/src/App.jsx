import { useState, useRef, useCallback, useEffect } from 'react'

/* ─── Constants ───────────────────────────────────────────────── */
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

/* ─── App ─────────────────────────────────────────────────────── */
export default function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  const [theme, setTheme] = useState('dark')
  const [inputMode, setInputMode] = useState('upload') // 'upload' | 'dashcam'
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [realTimeActive, setRealTimeActive] = useState(false)

  /* ── Theme Management ──────────────────────────────────────── */
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(t => (t === 'dark' ? 'light' : 'dark'))
  }

  /* ── Check API health ──────────────────────────────────────── */
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`)
        if (res.ok) {
          setApiStatus('online')
          return
        }
      } catch {
        // Health check on API_BASE failed
      }
      setApiStatus('offline')
    }
    check()
    const iv = setInterval(check, 15000)
    return () => clearInterval(iv)
  }, [])

  /* ── Dashcam Management ────────────────────────────────────── */
  useEffect(() => {
    if (inputMode === 'dashcam') {
      startCamera()
    } else {
      stopCamera()
    }
    return () => stopCamera()
  }, [inputMode])

  // Real-time dashcam capture loop
  useEffect(() => {
    if (inputMode !== 'dashcam' || !realTimeActive) return

    let intervalId = setInterval(async () => {
      // Only capture and analyze if not currently loading another frame
      if (!loading) {
        const file = await captureFrame()
        if (file) {
          handleFile(file, true) // skip preview to keep video running
          analyzeImage(file)
        }
      }
    }, 6000) // capture every 6s

    return () => clearInterval(intervalId)
  }, [inputMode, realTimeActive, loading])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setError(null)
    } catch (err) {
      setError('Could not access camera. Please allow camera permissions.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }

  const captureFrame = () => {
    if (!videoRef.current) return null

    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
    
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], "dashcam_capture.jpg", { type: "image/jpeg" })
          resolve(file)
        } else {
          resolve(null)
        }
      }, 'image/jpeg', 0.85)
    })
  }

  const analyzeDashcamFrame = async () => {
    const file = await captureFrame()
    if (file) {
      handleFile(file, true) // skip preview update to keep video running
      analyzeImage(file)
    }
  }

  /* ── File selection ───────────────────────────────────────── */
  const handleFile = useCallback((file, skipPreview = false) => {
    if (!file || (!file.type.startsWith('image/') && !skipPreview)) {
      setError('Please upload a valid image file (JPEG, PNG)')
      return
    }
    setImage(file)
    setError(null)
    setResult(null)
    if (!skipPreview) {
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(file)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    if (inputMode !== 'upload') return
    const file = e.dataTransfer?.files?.[0]
    if (file) handleFile(file)
  }, [handleFile, inputMode])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    if (inputMode !== 'upload') return
    setDragging(true)
  }, [inputMode])

  const handleDragLeave = useCallback(() => setDragging(false), [])

  /* ── Analyze Image ────────────────────────────────────────── */
  const analyzeImage = async (fileToAnalyze = image) => {
    if (!fileToAnalyze) return
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', fileToAnalyze)

      const res = await fetch(`${API_BASE}/analyze?generate_voice=true&generate_viz=true`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Server error: ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to analyze image. Is the API running?')
    } finally {
      setLoading(false)
    }
  }

  /* ── Voice Playback ───────────────────────────────────────── */
  const playVoiceAlert = useCallback(() => {
    if (!result) return

    const voiceAlert = result.voice_alert
    const alertText = voiceAlert?.text || result.reasoning?.alert || 'No alert available'

    // Try backend audio first
    if (voiceAlert?.audio_base64) {
      try {
        const audio = new Audio(`data:audio/mp3;base64,${voiceAlert.audio_base64}`)
        audio.play().catch(() => {
          // Fallback to browser Speech API
          speakText(alertText)
        })
        return
      } catch { /* fall through */ }
    }

    // Browser Speech API fallback
    speakText(alertText)
  }, [result])

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(text)
      
      // Fetch browser voices
      const voices = window.speechSynthesis.getVoices()
      
      // Priority list of premium humanoid/neural voices
      const priorityPatterns = [
        'Google US English', 
        'Google UK English Female', 
        'Google UK English Male', 
        'Microsoft Aria Online',
        'Microsoft Guy Online',
        'Samantha',
        'Daniel',
        'Rishi',
        'Veena',
        'Google English (India)',
        'en-US',
        'en-IN',
        'en-GB'
      ]

      let selectedVoice = null
      if (voices.length > 0) {
        for (const pattern of priorityPatterns) {
          selectedVoice = voices.find(v => v.name.includes(pattern) || v.lang === pattern)
          if (selectedVoice) break
        }
        
        if (!selectedVoice) {
          selectedVoice = voices.find(v => v.lang.startsWith('en'))
        }
      }

      if (selectedVoice) {
        utterance.voice = selectedVoice
        utterance.lang = selectedVoice.lang
      } else {
        utterance.lang = 'en-IN'
      }

      utterance.rate = 0.95  // Human conversational rate
      utterance.pitch = 1.0  // Natural voice pitch
      utterance.volume = 1.0
      window.speechSynthesis.speak(utterance)
    } else {
      alert(`Voice Alert: ${text}`)
    }
  }

  /* ── Auto-play Voice Alert ─────────────────────────────────── */
  useEffect(() => {
    if (result && voiceEnabled) {
      const timer = setTimeout(() => {
        playVoiceAlert()
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [result, voiceEnabled, playVoiceAlert])

  /* ── Reset ────────────────────────────────────────────────── */
  const resetAll = () => {
    setImage(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
    setRealTimeActive(false)
  }

  /* ─── Render ────────────────────────────────────────────────── */
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-logo">IR</div>
            <div>
              <div className="header-title">Indian Road Intelligence</div>
              <div className="header-subtitle">AI Driver Assistant System</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <button className="btn btn-secondary" onClick={toggleTheme}>
              {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
            </button>
            <div className="header-status">
              <span className={`status-dot ${apiStatus === 'online' ? 'online' : 'offline'}`} />
              <span>API: {apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Checking...' : 'Offline'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Input Section */}
        <section className="upload-section" id="upload-section">
          <div className="card">
            <div className="card-header">
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button 
                  className={`btn ${inputMode === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => { setInputMode('upload'); resetAll(); }}
                >
                  Image Upload
                </button>
                <button 
                  className={`btn ${inputMode === 'dashcam' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => { setInputMode('dashcam'); resetAll(); }}
                >
                  Live Dashcam
                </button>
              </div>
              {image && inputMode === 'upload' && (
                <button className="btn btn-danger" onClick={resetAll}>
                  Clear
                </button>
              )}
            </div>
            <div className="card-body">
              {inputMode === 'upload' ? (
                <div
                  className={`upload-zone ${dragging ? 'dragging' : ''} ${imagePreview ? 'has-image' : ''}`}
                  onClick={() => !imagePreview && fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  id="upload-drop-zone"
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFile(e.target.files?.[0])}
                    style={{ display: 'none' }}
                    id="file-input"
                  />

                  {imagePreview ? (
                    <div className="upload-preview">
                      <img src={imagePreview} alt="Uploaded road scene" />
                      <div className="upload-actions">
                        <button
                          className="btn btn-primary"
                          onClick={(e) => { e.stopPropagation(); analyzeImage() }}
                          disabled={loading}
                          id="analyze-button"
                        >
                          {loading ? (
                            <>
                              <span className="spinner" />
                              Analyzing...
                            </>
                          ) : (
                            <>Analyze Scene</>
                          )}
                        </button>
                        <button
                          className="btn btn-secondary"
                          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                          id="change-image-button"
                        >
                          Change Image
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="upload-text">
                        Drop a road scene image here, or click to browse
                      </div>
                      <div className="upload-hint">
                        Supports JPEG, PNG • Optimized for Indian traffic scenes
                      </div>
                    </>
                  )}

                  {loading && (
                    <div className="loading-overlay">
                      <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
                      <div className="loading-text">Processing scene through AI pipeline...</div>
                      <div className="loading-bar">
                        <div className="loading-bar-fill" />
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="upload-zone has-image" style={{ cursor: 'default' }}>
                  <div className="upload-preview">
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      muted 
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '400px', 
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        backgroundColor: '#000'
                      }} 
                    />
                    <div className="upload-actions" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                      <button
                        className="btn btn-primary"
                        onClick={analyzeDashcamFrame}
                        disabled={loading || realTimeActive}
                      >
                        {loading && !realTimeActive ? (
                          <>
                            <span className="spinner" />
                            Analyzing...
                          </>
                        ) : (
                          <>Capture & Analyze</>
                        )}
                      </button>

                      <label style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.5rem', 
                        fontSize: '0.85rem', 
                        cursor: 'pointer',
                        padding: '0.5rem 1.2rem',
                        background: realTimeActive ? 'rgba(59, 130, 246, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                        border: realTimeActive ? '1px solid #3b82f6' : '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        color: realTimeActive ? '#3b82f6' : 'var(--text-secondary)',
                        transition: 'all 0.2s ease',
                        fontWeight: 600
                      }}>
                        <input 
                          type="checkbox" 
                          checked={realTimeActive} 
                          onChange={(e) => setRealTimeActive(e.target.checked)}
                          style={{ cursor: 'pointer' }}
                        />
                        Real-Time Tracking
                      </label>
                    </div>
                  </div>
                  {loading && (
                    <div className="loading-overlay">
                      <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
                      <div className="loading-text">Processing frame through AI pipeline...</div>
                      <div className="loading-bar">
                        <div className="loading-bar-fill" />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {error && (
                <div style={{
                  marginTop: '1rem',
                  padding: '0.75rem 1rem',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  borderRadius: 'var(--radius-md)',
                  color: 'var(--danger)',
                  fontSize: '0.85rem',
                }}>
                  Error: {error}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Results */}
        {result ? (
          <div className="results-section">
            {/* Decision Banner */}
            <div className="card decision-card fade-in">
              <div className="card-header">
                <div className="card-title">
                  AI Decision
                </div>
                <span className="card-title" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {result.reasoning?.reasoning_method === 'rule_based' ? 'Rule-Based' : 'LLM'}
                </span>
              </div>
              <div className="card-body">
                <div className={`decision-banner risk-${result.reasoning?.risk || 'none'}`}>
                  <div className="decision-text">
                    <h3>{(result.reasoning?.decision || 'unknown').replace(/_/g, ' ')}</h3>
                    <p>{result.reasoning?.context}</p>
                  </div>
                  <span className="decision-risk-badge">
                    {result.reasoning?.risk} Risk
                  </span>
                </div>

                {/* Voice Alert */}
                <div className="voice-alert-box">
                  <span className="voice-text">
                    🔊 Alert: {result.reasoning?.alert || 'No alert'}
                  </span>
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                      <input 
                        type="checkbox" 
                        checked={voiceEnabled} 
                        onChange={(e) => setVoiceEnabled(e.target.checked)}
                        style={{ cursor: 'pointer' }}
                      />
                      Auto-Speak
                    </label>
                    <button className="btn btn-voice" onClick={playVoiceAlert} id="play-voice-button">
                      Play Audio
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Visualization */}
            {result.visualization && (
              <div className="card viz-card slide-in-left">
                <div className="card-header">
                  <div className="card-title">
                    Annotated View
                  </div>
                </div>
                <div className="card-body">
                  <img
                    className="viz-image"
                    src={`data:image/jpeg;base64,${result.visualization}`}
                    alt="Annotated road scene with detections"
                  />
                </div>
              </div>
            )}

            {/* Detected Objects */}
            <div className="card slide-in-left">
              <div className="card-header">
                <div className="card-title">
                  Detected Objects
                </div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-accent)', fontWeight: 600 }}>
                  {result.detections?.count || 0} found
                </span>
              </div>
              <div className="card-body">
                <div className="object-list">
                  {(result.detections?.objects || []).map((obj, i) => {
                    const dist = result.distances?.distances?.[i]
                    return (
                      <div key={i} className="object-item">
                        <div className="object-info">
                          <div>
                            <div className="object-type-name">{obj.name || obj.type}</div>
                            <div className="object-confidence">
                              {(obj.confidence * 100).toFixed(1)}% confidence
                            </div>
                          </div>
                        </div>
                        <div className="object-meta">
                          {dist && (
                            <>
                              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                ~{dist.estimated_meters}m
                              </span>
                              <span className={`distance-badge ${dist.zone}`}>
                                {dist.zone}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    )
                  })}
                  {result.detections?.count === 0 && (
                    <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                      No objects detected in this scene
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Scene Graph */}
            <div className="card slide-in-right">
              <div className="card-header">
                <div className="card-title">
                  Scene Graph
                </div>
              </div>
              <div className="card-body">
                {/* Zones */}
                <div className="scene-zones">
                  {['left', 'center', 'right'].map((zone) => (
                    <div key={zone} className="zone-section">
                      <div className="zone-title">
                        {zone.toUpperCase()}
                      </div>
                      <div className="zone-items">
                        {(result.scene_graph?.zones?.[zone] || []).map((obj, j) => (
                          <span key={j} className="zone-chip">
                            {obj.name || obj.type}
                          </span>
                        ))}
                        {(result.scene_graph?.zones?.[zone] || []).length === 0 && (
                          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Empty</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Relations */}
                <div className="zone-title" style={{ marginBottom: '0.5rem' }}>Spatial Relations</div>
                <div className="relations-list">
                  {(result.scene_graph?.relations || []).slice(0, 10).map((rel, i) => (
                    <div key={i} className="relation-item">
                      <span className="relation-subject">{rel.subject}</span>
                      <span className="relation-type">{rel.relation}</span>
                      <span className="relation-object">{rel.object}</span>
                    </div>
                  ))}
                  {(result.scene_graph?.relations || []).length === 0 && (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      No spatial relations detected
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Distance Summary */}
            <div className="card slide-in-left">
              <div className="card-header">
                <div className="card-title">
                  Distance Analysis
                </div>
              </div>
              <div className="card-body">
                <div className="perf-grid">
                  <div className="perf-item">
                    <div className="perf-value" style={{ color: 'var(--danger)' }}>
                      {result.distances?.summary?.near_objects || 0}
                    </div>
                    <div className="perf-label">Near (&lt;10m)</div>
                  </div>
                  <div className="perf-item">
                    <div className="perf-value" style={{ color: 'var(--warning)' }}>
                      {result.distances?.summary?.medium_objects || 0}
                    </div>
                    <div className="perf-label">Medium (10-30m)</div>
                  </div>
                  <div className="perf-item">
                    <div className="perf-value" style={{ color: 'var(--success)' }}>
                      {result.distances?.summary?.far_objects || 0}
                    </div>
                    <div className="perf-label">Far (&gt;30m)</div>
                  </div>
                </div>
                {result.distances?.summary?.closest_object && (
                  <div style={{
                    marginTop: '1rem',
                    padding: '0.75rem',
                    background: 'rgba(239, 68, 68, 0.06)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.85rem',
                    border: '1px solid rgba(239, 68, 68, 0.1)',
                  }}>
                    Closest: <strong>{result.distances.summary.closest_object.name || result.distances.summary.closest_object.type}</strong> at{' '}
                    <strong>{result.distances.summary.closest_object.estimated_meters}m</strong>
                    {' '}({result.distances.summary.closest_object.zone})
                  </div>
                )}
              </div>
            </div>

            {/* Performance */}
            <div className="card slide-in-right">
              <div className="card-header">
                <div className="card-title">
                  Performance
                </div>
              </div>
              <div className="card-body">
                <div className="perf-grid">
                  <div className="perf-item">
                    <div className="perf-value">
                      {((result.performance?.total_seconds || 0) * 1000).toFixed(0)}
                    </div>
                    <div className="perf-label">Total (ms)</div>
                  </div>
                  <div className="perf-item">
                    <div className="perf-value">
                      {((result.performance?.detection_seconds || 0) * 1000).toFixed(0)}
                    </div>
                    <div className="perf-label">Detection</div>
                  </div>
                  <div className="perf-item">
                    <div className="perf-value">
                      {((result.performance?.reasoning_seconds || 0) * 1000).toFixed(0)}
                    </div>
                    <div className="perf-label">Reasoning</div>
                  </div>
                  <div className="perf-item">
                    <div className="perf-value">{result.detections?.count || 0}</div>
                    <div className="perf-label">Objects</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Temporal Buffer */}
            {result.temporal && (
              <div className="card voice-section fade-in">
                <div className="card-header">
                  <div className="card-title">
                    Temporal Context ({result.temporal.frame_count} frames, {result.temporal.window_seconds}s)
                  </div>
                </div>
                <div className="card-body">
                  <div className="voice-alert-box">
                    <span className="voice-text">{result.temporal.warning}</span>
                    <span className={`decision-risk-badge`} style={{
                      background: 'var(--info-bg)',
                      color: 'var(--info)',
                    }}>
                      Confidence: {(result.temporal.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          !loading && (
            <div className="idle-state">
              <div className="idle-title">System Ready</div>
              <div className="idle-desc">
                Select an input method above to begin processing road scenes through the perception pipeline.
              </div>
            </div>
          )
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div>Indian Road Intelligence System v1.0</div>
        <div>From perception to reasoning — building the AI brain for Indian roads.</div>
      </footer>
    </div>
  )
}

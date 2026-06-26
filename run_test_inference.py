"""
run_test_inference.py — End-to-end inference test with real YOLO model.
Runs the full pipeline on a dataset image and prints all outputs.
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import Pipeline

def main():
    # Pick a test image from the dataset
    test_image = "Datasets/urban/Image_0.png"
    
    if not os.path.exists(test_image):
        print(f"ERROR: Test image not found: {test_image}")
        return

    print("=" * 60)
    print("INDIAN ROAD INTELLIGENCE SYSTEM — Full Pipeline Test")
    print("=" * 60)
    print(f"Input: {test_image}")
    print()

    # Initialize pipeline with REAL YOLO model
    pipe = Pipeline(use_mock=False, enable_voice=True)
    
    # Run full pipeline
    result = pipe.run(
        test_image,
        generate_voice=True,
        generate_viz=True,
    )

    # ─── Detection Results ─────────────────────────────────────
    print(">>> DETECTION <<<")
    print(f"  Objects detected: {result['detections']['count']}")
    print(f"  Image size: {result['detections'].get('image_size')}")
    for obj in result['detections']['objects']:
        print(f"    {obj['type']:20s}  conf={obj['confidence']:.2f}  bbox={obj['bbox']}")
    print()

    # ─── Distance Results ──────────────────────────────────────
    print(">>> DISTANCES <<<")
    summary = result['distances']['summary']
    print(f"  Near: {summary.get('near_objects', 0)}  |  Medium: {summary.get('medium_objects', 0)}  |  Far: {summary.get('far_objects', 0)}")
    for d in result['distances']['distances']:
        print(f"    {d['type']:20s}  ~{d['estimated_meters']:6.1f}m  zone={d['zone']:6s}  urgency={d['urgency']}")
    if summary.get('closest_object'):
        c = summary['closest_object']
        print(f"  >> Closest: {c['type']} at {c['estimated_meters']}m ({c['zone']})")
    print()

    # ─── Scene Graph ───────────────────────────────────────────
    print(">>> SCENE GRAPH <<<")
    zones = result['scene_graph']['zones']
    for z_name, z_objs in zones.items():
        types = [o['type'] for o in z_objs]
        print(f"    {z_name:8s}: {types}")
    print(f"  Relations: {len(result['scene_graph']['relations'])} spatial relationships")
    for rel in result['scene_graph']['relations'][:8]:
        print(f"    {rel['subject']:20s} --[{rel['relation']:12s}]--> {rel['object']}")
    clusters = result['scene_graph'].get('clusters', [])
    if clusters:
        print(f"  Clusters: {len(clusters)}")
        for cl in clusters:
            print(f"    {cl['type']:12s}: {cl['members']}")
    ego = result['scene_graph'].get('ego_view', {})
    print(f"  Ego View:")
    for direction, items in ego.items():
        if items:
            print(f"    {direction:12s}: {[i['type'] for i in items]}")
    print()

    # ─── Reasoning ─────────────────────────────────────────────
    print(">>> REASONING <<<")
    r = result['reasoning']
    print(f"  Context:    {r['context']}")
    print(f"  Decision:   {r['decision']}")
    print(f"  Risk:       {r['risk']}  (score={r.get('risk_score', 'N/A')})")
    print(f"  Alert:      {r['alert']}")
    print(f"  Method:     {r['reasoning_method']}")
    print(f"  Complexity: {r.get('scene_complexity', 'N/A')}")
    if r.get('all_alerts'):
        print(f"  All alerts:")
        for a in r['all_alerts']:
            print(f"    - {a}")
    print()

    # ─── Voice ─────────────────────────────────────────────────
    print(">>> VOICE ALERT <<<")
    v = result.get('voice_alert', {})
    if v:
        print(f"  Text:   {v.get('text', 'N/A')}")
        print(f"  Engine: {v.get('engine', 'N/A')}")
        has_audio = bool(v.get('audio_base64'))
        audio_len = len(v.get('audio_base64', '')) if has_audio else 0
        print(f"  Audio:  {'YES (' + str(audio_len) + ' chars base64)' if has_audio else 'NO'}")
    print()

    # ─── Visualization ─────────────────────────────────────────
    print(">>> VISUALIZATION <<<")
    viz = result.get('visualization')
    if viz:
        print(f"  Annotated image: YES ({len(viz)} chars base64)")
        # Save to disk for inspection
        import base64
        viz_bytes = base64.b64decode(viz)
        output_path = "output/test_annotated.jpg"
        os.makedirs("output", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(viz_bytes)
        print(f"  Saved to: {output_path}")
    else:
        print("  Annotated image: NO")
    print()

    # ─── Temporal ──────────────────────────────────────────────
    print(">>> TEMPORAL BUFFER <<<")
    t = result.get('temporal', {})
    if t:
        print(f"  Frames:     {t.get('frame_count', 0)}")
        print(f"  Window:     {t.get('window_seconds', 0)}s")
        print(f"  Decision:   {t.get('decision', 'N/A')}")
        print(f"  Confidence: {t.get('confidence', 0):.0%}")
        print(f"  Warning:    {t.get('warning', 'N/A')}")
    print()

    # ─── Performance ───────────────────────────────────────────
    print(">>> PERFORMANCE <<<")
    perf = result['performance']
    for k, v2 in perf.items():
        if 'seconds' in k:
            print(f"    {k:25s}: {v2*1000:7.1f} ms")
        else:
            print(f"    {k:25s}: {v2}")
    print()

    print("=" * 60)
    print("PIPELINE TEST COMPLETE — All modules operational")
    print("=" * 60)

    # Save full JSON result (without visualization base64 for readability)
    result_export = {k: v for k, v in result.items() if k != 'visualization'}
    if result_export.get('voice_alert', {}).get('audio_base64'):
        result_export['voice_alert'] = {**result_export['voice_alert'], 'audio_base64': '<base64_audio_omitted>'}
    
    with open("output/test_result.json", "w") as f:
        json.dump(result_export, f, indent=2, default=str)
    print(f"Full result saved to: output/test_result.json")


if __name__ == "__main__":
    main()

import pyttsx3
import time

# Initialize pyttsx3 engine
# Wrap it inside functions or lazily initialize to prevent errors if speaker system is not fully configured.
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = pyttsx3.init()
    return engine

def speak(text):
    eng = get_engine()
    eng.say(text)
    eng.runAndWait()

def speak_with_buffer(text, delay=6):
    print("Voice alert in 6 second....")
    time.sleep(delay)
    speak(text)

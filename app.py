import sounddevice as sd
import numpy as np
import whisper
import tempfile
from scipy.io.wavfile import write
import os

model = whisper.load_model("base")  # Use 'tiny' model for faster processing

SAMPLE_RATE = 16000
DURATION = 10

def process_audio_chunk(audio_data):
    """
    Process and transcribe an audio chunk.
    """
    print("Processing audio chunk...")

    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()

    try:
        write(temp_wav.name, SAMPLE_RATE, audio_data)

        result = model.transcribe(temp_wav.name, fp16=False)  # Use fp16=False for CPU
        print("Transcription:", result['text'])
    finally:
        os.unlink(temp_wav.name)

def audio_callback(indata, frames, time, status):
    """
    Callback function for audio streaming.
    """
    if status:
        print(f"Status: {status}")
    process_audio_chunk(indata.flatten())

print("Starting real-time transcription... Speak into your microphone!")
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=SAMPLE_RATE * DURATION):
        sd.sleep(DURATION * 1000 * 5)
except Exception as e:
    print(f"Error: {e}")

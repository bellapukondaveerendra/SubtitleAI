import sounddevice as sd
import numpy as np
import whisper
import tempfile
from scipy.io.wavfile import write
import os

# Load the Whisper model
model = whisper.load_model("base")  # Use 'tiny' model for faster processing

# Settings for real-time audio capture
SAMPLE_RATE = 16000  # Whisper requires 16kHz
DURATION = 10  # Smaller chunk duration to reduce processing delay

def process_audio_chunk(audio_data):
    """
    Process and transcribe an audio chunk.
    """
    print("Processing audio chunk...")

    # Normalize and convert audio to float32
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))

    # Create a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()  # Close the file so we can write to it

    try:
        # Save the audio data to the temporary WAV file
        write(temp_wav.name, SAMPLE_RATE, audio_data)

        # Transcribe the saved audio file
        result = model.transcribe(temp_wav.name, fp16=False)  # Use fp16=False for CPU
        print("Transcription:", result['text'])
    finally:
        # Clean up the temporary file
        os.unlink(temp_wav.name)

def audio_callback(indata, frames, time, status):
    """
    Callback function for audio streaming.
    """
    if status:
        print(f"Status: {status}")

    # Flatten the audio data and send it for processing
    process_audio_chunk(indata.flatten())

print("Starting real-time transcription... Speak into your microphone!")
try:
    # Start audio input stream with increased blocksize
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=SAMPLE_RATE * DURATION):
        sd.sleep(DURATION * 1000 * 5)  # Run the stream for 5 chunks
except Exception as e:
    print(f"Error: {e}")

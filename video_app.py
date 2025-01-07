import sounddevice as sd
import numpy as np
import whisper
import tempfile
from scipy.io.wavfile import write
import os
import cv2
import textwrap

model = whisper.load_model("base")  # Use 'tiny' model for faster transcription


SAMPLE_RATE = 16000
DURATION = 5


subtitle_text = "" 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_audio_chunk(audio_data):
    """
    Process and transcribe an audio chunk.
    """
    global subtitle_text
    print("Processing audio chunk...")

    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav.close()

    try:
        write(temp_wav.name, SAMPLE_RATE, audio_data)
        result = model.transcribe(temp_wav.name, fp16=False)  # Use fp16=False for CPU
        subtitle_text = result['text']
        print("Transcription:", subtitle_text)
    finally:
        os.unlink(temp_wav.name)

def audio_callback(indata, frames, time, status):
    """
    Callback function for audio streaming.
    """
    if status:
        print(f"Status: {status}")
    process_audio_chunk(indata.flatten())

def draw_wrapped_text(frame, text, x, y, max_width, font_scale, color, thickness):
    """
    Draw word-wrapped text on the frame.
    """
    wrapped_text = textwrap.wrap(text, width=max_width)
    y_offset = 0
    for line in wrapped_text:
        cv2.putText(frame, line, (x, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += int(font_scale * 30)

def start_camera():
    """
    Start the video stream and overlay subtitles dynamically above detected faces.
    """
    global subtitle_text
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting camera... Press 'q' to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (frame_width, frame_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                subtitle_y = max(y - 30, 30)
                draw_wrapped_text(frame, subtitle_text, x, subtitle_y, max_width=40, font_scale=0.5, color=(0, 0, 255), thickness=1)
            cv2.imshow("Real-Time Subtitles", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

print("Starting real-time transcription and video processing... Speak into your microphone!")
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=SAMPLE_RATE * DURATION):
        start_camera()
except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")

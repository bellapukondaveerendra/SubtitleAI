import sounddevice as sd
import numpy as np
import whisper
import tempfile
from scipy.io.wavfile import write
import os
import cv2
import textwrap  # For word wrapping

# Load the Whisper model
model = whisper.load_model("base")  # Use 'tiny' model for faster transcription

# Settings for audio capture
SAMPLE_RATE = 16000  # Whisper requires 16kHz
DURATION = 5  # Chunk duration in seconds

# Rolling subtitles buffer
subtitle_text = ""  # Store the latest transcription

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_audio_chunk(audio_data):
    """
    Process and transcribe an audio chunk.
    """
    global subtitle_text
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
        subtitle_text = result['text']  # Update global subtitle text
        print("Transcription:", subtitle_text)
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

def draw_wrapped_text(frame, text, x, y, max_width, font_scale, color, thickness):
    """
    Draw word-wrapped text on the frame.
    """
    wrapped_text = textwrap.wrap(text, width=max_width)
    y_offset = 0
    for line in wrapped_text:
        cv2.putText(frame, line, (x, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += int(font_scale * 30)  # Adjust line spacing

def start_camera():
    """
    Start the video stream and overlay subtitles dynamically above detected faces.
    """
    global subtitle_text

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting camera... Press 'q' to exit.")
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to ensure full-screen experience
            frame_height, frame_width, _ = frame.shape
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            # Overlay subtitles for detected faces
            for (x, y, w, h) in faces:
                # Position subtitles dynamically above the face
                subtitle_y = max(y - 30, 30)  # Ensure subtitles stay within frame bounds
                draw_wrapped_text(frame, subtitle_text, x, subtitle_y, max_width=40, font_scale=0.5, color=(0, 0, 255), thickness=1)

            # Show the frame
            cv2.imshow("Real-Time Subtitles", frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

print("Starting real-time transcription and video processing... Speak into your microphone!")
try:
    # Start audio input stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=SAMPLE_RATE * DURATION):
        # Start the camera stream in parallel
        start_camera()
except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")

import whisper
import torch
import time 
print(torch.cuda.is_available())  # Should return True

# Load the Whisper model (small model for faster processing)
model = whisper.load_model("base")
print()
result = model.transcribe("sample_audio.mp3", fp16=False)  # Replace with your own audio file
print("Transcription:", result['text'])

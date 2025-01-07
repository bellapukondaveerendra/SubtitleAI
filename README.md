### **SubTitleAI**

`

SubTitleAI is a real-time speech-to-text and video subtitle overlay tool powered by AI. The application captures live speech, transcribes it into subtitles using OpenAI's Whisper model, and dynamically overlays the subtitles on a camera feed above detected faces. Ideal for real-time accessibility, video streaming, and language learning applications.

## Features

- **Real-Time Speech Transcription**: Converts speech to text in real time.
- **Dynamic Face Detection**: Detects faces in the video feed and positions subtitles above them.
- **Subtitle Styling**: Displays subtitles in a chat bubble format with customizable colors and text wrapping.
- **Full-Screen Camera Feed**: Ensures the camera feed occupies the entire screen for an immersive experience.
- **Word Wrapping**: Long subtitles are split into multiple lines for readability.

## Technologies Used

- **OpenCV**: For video processing and face detection.
- **SoundDevice**: For real-time audio capture.
- **Whisper**: OpenAI's Whisper model for speech-to-text transcription.
- **Python**: Core programming language for the project.

## Installation

1. **Clone the Repository**:

   ```
   git clone https://github.com/bellapukondaveerendra/SubtitleAI
   cd SubtitleAI

   ```

1. **Install Dependencies**: Ensure you have Python 3.7+ installed, then install the required libraries:

   `pip install opencv-python sounddevice openai-whisper numpy scipy textwrap`

1. **Download Whisper Model**: The Whisper model will be downloaded automatically when you run the project for the first time.

1. **Run the Application**:

   `python video_app.py`

## Usage

- **Start Transcription**:
  - Speak into your microphone, and the application will transcribe your speech in real time.
- **Watch Subtitles**:
  - Subtitles will appear dynamically above detected faces in the camera feed.
- **Exit**:
  - Press `q` to quit the application.

## Customization

1.  **Subtitle Color**:

    - Modify the `color` parameter in the `draw_wrapped_text` function to use your preferred subtitle color.
    - Example: Use `(0, 0, 255)` for red or `(139, 0, 0)` for dark blue.

2.  **Font Size**:

    - Adjust the `font_scale` parameter in `draw_wrapped_text` for larger or smaller text.

3.  **Camera Resolution**:

    - Change the resolution in the `cv2.resize()` function to match your screen size.

4.  **Model Accuracy vs. Speed**:

    - Switch to a smaller Whisper model (e.g., `"tiny"`) for faster performance:

      `model = whisper.load_model("tiny")`

## Future Enhancements

- **Multi-Language Support**: Add real-time translation for multilingual subtitles.
- **Improved Face Tracking**: Use advanced face tracking algorithms for smoother subtitle placement.
- **Multi-Speaker Handling**: Assign subtitles to specific speakers in multi-user scenarios.
- **Subtitle Animations**: Add animations for chat bubbles to enhance visual appeal.

## Acknowledgments

- OpenAI for providing the Whisper speech-to-text model.
- OpenCV for robust video processing and face detection.
- Python community for supporting rich libraries and tools.

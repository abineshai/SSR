🎙️ Real-Time Silent Speech Recognition

Lip-reading powered by deep learning — converts silent lip movements into text in real time, with no audio required.


📌 About the Project
This project implements a Real-Time Silent Speech Recognition (SSR) system that reads lip movements from a live webcam feed and converts them into text — entirely without audio input.
The system uses a CNN + LSTM deep learning pipeline for visual feature extraction and sequence modeling, MediaPipe for face and lip landmark detection, and an Ollama language model (LLaMA 3.2) for post-processing and grammar correction. Recognized text is displayed live in a Tkinter GUI and simultaneously written to a text file and Notepad.
Three UI options are available: a desktop hotkey-driven app, a Gradio web interface, and a Streamlit web interface.

🧠 How It Works
Webcam Input
    ↓
MediaPipe Face Mesh → Mouth Region Extraction → Preprocessing & Augmentation
    ↓
CNN (Spatial Feature Extraction)
    ↓
LSTM (Temporal Sequence Modeling)
    ↓
CTC Decoding → Raw Text
    ↓
Ollama LLaMA 3.2 (Grammar & Context Correction)
    ↓
Output → Tkinter GUI + Notepad + transcription.txt

✨ Features

🔇 Audio-Free Recognition — works entirely from lip movements, no microphone needed
📷 Real-Time Webcam Processing — live feed processed frame by frame
🧠 AI-Powered Correction — LLaMA 3.2 via Ollama refines raw lip-read output
🖥️ Multiple UI Options — Desktop (Tkinter), Gradio web UI, Streamlit web UI
⌨️ Hotkey Control — Alt / T to toggle recording, Esc to quit
📝 Auto Text Output — recognized text written to Notepad and saved to file
🌐 Browser Support — upload or record video via Gradio or Streamlit interface


🗂️ Project Structure
SSR/
├── configs/
│   └── LRS3_V_WER19.1.ini       # Model config
├── benchmarks/
│   └── LRS3/
│       ├── language_models/
│       │   └── lm_en_subword/   # Language model weights
│       └── models/
│           └── LRS3_V_WER19.1/  # VSR model weights
├── hydra_configs/
│   └── default.yaml             # Hydra config
├── pipelines/
│   └── pipeline.py              # Inference pipeline
├── main.py                      # Desktop app (Tkinter + hotkeys)
├── ui_gradio.py                 # Gradio web interface
├── web_ui.py                    # Streamlit web interface
├── web_ui_advanced.py           # Streamlit web interface (advanced)
├── transcription.txt            # Live output log
├── requirements.txt
└── README.md

⚙️ Requirements
Hardware
ComponentMinimumRecommendedRAM4 GB8–16 GBCPUDual Corei5 / i7 / Ryzen 5 / 7GPUIntegratedCUDA-capable (NVIDIA)Webcam720p1080p HDStorage10 GB free20+ GB free
Software

Windows 10/11
Python 3.9 or 3.10
CUDA 11.7 / 11.8 (optional, for GPU acceleration)
FFmpeg
Ollama with llama3.2 model


🚀 Setup
1. Clone the repository
bashgit clone https://github.com/abineshai/SSR.git
cd SSR
2. Download model weights
Download the following and place them in the correct directories:

LRS3_V_WER19.1 → benchmarks/LRS3/models/LRS3_V_WER19.1/
lm_en_subword → benchmarks/LRS3/language_models/lm_en_subword/

3. Install dependencies
bashpip install -r requirements.txt
4. Install and run Ollama
bash# Install Ollama from https://ollama.com
ollama pull llama3.2
ollama serve
5. Install FFmpeg
Download from ffmpeg.org and add to system PATH.

▶️ Usage
Option 1 — Desktop App (Tkinter + Hotkeys)
bashpython main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe

Press Alt or T to start recording
Mouth your words silently in front of the webcam
Press Alt or T again to stop — text will appear in the GUI and Notepad
Press Esc or Q to quit

Option 2 — Gradio Web Interface
bashpython ui_gradio.py
Open your browser at http://localhost:7860 — upload or record a silent video to get transcription.
Option 3 — Streamlit Web Interface
bashstreamlit run web_ui.py
# or for the advanced version:
streamlit run web_ui_advanced.py

📊 Performance

✅ Accurate lip detection under good lighting with direct face-to-camera angle
✅ Reliable recognition for common English words and short phrases
⚠️ Accuracy drops with fast speech, facial hair, or low lighting
⚠️ Best results when user faces the camera directly with minimal head movement


🛠️ Tech Stack
LayerTechnologyLanguagePython 3.9Deep LearningPyTorch, CNN + LSTM + CTCFace DetectionMediaPipeVideo ProcessingOpenCV, FFmpegLanguage CorrectionOllama (LLaMA 3.2)Desktop UITkinterWeb UIGradio, StreamlitConfig ManagementHydraVersion ControlGit, GitHub

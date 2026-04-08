# 🎙️ NoMic AI — SilentSense AI
 
A real-time lip-reading application that converts **silent lip movements into text** using deep learning — no audio required. Runs fully locally with webcam input.
 
---
 
## 📸 Features
 
- 📷 **Real-time webcam lip reading** — processes live video frame by frame
- 🔇 **Audio-free recognition** — converts visual lip motion into text without any sound
- 🧠 **AI-powered correction** — LLaMA 3.2 via Ollama refines raw transcription output
- 🖥️ **Multiple UI options** — Desktop (Tkinter), Gradio web UI, Streamlit web UI
- ⌨️ **Hotkey control** — `Alt` / `T` to toggle recording, `Esc` to quit
- 📝 **Auto text output** — recognized text written live to Notepad and saved to file
- 🌐 **Browser support** — upload or record silent video via Gradio or Streamlit
 
---
 
## 🏗️ Project Structure
 
```
/NoMicAI
├── benchmarks/
│   └── LRS3/
│       ├── language_models/
│       │   └── lm_en_subword/          # Language model weights
│       └── models/
│           └── LRS3_V_WER19.1/         # VSR model weights
├── configs/
│   └── LRS3_V_WER19.1.ini              # Model config file
├── espnet/                             # ESPnet dependency
├── hydra_configs/
│   └── default.yaml                    # Hydra configuration
├── pipelines/
│   └── pipeline.py                     # Inference pipeline
├── main.py                             # Desktop app (Tkinter + hotkeys)
├── ui_gradio.py                        # Gradio web interface
├── web_ui.py                           # Streamlit web interface
├── web_ui_advanced.py                  # Streamlit web interface (advanced)
├── transcription.txt                   # Live output log
├── requirements.txt
└── README.md
```
 
---
 
## ⚙️ Setup Instructions
 
### Prerequisites
 
- Windows 10 / 11
- Python 3.9 or 3.10
- CUDA 11.7 / 11.8 (optional, for GPU acceleration)
- FFmpeg
- [Ollama](https://ollama.com) with `llama3.2` model pulled
 
---
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/abineshai/NoMicAI.git
cd NoMicAI
```
 
---
 
### 2. Download Model Weights
 
Download both components and place them in the correct directories:
 
- [`LRS3_V_WER19.1`](https://drive.google.com/file/d/1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6/view) → `benchmarks/LRS3/models/LRS3_V_WER19.1/`
- [`lm_en_subword`](https://drive.google.com/file/d/1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv/view) → `benchmarks/LRS3/language_models/lm_en_subword/`
 
---
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
---
 
### 4. Install FFmpeg
 
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.
 
---
 
### 5. Set Up Ollama
 
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
ollama serve
```
 
---
 
## ▶️ Usage
 
### Option 1 — Desktop App (Tkinter + Hotkeys)
 
```bash
python main.py config_filename=./configs/LRS3_V_WER19.1.ini detector=mediapipe
```
 
| Key | Action |
|-----|--------|
| `Alt` or `T` | Toggle recording on/off |
| `Esc` or `Q` | Quit the application |
 
Once recording stops, recognized text appears in the GUI window and is written to Notepad automatically.
 
---
 
### Option 2 — Gradio Web Interface
 
```bash
python ui_gradio.py
```
 
Open `http://localhost:7860` — upload or record a silent video to get the transcription.
 
---
 
### Option 3 — Streamlit Web Interface
 
```bash
streamlit run web_ui.py
# or for the advanced version:
streamlit run web_ui_advanced.py
```
 
---
 
## 🔄 How It Works
 
```
Webcam Input
    ↓
MediaPipe Face Mesh → Mouth Region Extraction → Preprocessing & Augmentation
    ↓
CNN  →  Spatial feature extraction (shape, texture, appearance)
    ↓
LSTM →  Temporal sequence modeling (lip motion over time)
    ↓
CTC Decoding  →  Raw recognized text
    ↓
Ollama LLaMA 3.2  →  Grammar & context correction
    ↓
Output: Tkinter GUI + Notepad + transcription.txt
```
 
---
 
## 🧮 Model Details
 
| Component | Detail |
|-----------|--------|
| Base Model | Auto-AVSR — trained on LRS3 dataset |
| Model Config | `LRS3_V_WER19.1` (Word Error Rate: 19.1%) |
| Face Detection | MediaPipe Face Mesh |
| Sequence Decoder | CTC (Connectionist Temporal Classification) |
| Language Correction | Ollama — LLaMA 3.2 |
 
---
 
## 📊 Performance
 
- ✅ Accurate lip detection under good lighting with direct face-to-camera angle
- ✅ Reliable recognition for common English words and short phrases
- ⚠️ Accuracy drops with fast speech, facial hair, or low lighting
- ⚠️ Best results when the user faces the camera directly with minimal head movement
 
---
 
## 🛠️ Tech Stack
 
### Backend / Core
- `torch` + `torchvision` + `torchaudio` — deep learning
- `opencv-python` — webcam capture and video processing
- `mediapipe` — face mesh and lip landmark detection
- `ollama` + `pydantic` — LLM integration and structured output
- `hydra-core` — config management
- `keyboard` — hotkey listener
- `scipy` + `scikit-image` — preprocessing
 
### Frontend / UI
- `tkinter` — desktop GUI with live scrolling text
- `gradio` — browser-based video upload interface
- `streamlit` + `streamlit-webrtc` — browser-based live webcam interface
 
---
 
## 💻 Hardware Requirements
 
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8–16 GB |
| CPU | Dual Core | i5 / i7 / Ryzen 5 / 7 |
| GPU | Integrated | CUDA-capable NVIDIA GPU |
| Webcam | 720p | 1080p HD |
| Storage | 10 GB free | 20+ GB free |
 
---
 

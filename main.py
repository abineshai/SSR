import os
import cv2
import torch
import keyboard
import time
from datetime import datetime
from pipelines.pipeline import InferencePipeline
from pywinauto import Application

# -----------------------------------------------------------
# 🧠 Model Setup
# -----------------------------------------------------------
print("[main] Starting Chaplin Silent Speech Recognition...")
device = torch.device("cpu")

model = InferencePipeline(
    "./configs/LRS3_V_WER19.1.ini",
    device=device,
    detector="mediapipe",
    face_track=True
)
print("[main] Model loaded successfully!")

# -----------------------------------------------------------
# 🧩 Webcam Setup
# -----------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam. Please check your camera.")

print("[startup] Webcam started. Press ALT to start/stop recording, ESC to exit.")

# Variables
recording = False
frames = []
save_dir = "recordings"
os.makedirs(save_dir, exist_ok=True)

# -----------------------------------------------------------
# 📝 Function to send text to Notepad
# -----------------------------------------------------------
def send_to_notepad(text):
    try:
        app = Application(backend="win32").connect(title_re=".* - Notepad")
        app_window = app.top_window()
        app_window.type_keys(f"\n{text}", with_spaces=True)
    except Exception:
        # If Notepad not open, open a new one and type
        app = Application(backend="win32").start("notepad.exe")
        time.sleep(1)
        app_window = app.top_window()
        app_window.type_keys(text, with_spaces=True)

# -----------------------------------------------------------
# 🎥 Function to save and process video
# -----------------------------------------------------------
def process_video(frames):
    if not frames:
        print("[warning] No frames captured. Try again.")
        return

    filename = os.path.join(save_dir, f"webcam_{int(time.time())}.mp4")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for f in frames:
        out.write(f)
    out.release()

    print(f"[record] Saved video: {filename}, frames={len(frames)}")
    print("[inference] Processing video...")

    try:
        result = model(filename)
        print(f"[inference] Recognized Text: {result}")
        send_to_notepad(result)
    except Exception as e:
        print(f"[error] Inference failed: {str(e)}")

# -----------------------------------------------------------
# 🔁 Main Loop
# -----------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Silent Speech Recognition - Press ALT to Start/Stop, ESC to Quit", frame)

    # Toggle recording with ALT
    if keyboard.is_pressed("alt"):
        if not recording:
            recording = True
            frames = []
            print("[hotkey] Recording started...")
            time.sleep(0.5)
        else:
            recording = False
            print("[hotkey] Recording stopped.")
            process_video(frames)
            time.sleep(0.5)

    # Save frames while recording
    if recording:
        frames.append(frame.copy())

    # Quit on ESC
    if keyboard.is_pressed("esc"):
        print("[shutdown] Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()

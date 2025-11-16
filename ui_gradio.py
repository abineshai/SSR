import cv2
import time
import gradio as gr
import numpy as np
from pipelines.pipeline import InferencePipeline
import torch
import threading

# -----------------------------
# Load model once
# -----------------------------
device = torch.device("cpu")
model = InferencePipeline(
    "./configs/LRS3_V_WER19.1.ini",
    device=device,
    detector="mediapipe",
    face_track=True
)

# Shared resources
frame_buffer = []
lock = threading.Lock()
last_inference_text = "Waiting for input..."


# -----------------------------
# Background inference thread
# -----------------------------
def inference_loop():
    global last_inference_text, frame_buffer

    while True:
        time.sleep(2)  # every 2 sec

        lock.acquire()
        frames = frame_buffer.copy()
        frame_buffer.clear()
        lock.release()

        if len(frames) < 10:
            continue

        # Save temporary video
        h, w, _ = frames[0].shape
        temp_video = "temp_live.mp4"
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
        for f in frames:
            out.write(f)
        out.release()

        # Run model
        try:
            text = model(temp_video)
            last_inference_text = text if text.strip() else "No speech detected"
        except Exception as e:
            last_inference_text = f"⚠️ Model Error: {e}"


# Start inference thread
threading.Thread(target=inference_loop, daemon=True).start()


# -----------------------------
# Webcam frame callback
# -----------------------------
def grab_frame(frame):
    """
    frame → numpy array (RGB)
    """
    global frame_buffer

    if frame is None:
        return last_inference_text

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    lock.acquire()
    frame_buffer.append(img)
    lock.release()

    return last_inference_text


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 🧠 Real-Time Silent Speech Recognition (Lip→Text)")
    gr.Markdown("Speak silently — the system reads your lip movement.")

    webcam = gr.Image(label="Webcam Input", type="numpy", webcam=True)



    output_text = gr.Textbox(
        label="Recognized Text",
        value="Waiting…",
        interactive=False
    )

    webcam.change(
        fn=grab_frame,
        inputs=webcam,
        outputs=output_text
    )

demo.launch()

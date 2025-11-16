import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import threading
import queue
import time
import cv2
import torch
from pipelines.pipeline import InferencePipeline

# ----------------------
# Load Model Once
# ----------------------
device = torch.device("cpu")
model = InferencePipeline(
    "./configs/LRS3_V_WER19.1.ini",
    device=device,
    detector="mediapipe",
    face_track=True
)

# Queue for model output
output_queue = queue.Queue()


# ----------------------
# Live Lip Reader Class
# ----------------------
class LiveLipReader(VideoTransformerBase):
    def __init__(self):
        self.frame_buffer = []   # store recent frames
        self.last_process_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Store last 1.5 seconds of frames (approx 25 FPS)
        self.frame_buffer.append(img)
        if len(self.frame_buffer) > 40:  # 40 frames ≈ 1.5 sec
            self.frame_buffer.pop(0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------
# Inference Worker
# ----------------------
def process_live_frames(transformer):
    global output_queue

    while True:
        time.sleep(2)  # process every 2 seconds

        if not transformer.frame_buffer:
            continue

        frames = list(transformer.frame_buffer)

        # Save temp video
        temp_path = f"temp_live_{int(time.time())}.mp4"
        height, width, _ = frames[0].shape

        writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height))
        for f in frames:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            writer.write(f)
        writer.release()

        # Run model
        try:
            text = model(temp_path)
            if text.strip():
                output_queue.put(text)
        except Exception as e:
            output_queue.put(f"⚠️ Model Error: {e}")

        os.remove(temp_path)


# ----------------------
# Streamlit UI
# ----------------------
st.title("🧠 LIVE Silent Speech Recognition")
st.write("Lip movements → Real-time text output (no sound required)")

ctx = webrtc_streamer(
    key="live_lip_read",
    video_transformer_factory=LiveLipReader,
    media_stream_constraints={"video": True, "audio": False}
)

# Start worker thread once
if "worker_started" not in st.session_state:
    if ctx.video_transformer:
        threading.Thread(target=process_live_frames, args=(ctx.video_transformer,), daemon=True).start()
        st.session_state["worker_started"] = True

# Display results
if not output_queue.empty():
    text = output_queue.get()
    st.session_state["last_text"] = text

if "last_text" in st.session_state:
    st.success(f"📝 **{st.session_state['last_text']}**")

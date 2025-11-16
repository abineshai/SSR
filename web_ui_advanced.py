import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av, cv2, torch, time, threading, queue
from pipelines.pipeline import InferencePipeline

device = torch.device("cpu")
model = InferencePipeline("./configs/LRS3_V_WER19.1.ini", device=device, detector="mediapipe", face_track=True)

output_text_queue = queue.Queue()

class LipReader(VideoTransformerBase):
    def __init__(self):
        self.record = False
        self.frames = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.record:
            self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def infer_video(frames):
    global output_text_queue
    if len(frames) == 0:
        output_text_queue.put("⚠️ Speak again — No frames recorded.")
        return

    temp = f"temp_{int(time.time())}.mp4"
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames: out.write(f)
    out.release()

    try:
        result = model(temp)
        output_text_queue.put(result)
    except Exception as e:
        output_text_queue.put(f"⚠️ Model Error: {e}")

    os.remove(temp)

st.markdown("<h1 style='text-align:center;'>🧠 Silent Speech Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Speak without sound, your lips tell the message.</p>", unsafe_allow_html=True)

ctx = webrtc_streamer(key="webcam", video_transformer_factory=LipReader)

col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Start Recording"):
        ctx.video_transformer.record = True
with col2:
    if st.button("🛑 Stop Recording"):
        ctx.video_transformer.record = False
        frames = ctx.video_transformer.frames.copy()
        ctx.video_transformer.frames = []
        threading.Thread(target=infer_video, args=(frames,), daemon=True).start()

st.divider()

if not output_text_queue.empty():
    st.session_state["result"] = output_text_queue.get()

if "result" in st.session_state:
    st.markdown("<h3>✅ Recognized Text:</h3>", unsafe_allow_html=True)
    st.success(st.session_state["result"])

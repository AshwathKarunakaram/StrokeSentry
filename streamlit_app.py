# streamlit_app.py

import streamlit as st
import numpy as np
import subprocess
import json
import tempfile
import os
from PIL import Image
import tensorflow as tf
from pydub import AudioSegment
import librosa

# ---- Config ----
MODEL_SPEC = "gemma3n:e2b-it-q4_K_M"  # your pulled model
OLLAMA_CMD = "ollama"

# ---- Helpers for TFLite inference ----
@st.cache_resource
def load_tflite_model(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    return interp, inp, out

def run_tflite(interp, inp, outp, data):
    arr = np.array(data, dtype=inp[0]["dtype"]).reshape(tuple(inp[0]["shape"]))
    interp.set_tensor(inp[0]["index"], arr)
    interp.invoke()
    out = np.squeeze(interp.get_tensor(outp[0]["index"]))
    return float(out) if np.ndim(out)==0 else float(np.mean(out))

# ---- Preprocessing ----
def preprocess_audio(uploaded_file, shape):
    # 16 kHz mono
    audio = AudioSegment.from_file(uploaded_file).set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    shape = list(shape)
    if len(shape)==2:
        N = shape[1]
        samples = np.pad(samples, (0, max(0, N-samples.size)))[:N]
        return samples.reshape(tuple(shape))
    # mel‑spectrogram for [1,H,W] or [1,H,W,1]
    H, W = shape[1], shape[2]
    hop = max(1, samples.shape[0]//W)
    mel = librosa.feature.melspectrogram(y=samples, sr=16000, n_mels=H, hop_length=hop)
    db  = librosa.power_to_db(mel, ref=np.max)
    norm = (db - db.min())/(db.max()-db.min())
    if norm.shape[1] < W:
        norm = np.pad(norm, ((0,0),(0,W-norm.shape[1])))
    norm = norm[:, :W]
    if len(shape)==4:
        norm = norm[..., None]
    return norm[np.newaxis, ...].astype(np.float32)

def preprocess_image(uploaded_file, shape):
    shape = list(shape)
    _, H, W, C = shape
    img = Image.open(uploaded_file).convert("RGB").resize((W, H))
    arr = np.array(img, dtype=np.float32)/255.0
    if arr.shape[-1] != C:
        arr = arr.mean(axis=-1, keepdims=True)
    return arr.reshape(tuple(shape))

# ---- Multimodal via Ollama CLI ----
def ask_gemma_cli(audio_bytes, img_bytes, slur, droop):
    # write to temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
        af.write(audio_bytes); audio_path = af.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pf:
        pf.write(img_bytes); img_path = pf.name

    # build prompt string embedding file paths
    prompt = (
        f"Audio slur score: {slur:.2f}, face droop score: {droop:.2f}. "
        f"Review the recording at {audio_path} and the photo at {img_path}. "
        "You are a board‑certified stroke triage specialist. "
        "Based on these inputs, determine if this is likely a stroke. "
        "If stroke signs are present, respond with:\n"
        "“CALL 911 IMMEDIATELY. Note time of onset, keep the patient lying flat with head slightly elevated, ensure airway is clear, loosen tight clothing, do NOT give food or drink, reassure and monitor breathing until EMS arrives.”\n"
        "If no stroke signs are detected, respond with:\n"
        "“No immediate emergency. Continue to monitor symptoms, keep the patient comfortable, and consult a healthcare professional if anything worsens.”"
    )


    # run Ollama CLI
    proc = subprocess.run(
        [OLLAMA_CMD, "run", MODEL_SPEC, prompt],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # clean up temp files
    os.unlink(audio_path); os.unlink(img_path)

    if proc.returncode != 0:
        return f"🛑 Ollama CLI error:\n{proc.stderr.strip()}"
    return proc.stdout.strip()

# ---- Streamlit UI ----
st.title("🚨 On‑Device Stroke Triage MVP")

audio_file = st.file_uploader("Upload 3–5 s voice sample", type=["wav","mp3"])
img_file   = st.file_uploader("Upload face photo",     type=["png","jpg","jpeg"])

if st.button("Run Analysis"):
    if not audio_file or not img_file:
        st.error("Both audio and image are required.")
    else:
        # Load models
        dys, d_in, d_out = load_tflite_model("models/dysarthria.tflite")
        face, f_in, f_out = load_tflite_model("models/droopy_face.tflite")

        # Run TFLite
        slur  = run_tflite(dys, d_in, d_out, preprocess_audio(audio_file, d_in[0]['shape']))
        droop = run_tflite(face, f_in, f_out, preprocess_image(img_file, f_in[0]['shape']))
        st.write(f"🔊 Slur score: **{slur:.2f}**")
        st.write(f"😐 Droop score: **{droop:.2f}**")

        # CLI multimodal call
        advice = ask_gemma_cli(audio_file.read(), img_file.read(), slur, droop)
        st.markdown("### 🤖 Gemma’s Multimodal Advice")
        st.write(advice)

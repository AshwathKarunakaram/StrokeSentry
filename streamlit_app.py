import streamlit as st
import cv2, mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time, tempfile, io, os, requests
from PIL import Image
import tensorflow as tf
from pydub import AudioSegment
import librosa.display
from scipy.signal import butter, filtfilt, find_peaks
from audiorecorder import audiorecorder

# ---- Page & Styling ----
st.set_page_config(page_title="FAST+P Stroke Triage Dashboard", layout="wide")
st.markdown(
    """
    <style>
      .stMetric { background: linear-gradient(135deg,#e0f7fa,#80deea); border-radius: 0.75rem; padding: 1rem; }
      .stButton>button { font-size: 1.1rem; padding: 0.75rem 1.5rem; border-radius: 0.5rem; }
      .css-1d391kg { background-color: #f0f4c3; }
    </style>
    """, unsafe_allow_html=True
)
st.title("🚨 FAST+P Stroke Triage Dashboard")

# ---- Config & Sidebar ----
OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL_NAME = "gemma3n:e2b-it-q4_K_M"
DYS_MODEL  = "models/dysarthria.tflite"
FACE_MODEL = "models/droopy_face.tflite"
with st.sidebar:
    st.header("⚙️ Settings")
    rppg_duration = st.slider("rPPG capture duration (s)", 5, 15, 10)
    audio_duration = st.slider("Audio record duration (s)", 1, 5, 3)

# ---- Helpers ----
@st.cache_resource
def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    return interp, inp, out

def run_tflite(interp, inp, outp, data):
    arr = np.array(data, dtype=inp[0]["dtype"]).reshape(inp[0]["shape"])
    interp.set_tensor(inp[0]["index"], arr)
    interp.invoke()
    out = np.squeeze(interp.get_tensor(outp[0]["index"]))
    return float(out) if out.ndim==0 else float(np.mean(out))

def capture_rppg(duration):
    cap = cv2.VideoCapture(0)
    t0=time.time(); frames=[]
    while time.time()-t0 < duration:
        ret,img=cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)<2: return None,None
    fd=mp.solutions.face_detection.FaceDetection()
    green=[]
    for f in frames:
        res=fd.process(f);h,w,_=f.shape
        if res.detections:
            b=res.detections[0].location_data.relative_bounding_box
            x1,y1=int(b.xmin*w),int(b.ymin*h)
            x2,y2=x1+int(b.width*w),y1+int(b.height*h)
            roi=f[y1:y2,x1:x2]; green.append(np.mean(roi[:,:,1]))
        else:
            green.append(np.nan)
    fd.close()
    sig=np.array(green); idx=np.flatnonzero(~np.isnan(sig))
    sig[np.isnan(sig)]=np.interp(np.flatnonzero(np.isnan(sig)),idx,sig[idx])
    fs=len(sig)/duration; b,a=butter(3,[0.7/(fs/2),4/(fs/2)],btype="band")
    filt=filtfilt(b,a,sig); peaks,_=find_peaks(filt,distance=fs*0.5)
    bpm=len(peaks)*(60/duration)
    fig,ax=plt.subplots(figsize=(4,2)); t=np.linspace(0,duration,len(filt))
    ax.plot(t,filt); ax.plot(t[peaks],filt[peaks],'ro'); ax.set_xticks([]);ax.set_yticks([])
    ax.set_title("Heartbeat Waveform")
    return bpm,fig

def prep_audio(bytes_data,shape,plot_col):
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False)
    tmp.write(bytes_data); tmp.close()
    audio=AudioSegment.from_file(tmp.name).set_frame_rate(16000).set_channels(1)
    os.unlink(tmp.name)
    samples=np.array(audio.get_array_of_samples(),dtype=np.float32)/32768.0
    N=shape[1]; arr=np.pad(samples,(0,max(0,N-len(samples))))[:N].reshape(shape)
    fig,ax=plt.subplots(figsize=(4,1)); librosa.display.waveshow(samples,sr=16000,ax=ax)
    ax.set_xticks([]);ax.set_yticks([]); plot_col.pyplot(fig,use_container_width=True)
    return arr

def prep_image(bytes_data,shape):
    _,H,W,C=shape; img=Image.open(io.BytesIO(bytes_data)).convert("RGB").resize((W,H))
    arr=np.array(img,dtype=np.float32)/255.0
    if C==1: arr=arr.mean(-1,keepdims=True)
    return arr.reshape(shape)

def ask_gemma_http(prompt):
    payload={"model":MODEL_NAME,"messages":[{"role":"user","content":prompt}],"max_tokens":200}
    r=requests.post(OLLAMA_URL,json=payload); r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ---- Mode Selection ----
mode = st.radio("Select Triage Mode:",["Live Triage","Upload Triage"])

if mode=="Live Triage":
    if st.button(f"✅ Start Live Triage (look & speak {max(rppg_duration,audio_duration)}s)"):
        with st.spinner("Capturing video for rPPG..."):
            bpm,fig=capture_rppg(rppg_duration)
        if bpm is None:
            st.error("rPPG capture failed."); st.stop()
        st.metric("❤️ BPM",f"{bpm:.1f}")
        st.pyplot(fig,use_container_width=True)
        with st.spinner("Recording audio..."):
            audio_bytes = audiorecorder(f"🎙 Speak now ({audio_duration}s)",audio_duration)
        if not audio_bytes:
            st.error("Audio record failed."); st.stop()
        tmpbuf=io.BytesIO(); audio_bytes.export(tmpbuf,format='wav') if isinstance(audio_bytes,AudioSegment) else tmpbuf.write(audio_bytes)
        wav=tmpbuf.getvalue(); st.audio(wav,format="audio/wav")
        # infer
        dys,di,do=load_tflite(DYS_MODEL); face,fi,fo=load_tflite(FACE_MODEL)
        slur=run_tflite(dys,di,do,prep_audio(wav,di[0]["shape"],st))
        droop=run_tflite(face,fi,fo,prep_image(wav if False else wav,fi[0]["shape"]))
        cols=st.columns(3); cols[0].metric("🔊 Slur",f"{slur:.2f}"); cols[1].metric("😐 Droop",f"{droop:.2f}"); cols[2].metric("❤️ BPM",f"{bpm:.1f}")
        prompt=(f"Audio slur score: {slur:.2f}, face droop score: {droop:.2f}, pulse: {bpm:.1f} bpm.\n"+
                "You are an expert stroke triage specialist. Based on these three vital signs, integrate into overall stroke likelihood, provide 3 bullet-point findings including heart rate, then finish with a single, clear recommendation.")
        advice=ask_gemma_http(prompt)
        if "CALL 911" in advice.upper(): st.error(advice,icon="🚨")
        else: st.success(advice,icon="✅")

else:  # Upload Triage
    st.subheader("📥 Upload Triage Inputs")
    img_u = st.file_uploader("Upload image (selfie)",type=["png","jpg","jpeg"])
    audio_u = st.file_uploader("Upload audio",type=["wav","mp3","m4a"])
    if st.button("🏷️ Run Upload Triage"):
        if not img_u or not audio_u:
            st.error("Please upload both image and audio."); st.stop()
        img_b=img_u.read(); st.image(img_b,use_column_width=True)
        aud_b=audio_u.read(); st.audio(aud_b,format="audio/wav")
        # infer
        bpm,fig=capture_rppg(1)  # optional quick blink BPM or dummy
        dys,di,do=load_tflite(DYS_MODEL); face,fi,fo=load_tflite(FACE_MODEL)
        slur=run_tflite(dys,di,do,prep_audio(aud_b,di[0]["shape"],st))
        droop=run_tflite(face,fi,fo,prep_image(img_b,fi[0]["shape"]))
        cols=st.columns(3); cols[0].metric("🔊 Slur",f"{slur:.2f}"); cols[1].metric("😐 Droop",f"{droop:.2f}"); cols[2].metric("❤️ BPM",f"{bpm:.1f}")
        prompt=(f"Audio slur score: {slur:.2f}, face droop score: {droop:.2f}, pulse: {bpm:.1f} bpm.\n"+
                "You are an expert stroke triage specialist. Based on these three vital signs, integrate into overall stroke likelihood, provide 3 bullet-point findings including heart rate, then finish with a single, clear recommendation.")
        advice=ask_gemma_http(prompt)
        if "CALL 911" in advice.upper(): st.error(advice,icon="🚨")
        else: st.success(advice,icon="✅")


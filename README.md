# StrokeSentry

**Realtime, Offline Stroke Detection & Guidance**

StrokeSentry is a mobile‑first, privacy‑first app that uses on‑device computer vision to detect facial droop—one of the earliest signs of stroke—and combines it with Google’s Gemma 3n LLM to deliver step‑by‑step, offline guidance in any language.

## 🔥 Why StrokeSentry?

* **FAST acronym in action**: Face droop detection + Gemma 3n guidance covers Face, Arm, Speech, and Time in one tap.
* **Offline & private**: All processing (vision + LLM) runs locally—no internet, no data leaks.
* **Ultra‑fast**: TFLite + XNNPACK + NNAPI delegate on modern phones gives <200 ms inference.
* **Life‑saving**: Early recognition and clear instructions can make the difference in the critical 3‑hour treatment window.

## 🚀 Features

* **Image‑only droop detector** via a lightweight CNN (MobileNetV2 → TFLite).
* **Buzzer for audio branch** (optional future extension for slurred speech detection).
* **Actionable guidance** with Gemma 3n, fully offline and multilingual.
* **Streamlit proof‑of‑concept** and Android app powered by Google AI Edge SDK.

## 🎯 Roadmap

1. **Image POC**: TFLite droop inference + Gemma prompt integration.
2. **Audio extension** (slur detection) using UA‑Speech or TORGO datasets (future hack).
3. **Mobile packaging**: Android app with MediaPipe Tasks GenAI + TFLite.
4. **Fine‑tuning**: LoRA adapter on local EMS protocols via Unsloth.

## 🛠️ Getting Started

### Prerequisites

* Python 3.8+ (for POC)
* Android Studio (for mobile)
* TensorFlow 2.x, Streamlit, Git

### 1. Clone & Setup POC

```bash
git clone https://github.com/<you>/StrokeSentry.git
cd StrokeSentry
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training Notebook (optional)

```bash
# in notebooks/train_stroke.ipynb
# trains stroke_model.h5 → stroke_model.tflite
jupyter lab
```

### 3. Run Streamlit Demo

```bash
streamlit run src/app/app.py
```

Upload a face photo and get:

* **Prediction**: Stroke vs. No Stroke
* **Advice**: Gemma 3n’s immediate next steps

### 4. Android App (Google AI Edge)

* Drop `stroke_model.tflite` and `gemma-3n.bin` into `app/src/main/assets`
* Open `app/` in Android Studio
* Build & run on device or emulator

## 🎓 Audio Extension (Future)

> **Datasets:** UA‑Speech, TORGO (dysarthric speakers) for slurred‑speech detection
> **Model:** TF‑based spectrogram‑CNN → TFLite
> **Gemma fusion:** image + audio confidence → richer triage

## 🤝 Contributing

Pull requests welcome! Please open issues for bugs or feature requests.

## 📄 License

[Apache 2.0](LICENSE)

---

*StrokeSentry* was built with ❤️ for Google Gemma 3n Hackathon. Keep your loved ones safe—every second counts.

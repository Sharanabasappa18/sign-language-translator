# AI Sign Language to Text Translator

Real-time web app that uses your webcam, MediaPipe hand tracking, and a machine-learning classifier to recognize sign gestures and translate them into text. Supports **English**, **Kannada**, **Hindi**, and **Tamil**, with optional text-to-speech.

## Features

- Live webcam feed with hand landmark overlay
- ML + rule-based gesture recognition
- Stability filter to ignore casual hand movement
- Multi-language translation output
- Browser text-to-speech (server gTTS fallback)
- Start / Stop camera, Clear text, language selector
- Extensible project layout for new gestures and languages

## Project structure

```
sign-language-translator/
├── backend/           # FastAPI server + WebSocket
├── frontend/          # Web UI
├── ml_model/          # Classifier, training, sample collection
├── training_dataset/  # Landmark feature samples (.npy)
├── utils/             # Landmarks, languages, TTS helpers
└── config.py
```

## Quick start

### 1. Create a virtual environment

```bash
cd sign-language-translator
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the hand model and bootstrap the ML classifier

```bash
python scripts/download_model.py
python -m ml_model.generate_synthetic --train
```

For better accuracy, collect real samples with your webcam:

```bash
python -m ml_model.collect_samples --gesture hello --count 50
python -m ml_model.train
```

### 4. Run the app

```bash
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser and allow camera access.

## Demo gestures

| Gesture   | Hand shape (approx.)      |
|-----------|---------------------------|
| hello     | Open palm                 |
| yes       | Thumbs up                 |
| no        | Closed fist               |
| one       | Index finger up           |
| two       | Index + middle up         |
| three     | Index + middle + ring     |
| love      | Index + pinky (ILY)       |
| please    | Closed hand               |
| thank_you | Four fingers up           |

Hold each sign **steady for ~1 second** so the stabilizer accepts it.

## Adding a new gesture

1. Add translations in `utils/language_maps.py`
2. Collect samples: `python -m ml_model.collect_samples --gesture my_sign --count 40`
3. Retrain: `python -m ml_model.train`
4. Refresh the web app

## Adding a new language

1. Add the language code to `config.py` → `LANGUAGES`
2. Add translations for each gesture in `utils/language_maps.py`
3. Add a `<option>` in `frontend/index.html`
4. Add TTS code in `frontend/js/app.js` → `TTS_CODES`

## Tech stack

- **Python** — FastAPI, OpenCV, MediaPipe, scikit-learn
- **Frontend** — HTML, CSS, JavaScript (WebSocket + WebRTC)
- **ML** — RandomForest on normalized hand landmarks

## Notes

- This is a **demo / learning framework**. Production sign-language systems need large, language-specific datasets and professional linguistic review.
- Recognition quality improves significantly when you replace synthetic data with real collected samples.
- Tamil/Kannada/Hindi TTS depends on browser support; Chrome generally works best.

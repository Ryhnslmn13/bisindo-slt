# BISINDO Sign Language Translation — Streamlit App

Modular refactor of `SLTpipeline__1_.ipynb` into a runnable Streamlit demo.

---

## Project structure

```
slt_app/
├── app.py                  # Streamlit entry-point
├── config.py               # All paths / hyper-params (edit here)
├── verify_parity.py        # Proves app ≡ notebook inference
├── requirements.txt
├── models/                 # ← put your artifacts here
│   ├── best_model.pth
│   ├── vocab.pkl
│   ├── hand_landmarker.task
│   └── pose_landmarker_lite.task
└── src/
    ├── __init__.py
    ├── preprocessing.py    # extract_keypoints, temporal_resample,
    │                       # compute_velocity, normalize, process_video
    ├── model.py            # Encoder, BahdanauAttention, Decoder, Seq2Seq
    └── inference.py        # greedy_decode, predict_video_with_confidence
```

---

## Setup

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Copy model artifacts

```
models/best_model.pth    ← from your Colab / training run
models/vocab.pkl         ← from your Colab / training run
```

### 3 — Download MediaPipe models

```bash
mkdir -p models
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task \
     -O models/pose_landmarker_lite.task

wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
     -O models/hand_landmarker.task
```

### 4 — Verify notebook parity (critical)

```bash
# Without ground truth
python verify_parity.py path/to/test_video.mp4

# With ground truth (proves exact match)
python verify_parity.py path/to/test_video.mp4 "aku suka kamu"
```

Expected output:
```
[1] Checking model artifacts … ✓ ✓ ✓ ✓
[2] Loading vocabulary … ✓ Vocab OK
[3] Loading model … ✓ Model loaded (strict=True)
[4] Preprocessing … ✓ Preprocessing shape chain OK
[5] Running inference …
    Predicted : 'aku suka kamu'
    Confidence: 94.3%
[6] ✅ EXACT MATCH — inference is identical to notebook
```

### 5 — Launch the app

```bash
streamlit run app.py
```

---

## Why this refactor preserves notebook behaviour

The previous refactor broke silently because:
- preprocessing steps were reordered
- the demo cell's **second** `temporal_resample` was dropped
- vocab/decoder mapping was rebuilt instead of loaded from `vocab.pkl`

This refactor fixes all three:

| Risk | Mitigation |
|------|------------|
| Preprocessing order | `src/preprocessing.py` is a direct copy; functions are not reordered |
| Double resample | `inference.py → predict_video_with_confidence` calls `temporal_resample` again after `process_video`, exactly as the notebook demo cell does |
| Vocab mapping | `load_vocab()` reads the **exact** `vocab.pkl` the notebook saved; never rebuilt |
| Model weights | `load_model()` uses `strict=True` — any key mismatch raises immediately |
| Tensor ops | All `torch` operations are copied verbatim |

---

## Changing paths

Edit `config.py` or set environment variables:

```bash
export CHECKPOINT_PATH=/path/to/best_model.pth
export VOCAB_PATH=/path/to/vocab.pkl
export HAND_MODEL_PATH=/path/to/hand_landmarker.task
export POSE_MODEL_PATH=/path/to/pose_landmarker_lite.task
streamlit run app.py
```

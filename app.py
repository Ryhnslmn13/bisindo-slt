"""
app.py — BISINDO Sign Language Translation Demo
================================================
Run with:  streamlit run app.py

Connects to the EXACT inference pipeline from the notebook.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Make src/ importable regardless of where the user launches from
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH,
    HAND_MODEL_PATH,
    POSE_MODEL_PATH,
    VOCAB_PATH,
    MAX_DECODE_LEN,
)
from src.inference import (
    load_model,
    load_vocab,
    predict_video_with_confidence,
    debug_preprocessing,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BISINDO SLT — Sign Language Translator",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#1a1a2e; }
    .subtitle    { font-size:1.05rem; color:#444; margin-bottom:1.5rem; }
    .result-box  { background:#f0f7ff; border-left:5px solid #2563eb;
                   padding:1.2rem 1.5rem; border-radius:8px; margin-top:1rem; }
    .result-text { font-size:1.8rem; font-weight:700; color:#1a1a2e; }
    .conf-label  { font-size:0.9rem; color:#555; margin-top:0.4rem; }
    .warn-box    { background:#fff7ed; border-left:5px solid #f97316;
                   padding:1rem 1.2rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached model + vocab loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_resources():
    """Load vocab and model once; cache for the lifetime of the session."""
    missing = []
    for label, path in [
        ("vocab.pkl",               VOCAB_PATH),
        ("best_model.pth",          CHECKPOINT_PATH),
        ("hand_landmarker.task",    HAND_MODEL_PATH),
        ("pose_landmarker_lite.task", POSE_MODEL_PATH),
    ]:
        if not os.path.isfile(path):
            missing.append(f"  • {label}  →  {path}")

    if missing:
        return None, None, None, missing

    word2idx, idx2word = load_vocab(VOCAB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, vocab_size=len(word2idx), device=device)
    return model, word2idx, idx2word, []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://flagcdn.com/id.svg", width=40)
    st.markdown("## 🤟 BISINDO SLT")
    st.markdown(
        "**B**ahasa **I**syarat **I**ndonesia sign-language translator "
        "powered by a CNN-GRU encoder + GRU-Bahdanau decoder."
    )
    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        "1. MediaPipe Pose + Hands\n"
        "2. Temporal resampling (→ 100 frames)\n"
        "3. Velocity concatenation (→ 450 features)\n"
        "4. Z-score normalisation\n"
        "5. Seq2Seq greedy decoding"
    )
    st.divider()
    st.markdown("### Model files expected")
    st.code(
        "models/\n"
        "├── best_model.pth\n"
        "├── vocab.pkl\n"
        "├── hand_landmarker.task\n"
        "└── pose_landmarker_lite.task",
        language="text",
    )
    debug_mode = st.checkbox("🔬 Show debug prints", value=False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="main-title">🤟 BISINDO Sign Language Translator</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Upload a video of a BISINDO gesture and the model '
    'will translate it into Indonesian text.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
with st.spinner("Loading model and vocabulary…"):
    model, word2idx, idx2word, missing_files = load_resources()

if missing_files:
    st.error("**Model artifacts not found.** Please place the following files:")
    for m in missing_files:
        st.error(m)
    st.markdown(
        """
        **Quick setup:**
        ```bash
        mkdir -p models
        # Copy best_model.pth and vocab.pkl from your training environment
        # Download MediaPipe models:
        wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task -O models/pose_landmarker_lite.task
        wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -O models/hand_landmarker.task
        ```
        """
    )
    st.stop()

device = next(model.parameters()).device
st.success(
    f"✅ Model loaded — vocab size: **{len(word2idx)}** tokens | "
    f"device: **{device}**"
)

# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 📤 Upload Sign Language Video")

uploaded = st.file_uploader(
    "Drop an MP4 / MOV / AVI video here",
    type=["mp4", "mov", "avi", "mkv"],
    help="The video should contain a single BISINDO sentence gesture.",
)

if uploaded is not None:
    # Preview
    st.markdown("#### 🎬 Uploaded Video Preview")
    st.video(uploaded)

    # Save to a temp file (OpenCV needs a real path)
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.divider()
    st.markdown("### 🔍 Run Inference")

    run_btn = st.button("▶ Translate", type="primary", use_container_width=True)

    if run_btn:
        # ----------------------------------------------------------------
        # Inference with progress feedback
        # ----------------------------------------------------------------
        progress = st.progress(0, text="Initialising…")
        status   = st.empty()

        try:
            status.info("⚙️ Extracting landmarks from video frames…")
            progress.progress(20, text="Extracting landmarks…")

            # Optional debug
            if debug_mode:
                with st.expander("🔬 Preprocessing debug output", expanded=True):
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        debug_preprocessing(tmp_path, HAND_MODEL_PATH, POSE_MODEL_PATH)
                    st.code(buf.getvalue(), language="text")

            progress.progress(55, text="Running encoder…")
            status.info("🧠 Running Seq2Seq model…")

            t0 = time.time()
            sentence, conf, per_token_conf, raw_token_ids = predict_video_with_confidence(
                video_path      = tmp_path,
                model           = model,
                word2idx        = word2idx,
                idx2word        = idx2word,
                device          = device,
                hand_model_path = HAND_MODEL_PATH,
                pose_model_path = POSE_MODEL_PATH,
                max_len         = MAX_DECODE_LEN,
            )
            elapsed = time.time() - t0

            progress.progress(100, text="Done!")
            status.empty()

            # ----------------------------------------------------------------
            # Results
            # ----------------------------------------------------------------
            st.markdown("### 📝 Prediction")

            if sentence.strip():
                st.markdown(
                    f'<div class="result-box">'
                    f'<div class="result-text">{sentence}</div>'
                    f'<div class="conf-label">Overall confidence: '
                    f'<strong>{conf*100:.1f}%</strong> &nbsp;|&nbsp; '
                    f'Inference time: <strong>{elapsed:.2f}s</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="warn-box">⚠️ The model could not generate a '
                    'prediction.  Try a clearer video with better lighting.</div>',
                    unsafe_allow_html=True,
                )

            # ----------------------------------------------------------------
            # Confidence visualisation
            # ----------------------------------------------------------------
            if per_token_conf:
                st.markdown("#### 📊 Per-token Confidence")

                # Decode token labels for the bar chart
                # (re-run decoding with word labels — lightweight)
                token_labels = []
                _idx = [word2idx["<SOS>"]]
                _hidden_ref = None  # we just need labels; use sentence words
                words_in_sentence = sentence.split()
                for i, c in enumerate(per_token_conf):
                    label = words_in_sentence[i] if i < len(words_in_sentence) else "<EOS>"
                    token_labels.append(label)

                import pandas as pd
                chart_df = pd.DataFrame({
                    "Token": token_labels,
                    "Confidence": [c * 100 for c in per_token_conf],
                })
                st.bar_chart(chart_df.set_index("Token"), height=260)

                # Confidence gauge
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean confidence",  f"{conf*100:.1f}%")
                col2.metric("Min token conf",   f"{min(per_token_conf)*100:.1f}%")
                col3.metric("Max token conf",   f"{max(per_token_conf)*100:.1f}%")

            # ----------------------------------------------------------------
            # Raw token-ID debug panel (only in debug mode)
            # ----------------------------------------------------------------
            if debug_mode:
                with st.expander("🔬 Raw token-ID trace (for notebook parity)", expanded=True):
                    st.markdown("**Raw token ID sequence** (copy this; compare with notebook output):")
                    st.code(str(raw_token_ids), language="text")

                    SOS_ID = word2idx["<SOS>"]
                    EOS_ID = word2idx["<EOS>"]
                    PAD_ID = word2idx["<PAD>"]

                    rows = []
                    for i, tid in enumerate(raw_token_ids):
                        tok = idx2word.get(tid, f"[UNK:{tid}]")
                        note = ""
                        if tid == SOS_ID: note = "<SOS>"
                        elif tid == EOS_ID: note = "<EOS> — stop"
                        elif tid == PAD_ID: note = "<PAD> unexpected"
                        rows.append({"step": i, "token_id": tid, "token": tok, "note": note})

                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    st.markdown("**To compare with notebook**, paste this cell into your notebook and run on the same video:")
                    st.code(
                        "import torch, torch.nn.functional as F\n\n"
                        "def notebook_raw_ids(video_path, max_len=30):\n"
                        "    sequence = process_video(video_path)\n"
                        "    sequence = temporal_resample(sequence, target_len=100)\n"
                        "    src = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)\n"
                        "    model.eval()\n"
                        "    with torch.no_grad():\n"
                        "        enc_out, hidden = model.encoder(src)\n"
                        "    ids = [word2idx['<SOS>']]\n"
                        "    for _ in range(max_len):\n"
                        "        t = torch.tensor([ids[-1]], dtype=torch.long).to(device)\n"
                        "        with torch.no_grad():\n"
                        "            out, hidden = model.decoder(t, hidden, enc_out)\n"
                        "        p = F.softmax(out, dim=1)\n"
                        "        pid = p.argmax(1).item()\n"
                        "        ids.append(pid)\n"
                        "        if pid == word2idx['<EOS>']: break\n"
                        "    print('NOTEBOOK RAW IDS:', ids)\n"
                        "    return ids\n\n"
                        "notebook_raw_ids('your_test_video.mp4')",
                        language="python",
                    )
                    st.info(
                        f"App raw IDs:  `{raw_token_ids}`\n\n"
                        "If notebook prints different IDs → preprocessing or weights diverged.\n"
                        "If IDs match but text differs → vocab/idx2word mapping bug."
                    )

        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"**Inference failed:** {exc}")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc(), language="text")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<small>BISINDO Sign Language Translation Pipeline — "
    "CNN-GRU Encoder · GRU-Bahdanau Decoder · MediaPipe Landmarks</small>",
    unsafe_allow_html=True,
)

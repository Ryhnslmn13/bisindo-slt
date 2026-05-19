import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CHECKPOINT_PATH,
    HAND_MODEL_PATH,
    MAX_DECODE_LEN,
    POSE_MODEL_PATH,
    VOCAB_PATH,
)
from src.inference import load_model, load_vocab, predict_video_with_confidence


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BISINDO Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hero banner ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 18px;
    padding: 2.6rem 3rem 2.2rem;
    margin-bottom: 2rem;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 80% 50%,
        rgba(99,102,241,.25) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.7rem;
    font-weight: 900;
    letter-spacing: -0.8px;
    margin: 0 0 0.55rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.05rem;
    opacity: 0.80;
    margin: 0 0 1.1rem;
    max-width: 600px;
    line-height: 1.6;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.chip {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 20px;
    padding: 0.22rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* ── Section labels ──────────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.5rem;
}

/* ── Translation result card ─────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #eff6ff 0%, #e0f2fe 100%);
    border: 1.5px solid #bfdbfe;
    border-radius: 16px;
    padding: 1.7rem 1.9rem 1.4rem;
}
.result-sentence {
    font-size: 2rem;
    font-weight: 800;
    color: #1e3a5f;
    line-height: 1.3;
    margin-bottom: 0.9rem;
    word-break: break-word;
}
.result-meta {
    font-size: 0.87rem;
    color: #64748b;
    margin-top: 0.6rem;
}

/* ── Confidence badges ───────────────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 24px;
    padding: 0.3rem 1.1rem;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.badge-green  { background:#dcfce7; color:#166534; border:1.5px solid #86efac; }
.badge-yellow { background:#fef9c3; color:#854d0e; border:1.5px solid #fde047; }
.badge-red    { background:#fee2e2; color:#991b1b; border:1.5px solid #fca5a5; }

/* ── Empty-state placeholder ─────────────────────────────────────────── */
.empty-state {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    color: #94a3b8;
}
.empty-icon { font-size: 2.8rem; margin-bottom: 0.7rem; }
.empty-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.35rem;
               color: #64748b; }
.empty-hint  { font-size: 0.9rem; }

/* ── Warning card ────────────────────────────────────────────────────── */
.warn-card {
    background: #fff7ed;
    border: 1.5px solid #fed7aa;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #92400e;
    font-size: 0.97rem;
}

/* ── Sidebar architecture steps ──────────────────────────────────────── */
.arch-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin-bottom: 0.55rem;
}
.arch-icon {
    font-size: 1.15rem;
    flex-shrink: 0;
    margin-top: 0.05rem;
}
.arch-name  { font-weight: 700; font-size: 0.9rem; line-height: 1.2; }
.arch-desc  { font-size: 0.78rem; color: #64748b; }
.arch-connector {
    margin-left: 0.55rem;
    border-left: 2px dashed #e2e8f0;
    height: 10px;
    margin-bottom: -4px;
}

/* ── Confidence legend ───────────────────────────────────────────────── */
.legend {
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    gap: 1rem;
    margin-top: 0.3rem;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 3px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loading  — @st.cache_resource, loads once
# INFERENCE LOGIC UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    missing = []
    for label, path in [
        ("vocab.pkl",                 VOCAB_PATH),
        ("best_model.pth",            CHECKPOINT_PATH),
        ("hand_landmarker.task",      HAND_MODEL_PATH),
        ("pose_landmarker_lite.task", POSE_MODEL_PATH),
    ]:
        if not os.path.isfile(path):
            missing.append(f"**{label}** → `{path}`")
    if missing:
        return None, None, None, missing
    word2idx, idx2word = load_vocab(VOCAB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(CHECKPOINT_PATH, vocab_size=len(word2idx), device=device)
    return model, word2idx, idx2word, []


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_video_meta(path: str) -> dict:
    cap         = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 1
    duration    = frame_count / fps
    cap.release()
    return {"frames": frame_count, "fps": round(fps, 1), "duration": round(duration, 2)}


def conf_badge_html(conf: float) -> str:
    pct = conf * 100
    if pct >= 70:
        cls, label = "badge-green",  f"✓ {pct:.1f}%  High confidence"
    elif pct >= 40:
        cls, label = "badge-yellow", f"~ {pct:.1f}%  Medium confidence"
    else:
        cls, label = "badge-red",    f"✗ {pct:.1f}%  Low confidence"
    return f'<span class="badge {cls}">{label}</span>'


def bar_color(c: float) -> str:
    if c >= 0.70: return "#22c55e"
    if c >= 0.40: return "#eab308"
    return "#ef4444"


def draw_landmark_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    """Draw pose + hand landmarks on a BGR frame for display only."""
    import mediapipe as mp
    mp_draw  = mp.solutions.drawing_utils
    mp_pose  = mp.solutions.pose
    mp_hands = mp.solutions.hands

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out       = frame_bgr.copy()

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
        res = pose.process(frame_rgb)
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 220, 90),  thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 160, 60),  thickness=2),
            )

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=0.3) as hands:
        res = hands.process(frame_rgb)
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    out, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255, 120, 30), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(200,  80,  0), thickness=2),
                )
    return out


def sample_frames(video_path: str, n: int = 12, overlay: bool = False):
    """Return up to n evenly-spaced RGB frames (with optional landmark overlay)."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, min(n, total), dtype=int)
    out   = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        if overlay:
            frame = draw_landmark_overlay(frame)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 BISINDO SLT")
    st.caption("Bahasa Isyarat Indonesia — Sign Language Translator")
    st.divider()

    # Architecture summary
    st.markdown("**Model Architecture**")
    arch = [
        ("📥", "Input",          "Raw video frames (BGR)"),
        ("👁️", "MediaPipe",      "33 pose + 21×2 hand keypoints"),
        ("📐", "Feature Eng.",   "Resample → Velocity → Z-norm"),
        ("🔷", "CNN-GRU",        "1-D conv + GRU encoder (256 h)"),
        ("🎯", "Attention",      "Bahdanau soft-attention"),
        ("📤", "GRU Decoder",    "Greedy decoding → Indonesian"),
    ]
    for i, (icon, name, desc) in enumerate(arch):
        st.markdown(
            f'<div class="arch-row">'
            f'  <span class="arch-icon">{icon}</span>'
            f'  <span><div class="arch-name">{name}</div>'
            f'        <div class="arch-desc">{desc}</div></span>'
            f'</div>'
            + (f'<div class="arch-connector"></div>' if i < len(arch) - 1 else ""),
            unsafe_allow_html=True,
        )

    st.divider()

    # Visualisation toggles
    st.markdown("**Visualisations**")
    show_overlay    = st.toggle("🦴 Landmark overlay on frames", value=False)
    show_conf_chart = st.toggle("📊 Per-token confidence chart",  value=True)
    show_frames     = st.toggle("🎞️ Sample frame grid",           value=False)

    st.divider()
    st.markdown("**Expected model files**")
    st.code(
        "models/\n"
        "├── best_model.pth\n"
        "├── vocab.pkl\n"
        "├── hand_landmarker.task\n"
        "└── pose_landmarker_lite.task",
        language="text",
    )


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🤟 BISINDO Sign Language Translator</div>
  <p class="hero-sub">
    Upload a sign language video and the model translates it into Indonesian text —
    end-to-end, powered by a CNN-GRU encoder and Bahdanau-attention decoder.
  </p>
  <div class="hero-chips">
    <span class="chip">MediaPipe</span>
    <span class="chip">CNN · GRU</span>
    <span class="chip">Bahdanau Attention</span>
    <span class="chip">Seq2Seq</span>
    <span class="chip">PyTorch</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading model weights…"):
    model, word2idx, idx2word, missing_files = load_resources()

if missing_files:
    st.error("**Model artifacts not found.** Place these files before running:")
    for m in missing_files:
        st.error(m)
    st.code(
        "mkdir -p models\n"
        "# Copy best_model.pth and vocab.pkl from your training run\n"
        "wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task "
        "-O models/pose_landmarker_lite.task\n"
        "wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task "
        "-O models/hand_landmarker.task",
        language="bash",
    )
    st.stop()

device = next(model.parameters()).device
st.success(
    f"✅ Model ready — vocab: **{len(word2idx)}** tokens · device: **{device}**",
    icon="✅",
)


# ─────────────────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT  (upload left · results right)
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — upload, preview, metrics, frame grid, translate button
# ══════════════════════════════════════════════════════════════════════════════
with left_col:

    # Upload
    st.markdown('<p class="section-label">Upload video</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a video here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Single BISINDO sentence gesture. Good lighting and visible hands required.",
        label_visibility="collapsed",
    )

    if uploaded is None:
        # Friendly empty state inside left column
        st.markdown("""
        <div class="empty-state" style="margin-top:1.5rem">
          <div class="empty-icon">🎬</div>
          <div class="empty-title">No video uploaded</div>
          <div class="empty-hint">Drag & drop an MP4, MOV or AVI file above</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Persist temp file across reruns so the path stays valid
        if ("tmp_path" not in st.session_state or
                st.session_state.get("uploaded_name") != uploaded.name):
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                st.session_state["tmp_path"]      = tmp.name
                st.session_state["uploaded_name"] = uploaded.name
                st.session_state.pop("result", None)   # clear stale result

        tmp_path = st.session_state["tmp_path"]

        # ── Inline video preview ──────────────────────────────────────────────
        st.markdown(
            '<p class="section-label" style="margin-top:1.2rem">Preview</p>',
            unsafe_allow_html=True,
        )
        st.video(uploaded)

        # ── Three video metrics ───────────────────────────────────────────────
        meta = get_video_meta(tmp_path)
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames",   meta["frames"])
        m2.metric("FPS",      meta["fps"])
        m3.metric("Duration", f"{meta['duration']}s")

        # ── Optional sample-frame grid ────────────────────────────────────────
        if show_frames:
            st.markdown(
                '<p class="section-label" style="margin-top:1.2rem">Sample Frames</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Sampling frames…"):
                frames = sample_frames(tmp_path, n=12, overlay=show_overlay)
            cols = st.columns(4)
            for i, fr in enumerate(frames):
                cols[i % 4].image(fr, use_container_width=True, caption=f"f{i+1}")

        # ── Translate button ──────────────────────────────────────────────────
        st.markdown("")
        run_btn = st.button("▶  Translate", type="primary", use_container_width=True)

        # ── INFERENCE  (logic 100% unchanged) ────────────────────────────────
        if run_btn:
            st.session_state.pop("result", None)

            try:
                with st.status("Running inference pipeline…", expanded=True) as status_box:
                    st.write("📥 Loading video and extracting frames…")
                    t0 = time.time()

                    st.write("👁️ Running MediaPipe — pose + hand landmark extraction…")

                    # ── INFERENCE CALL — UNCHANGED ──────────────────────────
                    sentence, conf, per_token_conf, raw_token_ids = (
                        predict_video_with_confidence(
                            video_path      = tmp_path,
                            model           = model,
                            word2idx        = word2idx,
                            idx2word        = idx2word,
                            device          = device,
                            hand_model_path = HAND_MODEL_PATH,
                            pose_model_path = POSE_MODEL_PATH,
                            max_len         = MAX_DECODE_LEN,
                        )
                    )
                    elapsed = time.time() - t0

                    st.write("📐 Feature engineering — resample, velocity, normalise…")
                    st.write("🔷 CNN-GRU encoder forward pass…")
                    st.write("🎯 Bahdanau attention + greedy GRU decoding…")
                    st.write(f"✅ Completed in **{elapsed:.2f}s**")
                    status_box.update(label="Inference complete!", state="complete")

                # Store in session_state — result persists across reruns
                st.session_state["result"] = {
                    "sentence":       sentence,
                    "conf":           conf,
                    "per_token_conf": per_token_conf,
                    "raw_token_ids":  raw_token_ids,
                    "elapsed":        elapsed,
                    "frames":         meta["frames"],
                }

            except Exception as exc:
                import traceback
                st.error(f"**Inference error:** {exc}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — translation result, confidence, optional charts
# ══════════════════════════════════════════════════════════════════════════════
with right_col:

    st.markdown('<p class="section-label">Translation</p>', unsafe_allow_html=True)

    result = st.session_state.get("result")

    # ── Empty state ───────────────────────────────────────────────────────────
    if result is None:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🤟</div>
          <div class="empty-title">No translation yet</div>
          <div class="empty-hint">Upload a video and press <strong>Translate</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Result ────────────────────────────────────────────────────────────────
    else:
        sentence       = result["sentence"]
        conf           = result["conf"]
        per_token_conf = result["per_token_conf"]
        raw_token_ids  = result["raw_token_ids"]
        elapsed        = result["elapsed"]
        frame_count    = result["frames"]

        SOS_ID = word2idx["<SOS>"]
        EOS_ID = word2idx["<EOS>"]
        PAD_ID = word2idx["<PAD>"]
        n_tokens = len([i for i in raw_token_ids
                        if i not in (SOS_ID, EOS_ID, PAD_ID)])

        # ── Translation card ──────────────────────────────────────────────────
        if sentence.strip():
            st.markdown(
                f'<div class="result-card">'
                f'  <p class="section-label">Indonesian translation</p>'
                f'  <div class="result-sentence">{sentence}</div>'
                f'  {conf_badge_html(conf)}'
                f'  <div class="result-meta">'
                f'    ⏱ {elapsed:.2f}s &nbsp;·&nbsp; '
                f'    {frame_count} input frames &nbsp;·&nbsp; '
                f'    {n_tokens} output token{"s" if n_tokens != 1 else ""}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-card">⚠️ The model could not generate a prediction. '
                'Try a clearer video with better lighting and visible hands.</div>',
                unsafe_allow_html=True,
            )

        # ── Three metrics row ─────────────────────────────────────────────────
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean confidence", f"{conf * 100:.1f}%")
        c2.metric("Input frames",    frame_count)
        c3.metric("Output tokens",   n_tokens)

        # ── Per-token confidence bar chart ────────────────────────────────────
        if show_conf_chart and per_token_conf:
            st.markdown("")
            st.markdown(
                '<p class="section-label">Per-token confidence</p>',
                unsafe_allow_html=True,
            )

            words        = sentence.split()
            token_labels = [
                words[i] if i < len(words) else "<EOS>"
                for i in range(len(per_token_conf))
            ]
            colors = [bar_color(c) for c in per_token_conf]

            try:
                import plotly.graph_objects as go

                fig = go.Figure(go.Bar(
                    x           = token_labels,
                    y           = [c * 100 for c in per_token_conf],
                    marker_color= colors,
                    text        = [f"{c * 100:.1f}%" for c in per_token_conf],
                    textposition= "outside",
                    cliponaxis  = False,
                ))
                fig.update_layout(
                    yaxis      = dict(range=[0, 115], title="Confidence %",
                                      gridcolor="#f1f5f9"),
                    xaxis_title= "Token",
                    height     = 290,
                    margin     = dict(t=24, b=20, l=10, r=10),
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    font          = dict(size=12),
                )
                fig.add_hline(y=70, line_dash="dot", line_color="#22c55e",
                              annotation_text=" 70%", annotation_position="right")
                fig.add_hline(y=40, line_dash="dot", line_color="#eab308",
                              annotation_text=" 40%", annotation_position="right")
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                # Fallback — plain st.bar_chart
                st.bar_chart(
                    pd.DataFrame({"Confidence": [c * 100 for c in per_token_conf]},
                                 index=token_labels),
                    height=260,
                )

            # Colour legend
            st.markdown(
                '<div class="legend">'
                '<span><span class="legend-dot" style="background:#22c55e"></span>'
                '≥ 70% high</span>'
                '<span><span class="legend-dot" style="background:#eab308"></span>'
                '≥ 40% medium</span>'
                '<span><span class="legend-dot" style="background:#ef4444"></span>'
                '< 40% low</span>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT SECTION  — collapsible expander
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
with st.expander("ℹ️ About this project", expanded=False):
    ab_left, ab_right = st.columns([3, 2])

    with ab_left:
        st.markdown("""
**BISINDO** (Bahasa Isyarat Indonesia) is the sign language used by the Deaf
community in Indonesia.  This demo runs a full end-to-end translation pipeline:

1. **MediaPipe landmark extraction** — 33 body-pose + 21 × 2 hand keypoints
   per frame = **225 values per frame**, extracted directly from raw BGR frames.
2. **Temporal resampling** — every video is resampled to exactly **100 frames**
   regardless of original length or frame rate.
3. **Velocity features** — frame-to-frame differences are computed and appended,
   doubling the feature vector to **450 dimensions**.
4. **Z-score normalisation** — per-feature standardisation stabilises scale
   across different signers, cameras, and backgrounds.
5. **CNN-GRU Encoder** — a 1-D convolution layer followed by a GRU (256 hidden
   units) captures spatial-temporal patterns in the landmark sequence.
6. **Bahdanau Attention** — the decoder attends over all 100 encoder timesteps
   at every decoding step, learning which frames are most relevant per token.
7. **GRU Decoder** — greedy token-by-token decoding produces the Indonesian
   sentence one word at a time.
        """)

    with ab_right:
        st.markdown("**Pipeline metrics**")
        st.dataframe(
            pd.DataFrame({
                "Stage":   ["Landmarks", "Resample", "Features", "Encoder", "Decoder"],
                "Output":  ["225-D",     "100 frames","450-D",   "256 hidden", "vocab-D"],
                "Detail":  ["33+42 kp",  "linspace",  "+velocity","CNN+GRU",   "Bahdanau"],
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("")
        st.markdown("**Tech stack**")
        stack = {
            "PyTorch":    "Model training & inference",
            "MediaPipe":  "Landmark detection",
            "OpenCV":     "Video I/O",
            "Streamlit":  "Web interface",
            "Plotly":     "Confidence charts",
        }
        for lib, desc in stack.items():
            st.markdown(f"**{lib}** — {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.82rem;color:#94a3b8;padding-bottom:0.5rem">'
    'BISINDO Sign Language Translation &nbsp;·&nbsp; '
    'CNN-GRU Encoder &nbsp;·&nbsp; GRU + Bahdanau Decoder &nbsp;·&nbsp; MediaPipe Landmarks'
    '</div>',
    unsafe_allow_html=True,
)

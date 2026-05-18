"""
inference.py
------------
EXACT copy of the notebook's inference functions:
    greedy_decode  (used in evaluation)
    predict_video_with_confidence  (used in demo)

The only difference from the notebook is that word2idx / idx2word / model /
device are passed as arguments instead of being globals.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .model import Decoder, Encoder, HIDDEN_DIM, INPUT_DIM, Seq2Seq
from .preprocessing import process_video, temporal_resample


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load word2idx and idx2word from the pickle saved by the notebook."""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab["word2idx"], vocab["idx2word"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(vocab_size: int) -> Seq2Seq:
    """Instantiate the Seq2Seq model with the same hyper-params as training."""
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM)
    decoder = Decoder(vocab_size, HIDDEN_DIM)
    return Seq2Seq(encoder, decoder)


def load_model(checkpoint_path: str,
               vocab_size: int,
               device: torch.device) -> Seq2Seq:
    """Load best_model.pth weights.  Strict=True ensures nothing is silently missing."""
    model = build_model(vocab_size)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# greedy_decode  — IDENTICAL to notebook evaluation function
# ---------------------------------------------------------------------------

def greedy_decode(seq: torch.Tensor,
                  model: Seq2Seq,
                  word2idx: Dict[str, int],
                  idx2word: Dict[int, str],
                  device: torch.device,
                  max_len: int = 20) -> str:
    """Greedy decoding for a single pre-processed sequence tensor (T, F)."""
    model.eval()
    seq = seq.unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(seq)
        input_token = torch.tensor([word2idx["<SOS>"]]).to(device)
        output_sentence: List[str] = []

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()

            if pred_token == word2idx["<EOS>"]:
                break

            word = idx2word.get(pred_token, "")
            if word not in ["<PAD>", "<SOS>"]:
                output_sentence.append(word)

            input_token = torch.tensor([pred_token]).to(device)

    return " ".join(output_sentence)


# ---------------------------------------------------------------------------
# predict_video_with_confidence  — IDENTICAL to notebook demo function
# ---------------------------------------------------------------------------

def predict_video_with_confidence(
    video_path: str,
    model: Seq2Seq,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    device: torch.device,
    hand_model_path: str,
    pose_model_path: str,
    max_len: int = 30,
) -> Tuple[str, float, List[float]]:
    """
    Full inference pipeline, identical to the notebook demo cell.

    Returns
    -------
    sentence     : predicted Indonesian sentence
    sentence_conf: mean per-token softmax confidence (0-1)
    confidences  : per-token confidence list
    """
    model.eval()

    # ---- preprocessing (same order as notebook) ----
    sequence = process_video(video_path, hand_model_path, pose_model_path)
    # NOTE: process_video already calls temporal_resample internally, but the
    # notebook demo cell calls temporal_resample AGAIN after process_video.
    # We replicate that behaviour exactly:
    sequence = temporal_resample(sequence, target_len=100)

    src = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    # ---- encoder ----
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    # ---- greedy decoding with confidence ----
    trg_idx = [word2idx["<SOS>"]]
    confidences: List[float] = []

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_idx[-1]], dtype=torch.long).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        probs = F.softmax(output, dim=1)
        pred_token = probs.argmax(1).item()
        conf = probs.max().item()

        trg_idx.append(pred_token)
        confidences.append(conf)

        if pred_token == word2idx["<EOS>"]:
            break

    tokens = [idx2word[idx] for idx in trg_idx]
    sentence_tokens = [t for t in tokens if t not in ["<SOS>", "<EOS>", "<PAD>"]]

    sentence_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # trg_idx is the raw token-ID list: [SOS, tok1, tok2, ..., EOS]
    # Returned as 4th element so callers can compare against notebook IDs directly.
    return " ".join(sentence_tokens), sentence_conf, confidences, trg_idx


# ---------------------------------------------------------------------------
# Debug helper — prints tensors identical to notebook for verification
# ---------------------------------------------------------------------------

def debug_preprocessing(video_path: str,
                         hand_model_path: str,
                         pose_model_path: str) -> None:
    """Print shape/stats of intermediate arrays to verify notebook parity."""
    import numpy as np
    from .preprocessing import (compute_velocity, normalize,
                                 temporal_resample)
    from .preprocessing import process_video as _proc

    print("=== DEBUG: preprocessing pipeline ===")

    # Re-run step-by-step
    import cv2
    from .preprocessing import extract_keypoints

    cap = cv2.VideoCapture(video_path)
    raw_seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        import cv2 as _cv2
        frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        raw_seq.append(extract_keypoints(frame_rgb, hand_model_path, pose_model_path))
    cap.release()

    raw_seq = np.array(raw_seq)
    print(f"  raw frames    : {raw_seq.shape}")

    after_resample = temporal_resample(raw_seq)
    print(f"  after resample: {after_resample.shape}")

    after_vel = compute_velocity(after_resample)
    print(f"  after velocity: {after_vel.shape}")

    after_norm = normalize(after_vel)
    print(f"  after normalize: {after_norm.shape}  mean={after_norm.mean():.6f}  std={after_norm.std():.6f}")

    # The demo cell re-resamples — replicate it
    final = temporal_resample(after_norm, target_len=100)
    print(f"  after re-resample (demo): {final.shape}")
    print("=== END DEBUG ===")

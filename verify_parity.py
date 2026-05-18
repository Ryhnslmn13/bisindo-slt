"""
verify_parity.py  (v3 — NUMERICAL VALUE LEVEL)
================================================
Performs TRUE numerical parity checking between the notebook pipeline and
the refactored app pipeline.

For every preprocessing stage it prints:
  - first 20 values
  - min / max / mean / std
  - np.allclose() result
  - max absolute difference

Finds the FIRST stage where values diverge.

Usage
-----
# Stage-by-stage numerical comparison (app pipeline only — sanity check):
python verify_parity.py video.mp4

# Full comparison: supply notebook numpy dumps produced by the notebook cell
# printed at the end of this script:
python verify_parity.py video.mp4 \\
    --nb-raw      notebook_raw.npy      \\
    --nb-resampled notebook_resampled.npy \\
    --nb-velocity  notebook_velocity.npy  \\
    --nb-normalized notebook_normalized.npy \\
    --nb-final     notebook_final.npy

# With notebook token IDs:
python verify_parity.py video.mp4 --notebook-ids 1,7,8,9,2

# Dump full vocab for diffing:
python verify_parity.py --dump-vocab
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_PATH, HAND_MODEL_PATH, MAX_DECODE_LEN,
    POSE_MODEL_PATH, VOCAB_PATH,
)
from src.inference import load_model, load_vocab
from src.preprocessing import (
    compute_velocity, extract_keypoints,
    normalize, process_video, temporal_resample,
)


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def hdr(s): print(f"\n{'='*70}\n  {s}\n{'='*70}")
def sub(s): print(f"\n  -- {s}")
def ok(s):  print(f"  [OK]  {s}")
def err(s): print(f"  [ERR] {s}")
def warn(s):print(f"  [WRN] {s}")


def stats(arr: np.ndarray, label: str, n: int = 20) -> None:
    """Print first n values plus min/max/mean/std for a numpy array."""
    flat = arr.flatten().astype(np.float64)
    print(f"\n  {label}:")
    print(f"    shape  = {arr.shape}  dtype={arr.dtype}")
    print(f"    min    = {flat.min():.8f}")
    print(f"    max    = {flat.max():.8f}")
    print(f"    mean   = {flat.mean():.8f}")
    print(f"    std    = {flat.std():.8f}")
    preview = flat[:n]
    print(f"    first {n} values:")
    print(f"    {np.array2string(preview, precision=8, separator=', ')}")


def compare(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str,
            rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    """Compare two arrays numerically, print diagnostics, return True if close."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    if a.shape != b.shape:
        err(f"Shape mismatch: {label_a}={a.shape}  {label_b}={b.shape}")
        return False

    close    = np.allclose(a, b, rtol=rtol, atol=atol)
    abs_diff = np.abs(a - b)
    max_diff = abs_diff.max()
    mean_diff= abs_diff.mean()
    n_wrong  = (~np.isclose(a, b, rtol=rtol, atol=atol)).sum()

    print(f"    np.allclose(rtol={rtol}, atol={atol}) = {close}")
    print(f"    max |diff|  = {max_diff:.2e}")
    print(f"    mean|diff|  = {mean_diff:.2e}")
    print(f"    n elements  = {a.size}   mismatched = {n_wrong}")

    if not close:
        # Show where the worst divergence is
        worst_idx = np.unravel_index(abs_diff.argmax(), a.shape)
        print(f"    worst at index {worst_idx}: "
              f"{label_a}={a[worst_idx]:.8f}  {label_b}={b[worst_idx]:.8f}")
        # First 10 mismatch positions (flattened)
        mismatch_flat = np.where(~np.isclose(a.flatten(), b.flatten(),
                                              rtol=rtol, atol=atol))[0]
        print(f"    first mismatches (flat idx): {mismatch_flat[:10].tolist()}")

    return close


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — raw frame pixel values
# ─────────────────────────────────────────────────────────────────────────────

def check_raw_frames(video_path: str) -> List[np.ndarray]:
    """
    Read raw frames as OpenCV cap.read() returns them (BGR uint8).
    The NOTEBOOK does exactly this — no conversion.
    Returns the list of raw BGR frames.
    """
    hdr("STAGE 0 — raw frame extraction (BGR, no conversion)")

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  total frames read : {len(frames)}")
    print(f"  frame shape       : {frames[0].shape}  dtype={frames[0].dtype}")

    # Print pixel stats for frame 0
    f0 = frames[0].astype(np.float64)
    print(f"  frame 0 pixel stats:")
    print(f"    min={f0.min():.1f}  max={f0.max():.1f}  mean={f0.mean():.4f}")
    print(f"  first 20 pixels (BGR flat): {frames[0].flatten()[:20].tolist()}")

    print()
    print("  NOTE: These frames are passed to extract_keypoints WITHOUT")
    print("  cvtColor conversion — identical to the notebook.")
    print()
    print("  If the notebook was extracting features with cvtColor (RGB),")
    print("  this is the divergence point: MediaPipe receives different pixels.")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — landmark extraction, frame by frame
# ─────────────────────────────────────────────────────────────────────────────

def check_landmarks(frames: List[np.ndarray]) -> np.ndarray:
    """Extract keypoints from every frame, print stats, return raw array."""
    hdr("STAGE 1 — landmark extraction (extract_keypoints per frame)")

    raw_seq = []
    for frame in frames:
        kp = extract_keypoints(frame, HAND_MODEL_PATH, POSE_MODEL_PATH)
        raw_seq.append(kp)
    raw_seq = np.array(raw_seq)

    stats(raw_seq, "raw landmark array", n=20)

    # How many frames had zero pose / zero hands?
    n_zero_pose  = (raw_seq[:, :99] == 0).all(axis=1).sum()
    n_zero_hands = (raw_seq[:, 99:] == 0).all(axis=1).sum()
    print(f"    frames with all-zero pose  : {n_zero_pose}/{len(frames)}")
    print(f"    frames with all-zero hands : {n_zero_hands}/{len(frames)}")

    return raw_seq


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — temporal_resample
# ─────────────────────────────────────────────────────────────────────────────

def check_resample(raw_seq: np.ndarray) -> np.ndarray:
    hdr("STAGE 2 — temporal_resample(target_len=100)")

    idx = np.linspace(0, len(raw_seq) - 1, 100).astype(int)
    print(f"  linspace indices (first 10): {idx[:10].tolist()}")
    print(f"  linspace indices (last 10) : {idx[-10:].tolist()}")

    resampled = temporal_resample(raw_seq, target_len=100)
    stats(resampled, "resampled array", n=20)
    return resampled


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — compute_velocity
# ─────────────────────────────────────────────────────────────────────────────

def check_velocity(resampled: np.ndarray) -> np.ndarray:
    hdr("STAGE 3 — compute_velocity")

    vel_arr = compute_velocity(resampled)
    stats(vel_arr, "velocity array", n=20)

    # Verify row 0 of velocity block is zeros
    vel_block_row0 = vel_arr[0, resampled.shape[1]:]
    print(f"    vel_block row 0 (should be zeros): "
          f"all_zero={np.allclose(vel_block_row0, 0)}")

    return vel_arr


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — normalize
# ─────────────────────────────────────────────────────────────────────────────

def check_normalize(vel_arr: np.ndarray) -> np.ndarray:
    hdr("STAGE 4 — normalize (z-score)")

    norm_arr = normalize(vel_arr)
    stats(norm_arr, "normalized array", n=20)

    # After z-score, global mean should be ~0, std ~1
    print(f"    global mean (should be ~0) : {norm_arr.mean():.8f}")
    print(f"    global std  (should be ~1) : {norm_arr.std():.8f}")

    return norm_arr


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — demo cell re-resample
# ─────────────────────────────────────────────────────────────────────────────

def check_final_resample(norm_arr: np.ndarray) -> np.ndarray:
    hdr("STAGE 5 — demo cell re-resample (temporal_resample again)")
    print("  NOTE: The notebook demo cell calls temporal_resample AGAIN after")
    print("  process_video.  Our inference.py replicates this.  The second")
    print("  resample on an already-100-frame array is a no-op IF linspace")
    print("  produces [0,1,2,...,99] — let's verify:")

    idx = np.linspace(0, 99, 100).astype(int)
    print(f"  idx = {idx.tolist()}")
    identity = np.array_equal(idx, np.arange(100))
    print(f"  idx == [0..99]: {identity}  (no-op = {identity})")

    final = temporal_resample(norm_arr, target_len=100)
    stats(final, "final array (after re-resample)", n=20)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — encoder inputs & outputs
# ─────────────────────────────────────────────────────────────────────────────

def check_encoder(final: np.ndarray, model, device) -> tuple:
    hdr("STAGE 6 — encoder input tensor + encoder output")

    src = torch.tensor(final, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"  src tensor shape : {tuple(src.shape)}  dtype={src.dtype}")

    src_np = src[0].cpu().numpy()
    stats(src_np, "src[0] (input to encoder)", n=20)

    with torch.no_grad():
        enc_out, hidden = model.encoder(src)

    enc_np    = enc_out[0].cpu().numpy()
    hidden_np = hidden[0].cpu().numpy()
    stats(enc_np,    "encoder_outputs[0]", n=20)
    stats(hidden_np, "hidden[0]",          n=20)

    return enc_out, hidden


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — decoding with raw token IDs + top-k
# ─────────────────────────────────────────────────────────────────────────────

def decode_with_ids(enc_out, hidden, model, w2i, i2w, device,
                    max_len: int, topk: int = 5) -> List[int]:
    hdr("STAGE 7 — greedy decoding (raw token IDs)")

    SOS_ID = w2i["<SOS>"]
    EOS_ID = w2i["<EOS>"]
    PAD_ID = w2i["<PAD>"]

    trg_ids = [SOS_ID]
    print(f"  {'step':>4}  {'pred_id':>7}  {'token':<22}  {'conf%':>6}  "
          f"top-{topk}: id:token:prob%")
    print("  " + "-" * 100)

    for step in range(max_len):
        t = torch.tensor([trg_ids[-1]], dtype=torch.long).to(device)
        with torch.no_grad():
            out, hidden = model.decoder(t, hidden, enc_out)

        probs   = F.softmax(out, dim=1)
        pred_id = probs.argmax(1).item()
        conf    = probs.max().item()

        topk_v, topk_i = probs[0].topk(topk)
        topk_str = "  ".join(
            f"{ti.item()}:{i2w.get(ti.item(),'?')!r}:{tv.item()*100:.1f}%"
            for ti, tv in zip(topk_i, topk_v)
        )

        tok  = i2w.get(pred_id, f"[UNK:{pred_id}]")
        flag = " <- EOS" if pred_id == EOS_ID else (
               " <- PAD??" if pred_id == PAD_ID else "")

        print(f"  {step:>4}  {pred_id:>7}  {tok:<22}  {conf*100:>6.1f}%  {topk_str}{flag}")
        trg_ids.append(pred_id)
        if pred_id == EOS_ID:
            break

    content = [i for i in trg_ids if i not in (SOS_ID, EOS_ID, PAD_ID)]
    decoded = " ".join(i2w.get(i, f"[UNK:{i}]") for i in content)

    print(f"\n  RAW TOKEN IDs   : {trg_ids}")
    print(f"  CONTENT IDs     : {content}")
    print(f"  DECODED         : {decoded!r}")
    return trg_ids


# ─────────────────────────────────────────────────────────────────────────────
# Stage-by-stage comparison with notebook numpy dumps
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_stages(
    raw_seq:   np.ndarray, nb_raw:   Optional[np.ndarray],
    resampled: np.ndarray, nb_res:   Optional[np.ndarray],
    vel_arr:   np.ndarray, nb_vel:   Optional[np.ndarray],
    norm_arr:  np.ndarray, nb_norm:  Optional[np.ndarray],
    final:     np.ndarray, nb_final: Optional[np.ndarray],
) -> None:
    hdr("NUMERICAL COMPARISON — app vs notebook stages")

    first_divergence = None

    stages = [
        ("Stage 1: raw landmarks",  raw_seq,   nb_raw),
        ("Stage 2: resampled",      resampled, nb_res),
        ("Stage 3: velocity",       vel_arr,   nb_vel),
        ("Stage 4: normalized",     norm_arr,  nb_norm),
        ("Stage 5: final (re-res)", final,     nb_final),
    ]

    for label, app_arr, nb_arr in stages:
        sub(label)
        if nb_arr is None:
            warn(f"No notebook data supplied for {label} — skipping comparison")
            continue

        stats(app_arr, f"  app_{label}", n=10)
        stats(nb_arr,  f"  nb_{label}",  n=10)
        close = compare(app_arr, nb_arr, "app", "notebook")

        if close:
            ok(f"{label}: MATCH")
        else:
            err(f"{label}: DIVERGES  <-- FIRST DIVERGENCE FOUND HERE")
            if first_divergence is None:
                first_divergence = label
            print("  All subsequent stages will also diverge.")
            print("  Fix the issue at this stage before comparing later ones.")
            break  # no point continuing

    print()
    if first_divergence:
        err(f"FIRST DIVERGENCE: {first_divergence}")
    else:
        ok("All supplied stages match numerically.")


# ─────────────────────────────────────────────────────────────────────────────
# Token-ID comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_token_ids(app_ids: List[int], nb_ids: List[int],
                      w2i: Dict, i2w: Dict) -> None:
    hdr("TOKEN ID COMPARISON")

    SOS = w2i["<SOS>"]
    EOS = w2i["<EOS>"]
    PAD = w2i["<PAD>"]

    print(f"  app IDs      : {app_ids}")
    print(f"  notebook IDs : {nb_ids}")
    print()

    max_len = max(len(app_ids), len(nb_ids))
    all_match = True

    print(f"  {'step':>4}  {'nb_id':>6}  {'app_id':>6}  {'match':>5}  "
          f"{'nb_tok':<20}  {'app_tok':<20}")
    print("  " + "-" * 80)

    for i in range(max_len):
        nb_id  = nb_ids[i]  if i < len(nb_ids)  else None
        app_id = app_ids[i] if i < len(app_ids) else None
        nb_tok  = i2w.get(nb_id,  "---") if nb_id  is not None else "---"
        app_tok = i2w.get(app_id, "---") if app_id is not None else "---"
        match = (nb_id == app_id)
        mark  = "OK" if match else "DIFF"
        if not match:
            all_match = False
        print(f"  {i:>4}  {str(nb_id):>6}  {str(app_id):>6}  {mark:>5}  "
              f"{nb_tok:<20}  {app_tok:<20}")

    print()
    if all_match:
        ok("TOKEN IDs MATCH — pipeline is numerically identical to notebook")
    else:
        n_diff = sum(1 for i in range(max_len)
                     if (nb_ids[i] if i < len(nb_ids) else None) !=
                        (app_ids[i] if i < len(app_ids) else None))
        err(f"TOKEN IDs differ at {n_diff} position(s)")
        print()
        print("  If you supplied numpy dumps and all stages matched but IDs differ,")
        print("  the issue is in the encoder/decoder weights or vocab.")
        print("  If stages diverged above, fix that stage first.")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook cell to print
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK_DUMP_CELL = '''
# ===================================================================
# PASTE THIS INTO YOUR NOTEBOOK — dumps numpy arrays at every stage
# Run on the SAME video file you pass to verify_parity.py
# ===================================================================

import numpy as np, torch, torch.nn.functional as F, cv2

VIDEO_PATH = "your_test_video.mp4"  # <-- change this

# --- Stage 0: raw frames
cap = cv2.VideoCapture(VIDEO_PATH)
frames_nb, raw_nb = [], []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames_nb.append(frame)
cap.release()

# --- Stage 1: landmarks
for frame in frames_nb:
    raw_nb.append(extract_keypoints(frame))   # notebook passes BGR as-is
raw_nb = np.array(raw_nb)
np.save("notebook_raw.npy", raw_nb)
print("Stage1 raw:", raw_nb.shape, "first20:", raw_nb.flatten()[:20])

# --- Stage 2: resample
res_nb = temporal_resample(raw_nb)
np.save("notebook_resampled.npy", res_nb)
print("Stage2 resampled:", res_nb.shape, "first20:", res_nb.flatten()[:20])

# --- Stage 3: velocity
vel_nb = compute_velocity(res_nb)
np.save("notebook_velocity.npy", vel_nb)
print("Stage3 velocity:", vel_nb.shape, "first20:", vel_nb.flatten()[:20])

# --- Stage 4: normalize
norm_nb = normalize(vel_nb)
np.save("notebook_normalized.npy", norm_nb)
print("Stage4 normalized:", norm_nb.shape, "first20:", norm_nb.flatten()[:20])

# --- Stage 5: demo re-resample
final_nb = temporal_resample(norm_nb, target_len=100)
np.save("notebook_final.npy", final_nb)
print("Stage5 final:", final_nb.shape, "first20:", final_nb.flatten()[:20])

# --- Stage 7: raw token IDs
src = torch.tensor(final_nb, dtype=torch.float32).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    enc_out, hidden = model.encoder(src)
ids = [word2idx["<SOS>"]]
for _ in range(30):
    t = torch.tensor([ids[-1]], dtype=torch.long).to(device)
    with torch.no_grad():
        out, hidden = model.decoder(t, hidden, enc_out)
    p = F.softmax(out, dim=1)
    pid = p.argmax(1).item()
    ids.append(pid)
    if pid == word2idx["<EOS>"]: break
print("NOTEBOOK RAW IDS:", ids)
# ===================================================================
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Numerical parity checker for BISINDO SLT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", nargs="?", default=None)
    p.add_argument("--nb-raw",        default=None, help="notebook_raw.npy")
    p.add_argument("--nb-resampled",  default=None, help="notebook_resampled.npy")
    p.add_argument("--nb-velocity",   default=None, help="notebook_velocity.npy")
    p.add_argument("--nb-normalized", default=None, help="notebook_normalized.npy")
    p.add_argument("--nb-final",      default=None, help="notebook_final.npy")
    p.add_argument("--notebook-ids",  default=None,
                   help='Comma-separated token IDs from notebook, e.g. "1,7,8,9,2"')
    p.add_argument("--topk",          type=int, default=5)
    p.add_argument("--dump-vocab",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("  BISINDO SLT — Numerical Parity Checker  (v3)")
    print("=" * 70)

    # Artifact check
    hdr("0  artifact presence")
    missing = False
    for label, path in [
        ("vocab.pkl",                 VOCAB_PATH),
        ("best_model.pth",            CHECKPOINT_PATH),
        ("hand_landmarker.task",      HAND_MODEL_PATH),
        ("pose_landmarker_lite.task", POSE_MODEL_PATH),
    ]:
        if os.path.isfile(path):
            ok(f"{label}  ({path})")
        else:
            err(f"{label} MISSING  ({path})")
            missing = True
    if missing:
        sys.exit(1)

    # Vocab
    hdr("Vocab")
    w2i, i2w = load_vocab(VOCAB_PATH)
    ok(f"Loaded vocab: {len(w2i)} tokens")
    ok(f"<PAD>={w2i['<PAD>']}  <SOS>={w2i['<SOS>']}  <EOS>={w2i['<EOS>']}")

    if args.dump_vocab:
        with open("vocab_dump.txt", "w") as f:
            f.write("idx\ttoken\n")
            for idx in sorted(i2w.keys()):
                f.write(f"{idx}\t{i2w[idx]}\n")
        ok("vocab_dump.txt written")

    # Model
    hdr("Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(CHECKPOINT_PATH, vocab_size=len(w2i), device=device)
    ok(f"Model loaded  params={sum(p.numel() for p in model.parameters()):,}  device={device}")

    if args.video is None:
        print("\n  No video supplied.")
        print("  Re-run: python verify_parity.py <video.mp4>\n")
        print("  Notebook cell to dump numpy arrays:")
        print(NOTEBOOK_DUMP_CELL)
        return

    if not os.path.isfile(args.video):
        err(f"Video not found: {args.video}")
        sys.exit(1)

    # ── Run app pipeline stage by stage ──────────────────────────────────────
    frames   = check_raw_frames(args.video)
    raw_seq  = check_landmarks(frames)
    resampled= check_resample(raw_seq)
    vel_arr  = check_velocity(resampled)
    norm_arr = check_normalize(vel_arr)
    final    = check_final_resample(norm_arr)
    enc_out, hidden = check_encoder(final, model, device)
    app_ids  = decode_with_ids(enc_out, hidden, model, w2i, i2w, device,
                                MAX_DECODE_LEN, args.topk)

    # ── Load notebook numpy dumps if provided ─────────────────────────────────
    def _load(path):
        if path and os.path.isfile(path):
            arr = np.load(path, allow_pickle=True)
            ok(f"Loaded {path}  shape={arr.shape}")
            return arr
        if path:
            warn(f"File not found: {path}")
        return None

    nb_raw   = _load(args.nb_raw)
    nb_res   = _load(args.nb_resampled)
    nb_vel   = _load(args.nb_velocity)
    nb_norm  = _load(args.nb_normalized)
    nb_final = _load(args.nb_final)

    if any(x is not None for x in [nb_raw, nb_res, nb_vel, nb_norm, nb_final]):
        compare_all_stages(
            raw_seq, nb_raw, resampled, nb_res,
            vel_arr, nb_vel, norm_arr,  nb_norm,
            final,   nb_final,
        )
    else:
        hdr("NOTEBOOK COMPARISON")
        warn("No notebook numpy dumps supplied.")
        print("  Paste the notebook cell below, run it, then re-run with --nb-* flags.")
        print(NOTEBOOK_DUMP_CELL)
        print("  Example:")
        print(f"    python verify_parity.py {args.video} \\")
        print( "        --nb-raw notebook_raw.npy \\")
        print( "        --nb-resampled notebook_resampled.npy \\")
        print( "        --nb-velocity notebook_velocity.npy \\")
        print( "        --nb-normalized notebook_normalized.npy \\")
        print( "        --nb-final notebook_final.npy")

    # ── Token-ID comparison ───────────────────────────────────────────────────
    if args.notebook_ids:
        nb_ids = [int(x.strip()) for x in args.notebook_ids.split(",") if x.strip()]
        compare_token_ids(app_ids, nb_ids, w2i, i2w)

    print()


if __name__ == "__main__":
    main()

"""
preprocessing.py
----------------
EXACT copy of every preprocessing/feature-engineering function from the
notebook (SLTpipeline__1_.ipynb).

ROOT-CAUSE FIX (BGR vs RGB):
  The notebook's extract_keypoints receives frames directly from cap.read().
  OpenCV cap.read() returns BGR frames.
  The notebook passes those BGR frames to mp.Image(format=SRGB, data=frame)
  WITHOUT any colour conversion.  MediaPipe then interprets the bytes as
  RGB, but since the bytes are actually BGR, the channels are swapped.

  The previous refactor added:
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  before calling extract_keypoints.  This sent TRUE RGB to MediaPipe,
  which is a completely different pixel input for a neural-network-based
  landmark detector → different (x,y,z) values → downstream numerical
  divergence all the way through velocity, normalization, and the encoder.

  FIX: process_video must pass the raw BGR frame (no conversion) to
  extract_keypoints, exactly as the notebook does.  The "wrong" colour
  order is intentional — it matches training data.

MANDATORY: nothing else here may be changed.
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------------
# MediaPipe detector singletons (lazy-initialised so import is cheap)
# ---------------------------------------------------------------------------

_hand_detector = None
_pose_detector = None


def _get_detectors(hand_model_path: str, pose_model_path: str):
    """Return (hand_detector, pose_detector), initialising once."""
    global _hand_detector, _pose_detector
    if _hand_detector is None:
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        _hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
        )
    if _pose_detector is None:
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        _pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_options_pose)
        )
    return _hand_detector, _pose_detector


# ---------------------------------------------------------------------------
# Landmark extraction — IDENTICAL to notebook cell "extract_keypoints"
#
# IMPORTANT: the notebook passes raw BGR frames from cap.read() here.
# We do the same.  Do NOT add cv2.cvtColor before calling this function.
# ---------------------------------------------------------------------------

def extract_keypoints(frame: np.ndarray,
                      hand_model_path: str,
                      pose_model_path: str) -> np.ndarray:
    """Extract pose + hand keypoints from a single frame.

    `frame` must be the raw uint8 array returned by cap.read() — i.e. BGR.
    It is passed to mp.Image with format=SRGB exactly as the notebook does,
    which means MediaPipe sees the B and R channels swapped relative to real
    colour order.  This matches how training data was extracted, so the same
    convention must be used at inference time.

    Returns a 1-D float64 array of length 33*3 + 21*3*2 = 225.
    """
    hand_detector, pose_detector = _get_detectors(hand_model_path, pose_model_path)

    # Pass frame as-is (BGR from OpenCV) with SRGB format label — identical
    # to the notebook.  Do NOT convert to RGB first.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_result = pose_detector.detect(image)
    hand_result = hand_detector.detect(image)

    pose_kp = np.zeros(33 * 3)
    hand_kp = np.zeros(21 * 3 * 2)

    if pose_result.pose_landmarks:
        pts = [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]
        pose_kp = np.array(pts).flatten()

    if hand_result.hand_landmarks:
        pts = [[lm.x, lm.y, lm.z]
               for hand in hand_result.hand_landmarks
               for lm in hand]
        hand_kp[:len(pts) * 3] = np.array(pts).flatten()

    return np.concatenate([pose_kp, hand_kp])


# ---------------------------------------------------------------------------
# Feature engineering — IDENTICAL to notebook "Feature Engineering" cell
# ---------------------------------------------------------------------------

def temporal_resample(seq: np.ndarray, target_len: int = 100) -> np.ndarray:
    """Linearly resample a (T, F) sequence to exactly target_len frames."""
    idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
    return seq[idx]


def compute_velocity(seq: np.ndarray) -> np.ndarray:
    """Append per-frame velocity (diff) to the feature vector."""
    vel = np.diff(seq, axis=0)
    vel = np.vstack([np.zeros(seq.shape[1]), vel])
    return np.concatenate([seq, vel], axis=1)


def normalize(seq: np.ndarray) -> np.ndarray:
    """Z-score normalise along the time axis."""
    mean = seq.mean(axis=0)
    std = seq.std(axis=0) + 1e-6
    return (seq - mean) / std


# ---------------------------------------------------------------------------
# Full video → feature tensor — IDENTICAL to notebook "process_video"
#
# KEY: frame is passed to extract_keypoints WITHOUT any colour conversion.
# This matches the notebook exactly.
# ---------------------------------------------------------------------------

def process_video(path: str,
                  hand_model_path: str,
                  pose_model_path: str) -> np.ndarray:
    """Read a video file and return the normalised feature array (100, 450).

    Pipeline (ORDER MUST NOT CHANGE):
        read frames (BGR) → extract_keypoints (pass BGR as-is, no conversion)
        → temporal_resample → compute_velocity → normalize
    """
    cap = cv2.VideoCapture(path)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Pass the raw BGR frame directly — no cvtColor.
        # The notebook does the same: sequence.append(extract_keypoints(frame))
        sequence.append(extract_keypoints(frame, hand_model_path, pose_model_path))

    cap.release()

    seq = np.array(sequence)
    seq = temporal_resample(seq)    # (100, 225)
    seq = compute_velocity(seq)     # (100, 450)
    seq = normalize(seq)            # (100, 450) float64
    return seq

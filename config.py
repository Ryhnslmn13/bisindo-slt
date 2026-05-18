"""
config.py
---------
Central place for all path/hyperparameter constants.
Edit this file to point to your actual model artifacts.
"""

import os

# ---------------------------------------------------------------------------
# Paths — override with environment variables if needed
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MediaPipe model files (downloaded by the notebook setup cells)
HAND_MODEL_PATH = os.environ.get(
    "HAND_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "hand_landmarker.task"),
)

POSE_MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task"),
)

# Trained checkpoint
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(BASE_DIR, "models", "best_model.pth"),
)

# Vocabulary pickle saved by the notebook
VOCAB_PATH = os.environ.get(
    "VOCAB_PATH",
    os.path.join(BASE_DIR, "models", "vocab.pkl"),
)

# ---------------------------------------------------------------------------
# Model hyperparameters — must match training
# ---------------------------------------------------------------------------

INPUT_DIM = 450    # 225 raw landmarks + 225 velocity
HIDDEN_DIM = 256
MAX_DECODE_LEN = 30

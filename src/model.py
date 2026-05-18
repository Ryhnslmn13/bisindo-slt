"""
model.py
--------
EXACT copy of every model class from the notebook:
    Encoder  →  BahdanauAttention  →  Decoder  →  Seq2Seq

These must not be changed; any architectural drift breaks checkpoint loading.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """CNN-GRU encoder.  Input: (B, T, input_dim)."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, out_channels=256, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.gru = nn.GRU(256, hidden, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.relu(self.conv1d(x))   # (B, 256, T)
        x = x.permute(0, 2, 1)          # (B, T, 256)
        outputs, hidden = self.gru(x)   # outputs: (B, T, H), hidden: (1, B, H)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W1 = nn.Linear(hidden, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.V = nn.Linear(hidden, 1)

    def forward(self, hidden, enc_out):
        hidden = hidden.permute(1, 0, 2)          # (B, 1, H)
        score = self.V(torch.tanh(self.W1(enc_out) + self.W2(hidden)))  # (B, T, 1)
        weights = torch.softmax(score, dim=1)     # (B, T, 1)
        context = (weights * enc_out).sum(dim=1)  # (B, H)
        return context


class Decoder(nn.Module):
    """GRU decoder with Bahdanau attention."""

    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.attn = BahdanauAttention(hidden)

    def forward(self, token, hidden, enc_out):
        emb = self.embedding(token).unsqueeze(1)           # (B, 1, H)
        context = self.attn(hidden, enc_out).unsqueeze(1)  # (B, 1, H)
        x = torch.cat([emb, context], dim=2)               # (B, 1, 2H)
        out, hidden = self.gru(x, hidden)                  # out: (B, 1, H)
        pred = self.fc(out.squeeze(1))                     # (B, vocab_size)
        return pred, hidden


class Seq2Seq(nn.Module):
    """Full sequence-to-sequence model (teacher-forcing during training)."""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        enc_out, hidden = self.encoder(src)
        input_token = trg[:, 0]
        outputs = []

        for t in range(1, trg.shape[1]):
            pred, hidden = self.decoder(input_token, hidden, enc_out)
            outputs.append(pred)
            input_token = trg[:, t]

        return torch.stack(outputs).permute(1, 0, 2)  # (B, T-1, vocab_size)


# ---------------------------------------------------------------------------
# Constants that match training
# ---------------------------------------------------------------------------

INPUT_DIM = 450   # 225 raw + 225 velocity
HIDDEN_DIM = 256

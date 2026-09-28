"""
Test-Triage Agent: attention-pooled BiLSTM over clock-aligned trace sequences.

Differences from the Phase 1 model, all of which were defects:
  * Phase 1 fed 2155-wide EVENT-COUNT vectors into nn.Embedding as token IDs.
    This consumes real ordered `OpName|status` tokens from src/data.
  * Phase 1 truncated to the first 100 of 2155 columns, discarding 95% of the
    signal. Sequences here are variable length and packed.
  * Phase 1 read `hidden[-1]` of a unidirectional LSTM over PADDED input, so
    the final state was dominated by padding. This packs the batch so the LSTM
    never sees a pad step, and pools with masked attention instead.
  * Phase 1 had a single sigmoid over an ill-defined binary label. This is a
    5-way head over classes with real operational meaning.

Attention weights are returned so a triage decision can point at the spans that
drove it - required by the Postmortem agent downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TriageLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, n_task_feats=9,
                 n_host_roles=4, n_agents=6, n_depths=16,
                 emb_dim=96, side_dim=8, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.host = nn.Embedding(n_host_roles, side_dim)
        self.agent = nn.Embedding(n_agents, side_dim)
        self.depth = nn.Embedding(n_depths, side_dim)
        self.dur = nn.Linear(1, side_dim)

        in_dim = emb_dim + 4 * side_dim
        self.in_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        seq_dim = hidden * 2

        # additive attention pooling, masked to real timesteps
        self.att = nn.Sequential(
            nn.Linear(seq_dim, 64), nn.Tanh(), nn.Linear(64, 1))

        self.feat = nn.Sequential(
            nn.Linear(n_task_feats, 32), nn.ReLU(), nn.Dropout(dropout))

        self.head = nn.Sequential(
            nn.Linear(seq_dim * 2 + 32, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))

    def forward(self, batch, return_attention=False):
        ids, mask, lens = batch["ids"], batch["mask"], batch["lens"]
        x = torch.cat([
            self.tok(ids),
            self.host(batch["hr"]),
            self.agent(batch["ag"]),
            self.depth(batch["dep"]),
            self.dur(batch["dur"].unsqueeze(-1)),
        ], dim=-1)
        x = self.in_drop(x)

        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=ids.size(1))

        scores = self.att(out).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        w = torch.softmax(scores, dim=1)
        attended = torch.bmm(w.unsqueeze(1), out).squeeze(1)

        masked = out.masked_fill(~mask.unsqueeze(-1),
                                 torch.finfo(out.dtype).min)
        pooled_max = masked.max(dim=1).values

        z = torch.cat([attended, pooled_max, self.feat(batch["feat"])], dim=-1)
        logits = self.head(z)
        return (logits, w) if return_attention else logits

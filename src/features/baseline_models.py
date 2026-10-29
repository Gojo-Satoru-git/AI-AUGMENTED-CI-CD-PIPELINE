"""
Comparison models: re-implementations, on the SAME corpus, split and training recipe, of the
architectures used by the published trace/log failure-analysis work that this project is compared
with. They are NOT the original authors' code or results: each is the published architecture
adapted to the same supervised 5-class task, and each is trained here with the same optimiser,
class weights, seed and early stopping as the proposed model.

  deeplog     DeepLog (Du et al., CCS 2017): 2-layer unidirectional LSTM over the event-key sequence.
              Original is next-key prediction (unsupervised); here the last hidden state is classified.
  logrobust   LogRobust (Zhang et al., FSE 2019): attention-based BiLSTM over the event sequence.
              Original uses semantic vectors of log text; operation names here are learned embeddings.
  textcnn     TextCNN (Kim, EMNLP 2014) over the event sequence: kernels 2/3/4, max-over-time pooling.
  multimodal  Span + timing LSTM (after Nedelkoski et al., CLOUD 2019): token embedding concatenated
              with the span duration, unidirectional LSTM.

All four read the operation tokens (and, for multimodal, durations). None sees the host / agent /
depth side features or the trace-level scalars that the proposed model also uses.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ARCHS = ("triage", "deeplog", "logrobust", "textcnn", "multimodal", "deeplog_full")


def _last_hidden(lstm, x, lens):
    packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
    _, (h, _) = lstm(packed)
    return h[-1]


class DeepLogStyle(nn.Module):
    def __init__(self, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, b):
        h = _last_hidden(self.lstm, self.drop(self.tok(b["ids"])), b["lens"])
        return self.head(self.drop(h))


class LogRobustStyle(nn.Module):
    def __init__(self, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, b):
        ids, mask, lens = b["ids"], b["mask"], b["lens"]
        x = self.drop(self.tok(ids))
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        s = self.att(out).squeeze(-1).masked_fill(~mask, torch.finfo(out.dtype).min)
        w = torch.softmax(s, dim=1)
        return self.head(self.drop(torch.bmm(w.unsqueeze(1), out).squeeze(1)))


class TextCNNStyle(nn.Module):
    def __init__(self, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, hidden, k, padding=k // 2) for k in (2, 3, 4)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(3 * hidden, n_classes)

    def forward(self, b):
        x = self.drop(self.tok(b["ids"])).transpose(1, 2)           # B, E, L
        mask = b["mask"].unsqueeze(1)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[..., : x.size(2)]
            h = h.masked_fill(~mask, torch.finfo(h.dtype).min)
            pooled.append(h.max(dim=2).values)
        return self.head(self.drop(torch.cat(pooled, dim=1)))


class MultimodalStyle(nn.Module):
    def __init__(self, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.dur = nn.Linear(1, 8)
        self.lstm = nn.LSTM(emb_dim + 8, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, b):
        x = torch.cat([self.tok(b["ids"]), self.dur(b["dur"].unsqueeze(-1))], dim=-1)
        return self.head(self.drop(_last_hidden(self.lstm, self.drop(x), b["lens"])))


class DeepLogFull(nn.Module):
    """DeepLog-style 2-layer unidirectional LSTM + last hidden state, given EXACTLY the inputs the proposed
    model gets (token, host role, agent type, depth, duration, nine trace-level features). Isolates the
    contribution of the architecture (attention + BiLSTM + max-pool) from the contribution of the inputs."""

    def __init__(self, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3, n_feats=9, side=8):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.host, self.agent, self.depth = nn.Embedding(4, side), nn.Embedding(6, side), nn.Embedding(16, side)
        self.dur = nn.Linear(1, side)
        self.lstm = nn.LSTM(emb_dim + 4 * side, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.feat = nn.Sequential(nn.Linear(n_feats, 32), nn.ReLU(), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + 32, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, b):
        x = torch.cat([self.tok(b["ids"]), self.host(b["hr"]), self.agent(b["ag"]), self.depth(b["dep"]),
                       self.dur(b["dur"].unsqueeze(-1))], dim=-1)
        h = _last_hidden(self.lstm, self.drop(x), b["lens"])
        return self.head(torch.cat([h, self.feat(b["feat"])], dim=-1))


def build(arch, vocab, n_classes, emb_dim=48, hidden=64, dropout=0.3):
    cls = {"deeplog": DeepLogStyle, "logrobust": LogRobustStyle,
           "textcnn": TextCNNStyle, "multimodal": MultimodalStyle,
           "deeplog_full": DeepLogFull}[arch]
    return cls(vocab, n_classes, emb_dim, hidden, dropout)

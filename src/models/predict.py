"""
Serving wrapper for the Test-Triage Agent.

Loads the checkpoint once and turns a trace into a triage decision with the
evidence behind it: the predicted class, calibrated-ish probabilities, and the
spans the attention head weighted most. The Postmortem agent consumes those
spans, so they are part of the contract, not a debugging aid.

Accepts either
  * a pre-tokenised sequence ("getFileInfo|OK RPC:addBlock|IOEXC ..."), or
  * raw span dicts (as the Jaeger adapter will emit), which are tokenised here
    with the same rules as src/data/build_sequences.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import N_TASK_FEATS, N_DEPTHS, UNK   # noqa: E402
from src.features.triage_lstm import TriageLSTM            # noqa: E402

DEFAULT_CKPT = ROOT / "src" / "models" / "saved" / "triage_lstm.pth"
DEFAULT_VOCAB = ROOT / "src" / "models" / "saved" / "vocab.json"


def host_role(hostname: str) -> int:
    h = (hostname or "").strip().lower()
    if h.startswith("namenode"):
        return 0
    if h.startswith("datanode"):
        return 1
    if h.startswith("client"):
        return 2
    return 3


AGENTS = ["Namenode", "Datanode", "RPC Client", "DFSClient", "User", "OTH"]
_AGENT_IX = {a.lower(): i for i, a in enumerate(AGENTS)}   # case-insensitive


def normalise_op(op: str) -> str:
    return op.strip().replace(" ", "_").replace("|", "/")


def spans_to_tokens(spans):
    """spans: [{op, status, duration, host, agent, depth}, ...]"""
    toks, dur, hr, ag, dep = [], [], [], [], []
    for s in spans:
        toks.append(f"{normalise_op(s['op'])}|{s.get('status', 'OK')}")
        dur.append(int(s.get("duration", 0)))
        hr.append(host_role(s.get("host", "")))
        ag.append(_AGENT_IX.get(str(s.get("agent", "")).strip().lower(), 5))
        dep.append(min(int(s.get("depth", 0)), N_DEPTHS - 1))
    return toks, dur, hr, ag, dep


class TriagePredictor:
    def __init__(self, ckpt=DEFAULT_CKPT, vocab_path=DEFAULT_VOCAB,
                 device=None):
        cfg = json.loads(Path(vocab_path).read_text())
        self.vocab = cfg["vocab"]
        self.labels = cfg["labels"]
        self.max_len = cfg["max_len"]
        self.feature_schema = cfg.get("feature_schema", "tracebench_v1")
        self.feat_mean = torch.tensor(cfg["feat_mean"])
        self.feat_std = torch.tensor(cfg["feat_std"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = cfg["model"]
        self.model = TriageLSTM(
            len(self.vocab), len(self.labels), n_task_feats=N_TASK_FEATS,
            emb_dim=m["emb_dim"], hidden=m["hidden"], layers=m["layers"],
            dropout=m["dropout"]).to(self.device)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------
    def _task_feats(self, toks, dur, hr, dep, extra):
        n_ev = max(len(toks), 1)
        n_err = sum(1 for t in toks
                    if t.rsplit("|", 1)[-1] not in ("OK", "USER"))
        return [
            math.log1p(n_ev),
            math.log1p(extra.get("n_tid", len(set(dep)) or 1)),
            math.log1p(extra.get("n_host", len(set(hr)) or 1)),
            math.log1p(extra.get("n_edge", n_ev)),
            math.log1p(extra.get("n_rep", n_ev)),
            math.log1p(extra.get("lat", sum(dur))),
            math.log1p(n_err),
            n_err / n_ev,
            math.log1p(len({t.split("|")[0] for t in toks})),
        ]

    @torch.no_grad()
    def predict(self, tokens=None, spans=None, extra=None, top_spans=5):
        if spans is not None:
            toks, dur, hr, ag, dep = spans_to_tokens(spans)
        else:
            toks = (tokens.split(" ") if isinstance(tokens, str)
                    else list(tokens))
            dur = [0] * len(toks)
            hr = [3] * len(toks)
            ag = [5] * len(toks)
            dep = [0] * len(toks)
        if not toks:
            raise ValueError("empty trace: nothing to triage")

        toks, dur, hr, ag, dep = (x[: self.max_len]
                                  for x in (toks, dur, hr, ag, dep))
        feats = self._task_feats(toks, dur, hr, dep, extra or {})
        return self._infer(toks, dur, hr, ag, dep, feats, top_spans)

    @torch.no_grad()
    def predict_record(self, r, top_spans=5):
        """Predict from a corpus record exactly as evaluate.py does: the real
        precomputed host/agent/depth codes and the real task-level feature
        vector r["f"]. Use this (not predict(spans=...)) whenever a corpus
        record is available - reconstructing spans loses host role, agent and
        the task scalars, which silently degrades every prediction."""
        from src.data.dataset import task_features
        toks = r["toks"].split(" ")[: self.max_len]
        n = len(toks)
        dur = [int(x) for x in r["dur"].split(",")][:n]
        dep = [min(int(x), N_DEPTHS - 1) for x in r["dep"].split(",")][:n]
        hr = [int(c) for c in r["hr"]][:n]
        ag = [int(c) for c in r["ag"]][:n]
        return self._infer(toks, dur, hr, ag, dep, task_features(r), top_spans)

    def _infer(self, toks, dur, hr, ag, dep, feats, top_spans):
        n = len(toks)
        feats = torch.tensor([feats], dtype=torch.float32)
        feats = (feats - self.feat_mean) / self.feat_std

        batch = {
            "ids": torch.tensor([[self.vocab.get(t, UNK) for t in toks]],
                                dtype=torch.long),
            "hr": torch.tensor([hr], dtype=torch.long),
            "ag": torch.tensor([ag], dtype=torch.long),
            "dep": torch.tensor([dep], dtype=torch.long),
            "dur": torch.tensor([[math.log1p(max(d, 0)) for d in dur]],
                                dtype=torch.float32),
            "mask": torch.ones(1, n, dtype=torch.bool),
            "lens": torch.tensor([n], dtype=torch.long),
            "feat": feats,
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits, att = self.model(batch, return_attention=True)
        probs = torch.softmax(logits, -1)[0]
        k = int(probs.argmax())

        w = att[0][:n]
        order = torch.argsort(w, descending=True)[:top_spans]
        evidence = [{"index": int(i), "token": toks[int(i)],
                     "attention": round(float(w[int(i)]), 4),
                     "duration_us": int(dur[int(i)])}
                    for i in order]
        # OOV rate is a live drift signal once traces come from a different
        # system (TrainTicket) than the one the model was trained on.
        oov = sum(1 for t in toks if t not in self.vocab) / n
        return {
            "classification": self.labels[k],
            "confidence": round(float(probs[k]), 4),
            "probabilities": {l: round(float(p), 4)
                              for l, p in zip(self.labels, probs)},
            "n_spans": n,
            "oov_rate": round(oov, 4),
            "evidence_spans": evidence,
        }

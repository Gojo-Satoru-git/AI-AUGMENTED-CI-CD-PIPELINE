"""Reads the measured results of the comparison models (reports/cmp, reports/cmp_*.log)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["PASS", "INFRA_FLAKE", "DATA_FAULT", "ENV_FAULT", "REAL_BUG"]
NAMES = {
    "triage": ("Proposed: attention BiLSTM + side features", "this work"),
    "logrobust": ("LogRobust-style: attention BiLSTM, tokens only", "Zhang et al., FSE 2019"),
    "multimodal": ("Span + timing LSTM", "after Nedelkoski et al., CLOUD 2019"),
    "deeplog": ("DeepLog-style: 2-layer LSTM, tokens only", "Du et al., CCS 2017"),
    "deeplog_full": ("DeepLog-style LSTM given the same inputs as ours", "Du et al. backbone, our inputs"),
    "textcnn": ("TextCNN, tokens only", "Kim, EMNLP 2014"),
}


def load():
    """-> list of dict(name, source, params, macro_f1, excl_real_bug, acc, per_class, seconds, epochs)"""
    rows = []
    ev = json.loads((ROOT / "reports/evaluation.json").read_text(encoding="utf-8"))["model"]
    tr = json.loads((ROOT / "reports/triage_train.json").read_text(encoding="utf-8"))
    pc = {k: v["f1"] for k, v in ev["per_class"].items()}
    rows.append(dict(key="triage", name=NAMES["triage"][0], source=NAMES["triage"][1], params=tr["params"],
                     macro_f1=ev["macro_f1"], acc=ev["accuracy"], per_class=pc,
                     excl=sum(v for k, v in pc.items() if k != "REAL_BUG") / 4,
                     seconds=tr["train_seconds"], epochs=len(tr["history"])))
    for key in ("logrobust", "multimodal", "deeplog", "textcnn", "deeplog_full"):
        f = ROOT / "reports/cmp" / f"{key}_train.json"
        log = ROOT / "reports" / f"cmp_{key}.log"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        per = {}
        for m in re.finditer(r"^\s+(PASS|INFRA_FLAKE|DATA_FAULT|ENV_FAULT|REAL_BUG)\s+F1 ([0-9.]+)", log.read_text(encoding="utf-8", errors="replace"), re.M):
            per[m.group(1)] = float(m.group(2))
        rows.append(dict(key=key, name=NAMES[key][0], source=NAMES[key][1], params=j["params"],
                         macro_f1=j["test_macro_f1"], acc=j["test_acc"], per_class=per,
                         excl=sum(v for k, v in per.items() if k != "REAL_BUG") / 4 if len(per) == 5 else None,
                         seconds=j["train_seconds"], epochs=len(j["history"])))
    return rows


if __name__ == "__main__":
    for r in load():
        print(r["key"], r["params"], round(r["macro_f1"], 4), round(r["acc"], 4), r["excl"] and round(r["excl"], 4), r["epochs"], r["seconds"])

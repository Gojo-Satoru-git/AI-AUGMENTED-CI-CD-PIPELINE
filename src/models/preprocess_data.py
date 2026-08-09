import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from src.features.trace_anomaly_lstm import TraceAnomalyLSTM

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_DIR = ROOT_DIR / "src" / "models" / "saved"

FAIL_CSV = DATA_DIR / "failure_trace.csv"
NORM_CSV = DATA_DIR / "normal_trace.csv"
FAIL_JSON = DATA_DIR / "failure_taskId.json"
NORM_JSON = DATA_DIR / "normal_taskId.json"
MODEL_PATH = MODEL_DIR / "trace_anomaly_lstm.pth"

MAX_SEQ_LEN = 100
BATCH_SIZE = 64


def clean_and_pad_pretokenized(seq_string: str) -> List[int]:
    tokens: List[int] = []
    for x in str(seq_string).split(","):
        x = x.strip()
        if x:
            try:
                tokens.append(int(float(x)))
            except ValueError:
                pass

    if len(tokens) > MAX_SEQ_LEN:
        return tokens[:MAX_SEQ_LEN]
    else:
        return tokens + [0] * (MAX_SEQ_LEN - len(tokens))


def build_balanced_dataset() -> pd.DataFrame:
    with open(FAIL_CSV, "r", encoding="utf-8") as f:
        fail_lines = f.read().splitlines()
    df_fail = pd.DataFrame(fail_lines, columns=["raw_trace"])

    with open(NORM_CSV, "r", encoding="utf-8") as f:
        norm_lines = f.read().splitlines()
    df_norm = pd.DataFrame(norm_lines, columns=["raw_trace"])

    df_fail[["TaskID", "Trace_Sequence"]] = df_fail["raw_trace"].str.split(",", n=1, expand=True)
    df_norm[["TaskID", "Trace_Sequence"]] = df_norm["raw_trace"].str.split(",", n=1, expand=True)

    with open(FAIL_JSON, "r", encoding="utf-8") as f:
        fail_ids = [item[0] for item in json.load(f)]

    with open(NORM_JSON, "r", encoding="utf-8") as f:
        norm_ids = [item[0] for item in json.load(f)]

    df_fail_filtered = df_fail[df_fail["TaskID"].isin(fail_ids)].copy()
    df_norm_filtered = df_norm[df_norm["TaskID"].isin(norm_ids)].copy()

    df_fail_filtered["Label"] = 1
    df_norm_filtered["Label"] = 0

    df_master = pd.concat([df_fail_filtered, df_norm_filtered], ignore_index=True)
    df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)

    df_majority = df_master[df_master["Label"] == 0]
    df_minority = df_master[df_master["Label"] == 1]
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42,
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    df_balanced["Encoded_Sequence"] = df_balanced["Trace_Sequence"].apply(clean_and_pad_pretokenized)
    return df_balanced


def prepare_tensors(df_balanced: pd.DataFrame):
    X = torch.tensor(df_balanced["Encoded_Sequence"].tolist(), dtype=torch.long)
    y = torch.tensor(df_balanced["Label"].tolist(), dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return X, y, dataloader


def save_preprocessed_dataset(output_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    df_balanced = build_balanced_dataset()
    X, y, dataloader = prepare_tensors(df_balanced)

    if output_path is None:
        output_path = MODEL_DIR / "preprocessed_dataset.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df_balanced.head(100).iterrows():
        records.append(
            {
                "task_id": row.get("TaskID"),
                "trace_sequence": row.get("Trace_Sequence"),
                "label": int(row.get("Label", 0)),
                "encoded_sequence": row.get("Encoded_Sequence"),
            }
        )

    artifact = {
        "metadata": {
            "rows": int(len(df_balanced)),
            "vocab_size": int(X.max()) + 1,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
        },
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return artifact


def load_preprocessed_dataset(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    if path is None:
        path = MODEL_DIR / "preprocessed_dataset.json"
    path = Path(path)
    if not path.exists():
        return save_preprocessed_dataset(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model() -> TraceAnomalyLSTM:
    metadata = load_preprocessed_dataset()
    vocab_size = metadata["metadata"]["vocab_size"]
    model = TraceAnomalyLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, output_dim=1)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

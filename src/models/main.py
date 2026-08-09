import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from src.features.trace_anomaly_lstm import TraceAnomalyLSTM

# ==========================================
# 1. LOAD THE FROZEN BRAIN
# ==========================================
VOCAB_SIZE = 49  # Matches your notebook's VOCAB_SIZE
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
MAX_SEQ_LEN = 100

model = TraceAnomalyLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, 1)
model.load_state_dict(torch.load("src/models/saved/trace_anomaly_lstm.pth"))
model.eval()  # Set to evaluation mode

# ==========================================
# 2. SETUP FASTAPI SERVER & SCHEMAS
# ==========================================
app = FastAPI(
    title="AI Test-Triage Agent API",
    description="Inference API for trace anomaly detection using PyTorch LSTM.",
)

# Request schema for local file path input
class FilePathRequest(BaseModel):
    file_path: str  # e.g., "data/raw/test_traces.csv" or "\x\y"

# Request schema for single trace item
class WebhookPayload(BaseModel):
    test_id: str
    trace_sequence: str  # e.g., "4, 4, 2, 20, 17"


# ==========================================
# 3. PREPROCESSING & INFERENCE LOGIC
# ==========================================
def preprocess_trace(seq_string: str):
    """The exact same cleaning function from your notebook"""
    tokens = []
    for x in str(seq_string).split(","):
        x = x.strip()
        if x:
            try:
                tokens.append(int(float(x)))
            except ValueError:
                pass  # Skip invalid values/empty strings

    if len(tokens) > MAX_SEQ_LEN:
        return tokens[:MAX_SEQ_LEN]
    else:
        return tokens + [0] * (MAX_SEQ_LEN - len(tokens))


def predict_single(test_id: str, trace_sequence: str) -> dict:
    """Helper function to run inference on a single task ID and trace sequence."""
    clean_seq = preprocess_trace(trace_sequence)

    # Convert to Tensor with Batch Dimension -> Shape: (1, 100)
    input_tensor = torch.tensor([clean_seq], dtype=torch.long)

    with torch.no_grad():
        probability = model(input_tensor).item()

    classification = "Flaky/Infrastructure" if probability > 0.80 else "Code Bug"

    return {
        "test_id": test_id,
        "anomaly_probability": round(probability, 4),
        "classification": classification,
        "action_recommended": (
            "RETRY_PIPELINE"
            if classification == "Flaky/Infrastructure"
            else "ALERT_DEVELOPER"
        ),
    }


def parse_and_process_file(file_path: str) -> list[dict]:
    """Reads lines formatted like your raw CSV ('TaskID,trace_sequence...'),

    preprocesses them, and runs model predictions.
    """
    # Normalize cross-platform file paths (handles '\' and '/' correctly)
    normalized_path = os.path.normpath(file_path)

    if not os.path.exists(normalized_path):
        raise HTTPException(
            status_code=404, detail=f"File not found at path: '{normalized_path}'"
        )

    results = []

    try:
        with open(normalized_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Matches the notebook logic: split TaskID from sequence on first comma
            parts = line.split(",", 1)
            if len(parts) == 2:
                task_id, trace_seq = parts[0].strip(), parts[1].strip()
            else:
                # Fallback if no task ID is present in the line
                task_id, trace_seq = f"line_{line_num}", parts[0].strip()

            prediction = predict_single(test_id=task_id, trace_sequence=trace_seq)
            results.append(prediction)

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading file '{normalized_path}': {str(e)}"
        )


# ==========================================
# 4. API ENDPOINTS
# ==========================================


# Option 1: Single Payload JSON Endpoint
@app.post("/api/v1/triage")
def triage_test_failure(payload: WebhookPayload):
    return predict_single(
        test_id=payload.test_id, trace_sequence=payload.trace_sequence
    )


# Option 2: POST File Path Endpoint (Body: {"file_path": "path/to/file"})
@app.post("/api/v1/triage/file-path")
def triage_from_file_path_post(payload: FilePathRequest):
    results = parse_and_process_file(payload.file_path)
    return {"total_records": len(results), "predictions": results}


# Option 3: GET File Path Endpoint (Query parameter: ?file_path=path/to/file)
@app.get("/api/v1/triage/file-path")
def triage_from_file_path_get(
    file_path: str = Query(
        ..., description="Relative or absolute server file path (e.g. data/raw/failure_trace.csv)"
    )
):
    results = parse_and_process_file(file_path)
    return {"total_records": len(results), "predictions": results}


# ==========================================
# 5. SERVER RUNNER
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from typing import Optional

from src.models.preprocess_data import (
    load_model,
    load_preprocessed_dataset,
    save_preprocessed_dataset,
)

from src.policy_engine.database import get_developer_tier
from src.policy_engine.graph_generator import generate_root_cause_graph

# ==========================================
# 1. LOAD THE FROZEN BRAIN
# ==========================================
MAX_SEQ_LEN = 100
model = load_model()

# ==========================================
# 2. SETUP FASTAPI SERVER & SCHEMAS
# ==========================================
app = FastAPI(
    title="AI Test-Triage Policy Engine API",
    description="Inference API and Policy Engine for trace anomaly detection using PyTorch LSTM.",
)

class FilePathRequest(BaseModel):
    file_path: str

class WebhookPayload(BaseModel):
    test_id: str
    trace_sequence: str
    developer_id: Optional[str] = "dev_junior_1" # Default to tier 1 for safety if missing

# ==========================================
# 3. PREPROCESSING & INFERENCE LOGIC
# ==========================================
def preprocess_trace(seq_string: str):
    tokens = []
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

def evaluate_policy(classification: str, developer_tier: int) -> str:
    """Evaluates the Policy Matrix based on AI classification and dev tier."""
    if classification == "Code Bug":
        return "BLOCK_MERGE_ALERT_DEV"
    
    # Flaky/Infrastructure cases
    if developer_tier >= 3:
        return "AUTO_RETRY_PIPELINE"
    else:
        return "PAUSE_FOR_MANUAL_REVIEW"

def predict_single(test_id: str, trace_sequence: str, developer_id: str = None) -> dict:
    clean_seq = preprocess_trace(trace_sequence)
    input_tensor = torch.tensor([clean_seq], dtype=torch.long)

    with torch.no_grad():
        probability = model(input_tensor).item()

    is_flaky = probability > 0.80
    classification = "Flaky/Infrastructure" if is_flaky else "Code Bug"

    # Evaluate Policy
    dev_tier = get_developer_tier(developer_id) if developer_id else 1
    action_recommended = evaluate_policy(classification, dev_tier)

    # Graph Integration
    graph_path = None
    if is_flaky:
        graph_path = generate_root_cause_graph(test_id)

    return {
        "test_id": test_id,
        "developer_id": developer_id,
        "trust_tier": dev_tier,
        "anomaly_probability": round(probability, 4),
        "classification": classification,
        "action_recommended": action_recommended,
        "root_cause_graph": graph_path
    }

def parse_and_process_file(file_path: str) -> list[dict]:
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
                continue

            parts = line.split(",", 1)
            if len(parts) == 2:
                task_id, trace_seq = parts[0].strip(), parts[1].strip()
            else:
                task_id, trace_seq = f"line_{line_num}", parts[0].strip()

            prediction = predict_single(test_id=task_id, trace_sequence=trace_seq)
            results.append(prediction)

        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading file '{normalized_path}': {str(e)}"
        )

def classify_preprocessed_records(limit: int = 10) -> dict:
    artifact = save_preprocessed_dataset(output_path="src/models/saved/preprocessed_dataset.json")
    records = artifact["records"][:limit]
    predictions = []

    for record in records:
        prediction = predict_single(
            test_id=record["task_id"],
            trace_sequence=record["trace_sequence"],
        )
        predictions.append(prediction)

    return {
        "total_records": len(predictions),
        "predictions": predictions,
        "metadata": artifact["metadata"],
    }

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.post("/api/v1/triage")
def triage_test_failure(payload: WebhookPayload):
    return predict_single(
        test_id=payload.test_id, 
        trace_sequence=payload.trace_sequence,
        developer_id=payload.developer_id
    )

@app.post("/api/v1/triage/file-path")
def triage_from_file_path_post(payload: FilePathRequest):
    results = parse_and_process_file(payload.file_path)
    return {"total_records": len(results), "predictions": results}

@app.get("/api/v1/triage/file-path")
def triage_from_file_path_get(
    file_path: str = Query(
        ..., description="Relative or absolute server file path (e.g. data/raw/failure_trace.csv)"
    )
):
    results = parse_and_process_file(file_path)
    return {"total_records": len(results), "predictions": results}

@app.get("/api/v1/triage/preprocessed")
def triage_preprocessed_records(limit: int = Query(10, ge=1, le=100)):
    return classify_preprocessed_records(limit=limit)

# ==========================================
# 5. SERVER RUNNER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    from src.policy_engine.database import init_db
    init_db() # Ensure DB is initialized when running main.py directly
    uvicorn.run(app, host="0.0.0.0", port=8000)

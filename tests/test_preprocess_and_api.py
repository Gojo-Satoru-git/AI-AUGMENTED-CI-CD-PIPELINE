import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.main import app
from src.models.preprocess_data import save_preprocessed_dataset


def test_preprocessed_dataset_and_api(tmp_path: Path):
    output_path = tmp_path / "preprocessed_dataset.json"

    artifact = save_preprocessed_dataset(output_path=output_path)

    assert artifact["metadata"]["vocab_size"] > 0
    assert len(artifact["records"]) > 0

    client = TestClient(app)
    response = client.get("/api/v1/triage/preprocessed")

    assert response.status_code == 200
    assert response.json()["total_records"] > 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_preprocessed_dataset_and_api(Path(tmp_dir))
        print("test_preprocess_and_api passed")

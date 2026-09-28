"""Entry point for the Test-Triage API."""
import os

import uvicorn

from src.policy_engine.database import init_db
from src.policy_engine.main import app

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

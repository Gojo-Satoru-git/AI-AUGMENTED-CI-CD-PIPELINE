import uvicorn
from src.policy_engine.main import app
from src.policy_engine.database import init_db

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Start the live dashboard.

    python scripts/serve_ui.py                      # http://127.0.0.1:8765
    python scripts/serve_ui.py --otlp http://127.0.0.1:4318/v1/traces   # also feed a real Jaeger

Needs the live model (src/models/saved_live); scripts/showcase.py builds it if missing.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-dir", default="src/models/saved_live")
    ap.add_argument("--otlp")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    mdir = ROOT / a.model_dir
    if not (mdir / "triage_lstm.pth").exists():
        sys.exit("live model missing: run `python scripts/showcase.py` once (it builds it), "
                 "or follow LIVE_PROJECT.md")
    import uvicorn
    from src.live.webapp import Session, create_app
    from src.models.predict import TriagePredictor

    predictor = TriagePredictor(ckpt=mdir / "triage_lstm.pth", vocab_path=mdir / "vocab.json")
    app = create_app(Session(predictor, otlp=a.otlp))
    url = f"http://127.0.0.1:{a.port}"
    print(f"dashboard: {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

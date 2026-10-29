# Triage / gate API.
#   docker build -t ai-gate-api .
#   docker run -p 8000:8000 -v ai-gate-state:/state ai-gate-api
# The trust state (SQLite) lives on the /state volume so tiers survive restarts.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRUST_DB_PATH=/state/trust_tiers.db \
    TRUST_RETENTION=1

WORKDIR /app

COPY requirements-service.txt .
# CPU-only torch keeps the image small; falls back to PyPI if the pin is absent
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-service.txt

COPY src ./src
COPY run_api.py .
COPY reports/security_gate.json ./reports/security_gate.json

RUN mkdir -p /state && useradd -m app && chown -R app /state /app
USER app
VOLUME /state
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health').status==200 else 1)"

CMD ["uvicorn", "src.policy_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]

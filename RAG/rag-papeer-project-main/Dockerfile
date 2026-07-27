FROM python:3.12

WORKDIR /app

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: bake the reranker model into the image ───────────────────────────
# Without this the ~34MB cross-encoder downloads on the first user query,
# turning a cold start into a visible stall.
ENV RERANK_CACHE_DIR=/app/rerank_cache
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/rerank_cache')"

# ── Layer 3: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 4: documents ─────────────────────────────────────────────────────────
COPY documents/ documents/

# ── Layer 5: application files (change most often — last for cache efficiency) ─
COPY app.py .
COPY evaluate.py .
COPY goldens.json .
COPY .streamlit/ .streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

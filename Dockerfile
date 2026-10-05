# Single-container demo (port 7860 follows the Hugging Face Spaces convention).
# Not yet built in CI; see README "Deployment".
#   docker build -t plc-rag .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... plc-rag
FROM python:3.12-slim

# Hugging Face Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch first: the default Linux wheel pulls in several GB of CUDA libraries.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies straight from pyproject.toml (+ reportlab to generate the demo manuals).
COPY --chown=user pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > requirements.txt \
 && pip install --user -r requirements.txt reportlab

COPY --chown=user . .

# Build the demo knowledge base into the image: fictional manuals -> chunks -> Chroma + BM25.
# This also caches the embedding model, so the container starts without contacting the Hub.
RUN python -m scripts.generate_demo_manuals && python -m app.ingest --rebuild

ENV HF_HUB_OFFLINE=1 \
    UI_BACKEND=embedded \
    LLM_PROVIDER=gemini \
    GEMINI_MODEL=gemini-flash-lite-latest \
    ASK_RATE_LIMIT_PER_MIN=5 \
    UI_SESSION_LIMIT_PER_MIN=3 \
    DEMO_NOTICE="Live demo on two fictional manuals (ACME FX-200 PLC and VD-500 drive) with a shared free-tier Gemini quota. Ask in English, e.g. a fault code like F-0011 or 16#8085. If the demo is busy, wait a minute and try again."

EXPOSE 7860
CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]

"""Streamlit demo UI.

Two backends:
  UI_BACKEND=api (default)  talk to a running FastAPI server over HTTP
      uvicorn app.api.main:app          # terminal 1
      streamlit run ui/streamlit_app.py # terminal 2
  UI_BACKEND=embedded       run the same FastAPI app in-process (single-process hosting,
                            e.g. one container): streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
BACKEND = os.getenv("UI_BACKEND", "api").strip().lower()
# Per-browser-session limit for public demos (0 = unlimited).
SESSION_LIMIT_PER_MIN = int(os.getenv("UI_SESSION_LIMIT_PER_MIN", "0") or 0)

EXAMPLES = {
    "qa": [
        "What does a flashing red ERROR LED mean on the FX-200?",
        "How do I replace the backup battery and when is it needed?",
        "What is event 16#8085?",
    ],
    "troubleshoot": [
        "The VD-500 trips with F-0011 every time the conveyor stops.",
        "Analog input on the FX-200 shows a DIAG LED flashing red and E-101.",
        "PROFINET I/O station keeps dropping out, CPU reports 16#39C4.",
    ],
}
CONFIDENCE_COLOR = {"high": "green", "medium": "orange", "low": "red"}

st.set_page_config(page_title="PLC Troubleshooting Assistant", page_icon="🔧", layout="wide")


@st.cache_resource(show_spinner="Loading the search index and embedding model…")
def api_client() -> httpx.Client:
    """HTTP client for the API, or an in-process TestClient wrapping the same FastAPI app."""
    if BACKEND == "embedded":
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from fastapi.testclient import TestClient

        from app.api.main import create_app

        app = create_app(warmup=False)
        try:  # load index + embedding model once, up front (not on the first question)
            _ = app.state.services.retriever
        except RuntimeError:
            pass  # no index yet: /health reports it and the sidebar shows the message
        return TestClient(app, raise_server_exceptions=False)
    return httpx.Client(base_url=API_URL, timeout=300)


BACKEND_LABEL = "in-process" if BACKEND == "embedded" else API_URL


@st.cache_data(ttl=30)
def get_json(path: str):
    response = api_client().get(path)
    response.raise_for_status()
    return response.json()


def ask(question: str, mode: str, retrieval: str, allow_general: bool) -> dict:
    response = api_client().post(
        "/ask",
        json={"question": question, "mode": mode, "retrieval": retrieval, "allow_general": allow_general},
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def session_rate_ok() -> bool:
    """Allow at most SESSION_LIMIT_PER_MIN questions per minute per browser session."""
    if SESSION_LIMIT_PER_MIN <= 0:
        return True
    now = time.time()
    recent = [t for t in st.session_state.get("ask_times", []) if now - t < 60]
    if len(recent) >= SESSION_LIMIT_PER_MIN:
        st.session_state["ask_times"] = recent
        return False
    st.session_state["ask_times"] = [*recent, now]
    return True


def citation_cards(citations: list[dict], numbered: bool) -> None:
    for i, c in enumerate(citations, start=1):
        label = f"[{c['n']}] " if numbered else f"{i}. "
        pages = f"p. {c['page']}" + (f"–{c['page_end']}" if c.get("page_end", c["page"]) != c["page"] else "")
        with st.expander(f"{label}{c['doc_title']} — {pages}"):
            if c.get("section_heading"):
                st.caption(c["section_heading"])
            st.markdown(f"> {c['snippet']}")


def render_general(data: dict) -> None:
    """Show the manuals' 'not found' result and, if present, the clearly labelled general answer."""
    if data.get("general_answer"):
        st.warning(
            "Not found in the ingested manuals. The answer below is from the model's general knowledge "
            "and has **not** been verified against your documentation."
        )
        st.markdown(data["general_answer"])
    else:
        st.warning("The manuals do not cover this question. Enable general answers to get a general-knowledge reply.")


def render_meta(data: dict) -> None:
    usage = data.get("usage") or {}
    st.caption(
        f"Model: {data.get('model')} · Latency: {data.get('latency_ms', 0) / 1000:.1f}s · "
        f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
    )


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Knowledge base")
    try:
        health = get_json("/health")
        st.success(f"Backend online ({BACKEND_LABEL}) · {health['llm_provider']} / {health['llm_model']}")
        documents = get_json("/documents") if health["index_loaded"] else []
        if not documents:
            st.warning(health.get("detail") or "No documents indexed. Run `python -m app.ingest --rebuild`.")
        for d in documents:
            st.markdown(f"**{d['doc_title']}**")
            st.caption(f"{d['pages']} pages · {d['chunks']} chunks · {d['fault_code_chunks']} fault codes")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the backend ({BACKEND_LABEL}): {exc}")

    st.divider()
    st.subheader("Quick fault-code lookup")
    code = st.text_input("Code", placeholder="e.g. F-0042 or 16#8085")
    if code:
        try:
            response = api_client().get(f"/fault-codes/{quote(code.strip(), safe='')}")
            if response.status_code == 404:
                st.info("Not found in the fault tables.")
            else:
                for e in response.json()["entries"]:
                    st.markdown(f"**{e['doc_title']}**, p. {e['page']}")
                    st.text(e["text"].split("\n", 1)[-1])
        except httpx.HTTPError as exc:
            st.error(str(exc))

# ------------------------------------------------------------------ main
st.title("🔧 PLC Troubleshooting Assistant")
st.caption(
    "Answers come from the ingested manuals with page-level citations; questions the manuals don't cover "
    "get a clearly labelled general-knowledge answer. Always follow lockout/tagout."
)
if os.getenv("DEMO_NOTICE"):
    st.info(os.getenv("DEMO_NOTICE"))

col_mode, col_retrieval = st.columns([2, 1])
mode_label = col_mode.radio("Mode", ["Q&A", "Troubleshoot"], horizontal=True)
mode = "qa" if mode_label == "Q&A" else "troubleshoot"
retrieval = col_retrieval.selectbox("Retrieval", ["hybrid", "vector", "bm25"])
allow_general = st.toggle(
    "Answer from general knowledge when the manuals don't cover the question",
    value=True,
    help="Manual-based answers always come first. General answers are labelled as not verified.",
)

st.write("Examples:")
example_cols = st.columns(len(EXAMPLES[mode]))
for col, example in zip(example_cols, EXAMPLES[mode]):
    if col.button(example, use_container_width=True):
        st.session_state["question"] = example

question = st.text_area(
    "Question or fault description",
    key="question",
    height=90,
    placeholder="Describe the symptom or enter a fault code…",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    if not session_rate_ok():
        st.warning(f"Demo limit: {SESSION_LIMIT_PER_MIN} questions per minute. Please wait a moment and try again.")
        st.stop()
    with st.spinner("Searching the manuals…"):
        try:
            st.session_state["result"] = ask(question.strip(), mode, retrieval, allow_general)
        except (RuntimeError, httpx.HTTPError) as exc:
            st.session_state["result"] = None
            st.error(f"Request failed: {exc}")

result = st.session_state.get("result")
if result and result["mode"] == "qa":
    qa = result["qa"]
    st.subheader("Answer")
    if qa["found_in_manuals"]:
        st.markdown(qa["answer"])
    else:
        render_general(qa)
    if qa["found_in_manuals"] and qa["citations"]:
        st.subheader("Sources")
        citation_cards(qa["citations"], numbered=True)
    render_meta(qa)

elif result and result["mode"] == "troubleshoot":
    ts = result["troubleshoot"]
    if not ts["ok"]:
        st.error(f"Could not produce a validated plan: {ts['error']}")
    elif not ts["result"]["found_in_manuals"]:
        st.subheader("Answer")
        render_general(ts)
    else:
        r = ts["result"]
        header = f"Fault {r['fault_code']}" if r["fault_code"] else "Troubleshooting plan"
        st.subheader(header)
        color = CONFIDENCE_COLOR[r["confidence"]]
        st.markdown(f":{color}[**Confidence: {r['confidence']}**] · Found in manuals: {'yes' if r['found_in_manuals'] else 'no'}")
        st.info(r["summary"])

        st.markdown("#### Likely causes")
        st.markdown("\n".join(f"- {cause}" for cause in r["likely_causes"] or ["—"]))
        st.markdown("#### Diagnostic steps")
        st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(r["diagnostic_steps"], start=1)))

        for note in r["safety_notes"]:
            st.warning(f"⚠️ {note}")

        if r["citations"]:
            st.markdown("#### Sources")
            citation_cards(r["citations"], numbered=False)

    with st.expander("Agent trace"):
        st.caption(f"Tool rounds: {ts['tool_rounds']} · Submission attempts: {ts['attempts']}")
        for step in ts["tool_trace"]:
            st.markdown(f"- `{step['tool']}({', '.join(f'{k}={v!r}' for k, v in step['args'].items())})` → {len(step['results'])} result(s)")
    with st.expander("Raw JSON"):
        st.json(ts["result"] or ts["raw_output"] or {})
    render_meta(ts)

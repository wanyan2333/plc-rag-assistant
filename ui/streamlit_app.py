"""Streamlit demo UI. Talks to the FastAPI backend.

    uvicorn app.api.main:app          # terminal 1
    streamlit run ui/streamlit_app.py # terminal 2
"""

from __future__ import annotations

import os
from urllib.parse import quote

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

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


@st.cache_data(ttl=30)
def get_json(path: str):
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def ask(question: str, mode: str, retrieval: str) -> dict:
    response = httpx.post(
        f"{API_URL}/ask", json={"question": question, "mode": mode, "retrieval": retrieval}, timeout=300
    )
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.json().get('detail', response.text)}")
    return response.json()


def citation_cards(citations: list[dict], numbered: bool) -> None:
    for i, c in enumerate(citations, start=1):
        label = f"[{c['n']}] " if numbered else f"{i}. "
        pages = f"p. {c['page']}" + (f"–{c['page_end']}" if c.get("page_end", c["page"]) != c["page"] else "")
        with st.expander(f"{label}{c['doc_title']} — {pages}"):
            if c.get("section_heading"):
                st.caption(c["section_heading"])
            st.markdown(f"> {c['snippet']}")


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
        st.success(f"API online · {health['llm_provider']} / {health['llm_model']}")
        documents = get_json("/documents") if health["index_loaded"] else []
        if not documents:
            st.warning(health.get("detail") or "No documents indexed. Run `python -m app.ingest --rebuild`.")
        for d in documents:
            st.markdown(f"**{d['doc_title']}**")
            st.caption(f"{d['pages']} pages · {d['chunks']} chunks · {d['fault_code_chunks']} fault codes")
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the API at {API_URL}: {exc}")

    st.divider()
    st.subheader("Quick fault-code lookup")
    code = st.text_input("Code", placeholder="e.g. F-0042 or 16#8085")
    if code:
        try:
            response = httpx.get(f"{API_URL}/fault-codes/{quote(code.strip(), safe='')}", timeout=10)
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
st.caption("Answers come only from the ingested manuals, with page-level citations. Always follow lockout/tagout.")

col_mode, col_retrieval = st.columns([2, 1])
mode_label = col_mode.radio("Mode", ["Q&A", "Troubleshoot"], horizontal=True)
mode = "qa" if mode_label == "Q&A" else "troubleshoot"
retrieval = col_retrieval.selectbox("Retrieval", ["hybrid", "vector", "bm25"])

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
    with st.spinner("Searching the manuals…"):
        try:
            st.session_state["result"] = ask(question.strip(), mode, retrieval)
        except (RuntimeError, httpx.HTTPError) as exc:
            st.session_state["result"] = None
            st.error(f"Request failed: {exc}")

result = st.session_state.get("result")
if result and result["mode"] == "qa":
    qa = result["qa"]
    st.subheader("Answer")
    if not qa["found_in_manuals"]:
        st.warning("The manuals do not cover this question.")
    st.markdown(qa["answer"])
    if qa["citations"]:
        st.subheader("Sources")
        citation_cards(qa["citations"], numbered=True)
    render_meta(qa)

elif result and result["mode"] == "troubleshoot":
    ts = result["troubleshoot"]
    if not ts["ok"]:
        st.error(f"Could not produce a validated plan: {ts['error']}")
    else:
        r = ts["result"]
        header = f"Fault {r['fault_code']}" if r["fault_code"] else "Troubleshooting plan"
        st.subheader(header)
        color = CONFIDENCE_COLOR[r["confidence"]]
        st.markdown(f":{color}[**Confidence: {r['confidence']}**] · Found in manuals: {'yes' if r['found_in_manuals'] else 'no'}")
        st.info(r["summary"])

        left, right = st.columns(2)
        with left:
            st.markdown("#### Likely causes")
            for cause in r["likely_causes"] or ["—"]:
                st.markdown(f"- {cause}")
        with right:
            st.markdown("#### Diagnostic steps")
            for i, step in enumerate(r["diagnostic_steps"], start=1):
                st.checkbox(f"{i}. {step}", key=f"step_{i}_{hash(step)}")

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

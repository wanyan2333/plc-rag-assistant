import logging

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.generation.llm import FakeLLM
from app.services import Services


@pytest.fixture(scope="module")
def client(built_index, settings, embedder):
    services = Services(settings, embedder=embedder, llm=FakeLLM())
    with TestClient(create_app(services, warmup=False)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_loaded"] and body["chunks"] > 0
    assert body["llm_provider"] == "fake"


def test_documents(client):
    docs = client.get("/documents").json()
    assert docs[0]["doc_id"] == "fx100-manual"
    assert docs[0]["fault_code_chunks"] == 5


def test_fault_code_lookup(client):
    response = client.get("/fault-codes/E-101")
    assert response.status_code == 200
    body = response.json()
    assert body["found"] and body["entries"][0]["page"] == 3
    assert "Overcurrent on axis 1" in body["entries"][0]["text"]


def test_fault_code_lookup_is_notation_insensitive(client):
    assert client.get("/fault-codes/e101").status_code == 200
    assert client.get("/fault-codes/16%238085").json()["entries"][0]["fault_code"] == "16#8085"


def test_fault_code_not_found(client):
    assert client.get("/fault-codes/X-404").status_code == 404


def test_ask_qa(client, caplog):
    with caplog.at_level(logging.INFO, logger="plc_rag.requests"):
        response = client.post("/ask", json={"question": "What does fault code E-205 mean?", "mode": "qa"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "qa" and body["troubleshoot"] is None
    qa = body["qa"]
    assert "Encoder signal lost" in qa["answer"]
    assert qa["citations"][0]["page"] == 3
    log_line = next(r.message for r in caplog.records if '"event": "ask"' in r.message)
    assert "latency_ms" in log_line and "chunk_ids" in log_line and "usage" in log_line


@pytest.mark.parametrize("retrieval", ["vector", "bm25", "hybrid"])
def test_ask_qa_retrieval_modes(client, retrieval):
    body = client.post("/ask", json={"question": "lockout tagout cabinet", "mode": "qa", "retrieval": retrieval}).json()
    assert body["retrieval"] == retrieval
    assert body["qa"]["retrieval_mode"] == retrieval


def test_ask_troubleshoot(client):
    response = client.post("/ask", json={"question": "Drive stopped with F-0042", "mode": "troubleshoot"})
    assert response.status_code == 200
    ts = response.json()["troubleshoot"]
    assert ts["ok"]
    assert ts["result"]["fault_code"] == "F-0042"
    assert ts["result"]["diagnostic_steps"]
    assert ts["tool_trace"][0]["tool"] == "lookup_fault_code"


def test_ask_off_topic_gets_labelled_general_answer(client):
    body = client.post("/ask", json={"question": "Are you Gemini?", "mode": "qa"}).json()["qa"]
    assert not body["found_in_manuals"]
    assert body["general_answer"]
    assert body["citations"] == []

    body = client.post("/ask", json={"question": "Are you Gemini?", "mode": "qa", "allow_general": False}).json()["qa"]
    assert body["general_answer"] is None


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "", "mode": "qa"}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "chat"}).status_code == 422
    assert client.post("/ask", json={"question": "x", "retrieval": "magic"}).status_code == 422


def test_rate_limiter_sliding_window():
    from app.api.main import RateLimiter

    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.allow() and limiter.allow()
    assert not limiter.allow()
    now[0] = 59.9
    assert not limiter.allow()
    now[0] = 60.0  # first call left the window
    assert limiter.allow()
    assert RateLimiter(0).allow()  # 0 = unlimited


def test_ask_rate_limit_returns_429(built_index, settings, embedder):
    limited = settings.model_copy(update={"ask_rate_limit_per_min": 1})
    with TestClient(create_app(Services(limited, embedder=embedder, llm=FakeLLM()), warmup=False)) as c:
        assert c.post("/ask", json={"question": "What does fault code E-205 mean?"}).status_code == 200
        response = c.post("/ask", json={"question": "What does fault code E-205 mean?"})
        assert response.status_code == 429
        assert "try again" in response.json()["detail"]
        assert c.get("/fault-codes/E-101").status_code == 200  # lookups are not rate limited


def test_missing_index_is_reported(tmp_path):
    settings = Settings(_env_file=None, llm_provider="fake", embedding_backend="hash", index_dir=tmp_path)
    with TestClient(create_app(Services(settings, llm=FakeLLM()), warmup=False)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/fault-codes/E-101").status_code == 503
        assert c.post("/ask", json={"question": "x"}).status_code == 503

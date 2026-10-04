import json

import pytest
from pydantic import ValidationError

from app.generation.llm import FakeLLM, LLMError, fake_response
from app.generation.troubleshoot import (
    SUBMIT_RESULT,
    TroubleshootResult,
    tool_call,
    troubleshoot,
    validate_submission,
)
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex
from tests.fixtures.make_fixture_pdf import FIXTURE_TITLE

VALID = {
    "fault_code": "E-101",
    "summary": "Overcurrent on axis 1.",
    "likely_causes": ["Motor cable short circuit", "Drive gain too high"],
    "diagnostic_steps": ["Apply lockout/tagout", "Check motor cable insulation", "Reduce drive gain"],
    "safety_notes": ["Apply lockout/tagout before touching wiring."],
    "citations": [{"doc_title": FIXTURE_TITLE, "page": 3, "snippet": "E-101 Overcurrent on axis 1"}],
    "confidence": "high",
    "found_in_manuals": True,
}


@pytest.fixture(scope="module")
def retriever(built_index, embedder):
    return HybridRetriever(ChunkStore.load(built_index), VectorIndex(built_index, embedder))


def lookup(code="E-101", call_id="c1"):
    return fake_response(tool_calls=[tool_call("lookup_fault_code", {"code": code}, call_id)])


def submit(payload, call_id="s1"):
    return fake_response(tool_calls=[tool_call("submit_troubleshooting_result", payload, call_id)])


# ---------------------------------------------------------------- schema

def test_schema_accepts_valid_result():
    result = TroubleshootResult.model_validate(VALID)
    assert result.confidence == "high"
    assert result.citations[0].page == 3


@pytest.mark.parametrize(
    "patch",
    [
        {"confidence": "certain"},
        {"diagnostic_steps": "do things"},
        {"citations": [{"doc_title": "x", "page": 0, "snippet": "s"}]},
        {"found_in_manuals": True, "diagnostic_steps": []},
        {"summary": ""},
    ],
)
def test_schema_rejects_invalid(patch):
    with pytest.raises(ValidationError):
        TroubleshootResult.model_validate({**VALID, **patch})


def test_schema_requires_all_fields():
    payload = dict(VALID)
    payload.pop("safety_notes")
    with pytest.raises(ValidationError):
        TroubleshootResult.model_validate(payload)


def test_not_found_result_is_valid_without_steps():
    TroubleshootResult.model_validate(
        {**VALID, "found_in_manuals": False, "confidence": "low", "diagnostic_steps": [], "citations": [], "likely_causes": []}
    )


def test_submit_tool_schema_is_the_pydantic_schema():
    props = SUBMIT_RESULT.parameters["properties"]
    assert set(props) == set(TroubleshootResult.model_fields)


def test_grounding_rejects_citation_not_returned_by_tools():
    result, error = validate_submission(VALID, seen={})
    assert result is None and "Grounding check failed" in error


# ---------------------------------------------------------------- loop

def test_happy_path_with_default_fake_policy(retriever, settings):
    response = troubleshoot("Axis 1 stopped, display shows E-101", retriever, FakeLLM(), settings)
    assert response.ok, response.error
    assert response.result.fault_code == "E-101"
    assert response.result.found_in_manuals
    assert response.tool_trace[0]["tool"] == "lookup_fault_code"
    assert response.attempts == 1


def test_validation_failure_is_fed_back_and_retried_once(retriever, settings):
    bad = {**VALID, "confidence": "certain"}
    llm = FakeLLM(script=[lookup(), submit(bad, "s1"), submit(VALID, "s2")])
    response = troubleshoot("E-101 on axis 1", retriever, llm, settings)
    assert response.ok
    assert response.attempts == 2
    # The third model call must have seen the validation error as an error tool_result.
    feedback = llm.calls[2]["messages"][-1].tool_results[0]
    assert feedback.is_error and feedback.call_id == "s1"
    assert "confidence" in feedback.content


def test_second_failure_returns_error_structure(retriever, settings):
    bad = {**VALID, "confidence": "certain"}
    llm = FakeLLM(script=[lookup(), submit(bad, "s1"), submit(bad, "s2")])
    response = troubleshoot("E-101 on axis 1", retriever, llm, settings)
    assert not response.ok
    assert response.result is None
    assert "failed validation twice" in response.error
    assert response.raw_output == bad
    json.dumps(response.model_dump())  # serialisable for the API


def test_ungrounded_citation_triggers_retry(retriever, settings):
    fabricated = {**VALID, "citations": [{"doc_title": "Some Other Manual", "page": 99, "snippet": "made up"}]}
    llm = FakeLLM(script=[lookup(), submit(fabricated), submit(VALID, "s2")])
    response = troubleshoot("E-101", retriever, llm, settings)
    assert response.ok and response.attempts == 2
    feedback = llm.calls[2]["messages"][-1].tool_results[0].content
    assert "Allowed sources" in feedback and FIXTURE_TITLE in feedback


def test_text_reply_without_submission_is_retried(retriever, settings):
    llm = FakeLLM(script=[lookup(), fake_response("Check the cable."), submit(VALID)])
    response = troubleshoot("E-101", retriever, llm, settings)
    assert response.ok and response.attempts == 2


def test_tool_rounds_are_capped(retriever, settings):
    search = lambda i: fake_response(tool_calls=[tool_call("search_manuals", {"query": f"fault {i}"}, f"q{i}")])
    llm = FakeLLM(script=[search(i) for i in range(settings.max_tool_rounds)] + [lookup(), submit(VALID)])
    response = troubleshoot("E-101", retriever, llm, settings)
    assert response.tool_rounds == settings.max_tool_rounds
    final_tools = [t.name for t in llm.calls[settings.max_tool_rounds]["tools"]]
    assert final_tools == ["submit_troubleshooting_result"]
    # The extra lookup after the budget was refused, so the E-101 citation is ungrounded unless search found page 3.
    assert response.attempts >= 1


def test_llm_error_does_not_crash(retriever, settings):
    def boom(*_):
        raise LLMError("service unavailable")

    response = troubleshoot("E-101", retriever, FakeLLM(script=[boom]), settings)
    assert not response.ok and "service unavailable" in response.error


def test_unknown_problem_reports_not_found(retriever, settings):
    llm = FakeLLM(script=[
        fake_response(tool_calls=[tool_call("lookup_fault_code", {"code": "Z-999"})]),
        submit({**VALID, "fault_code": "Z-999", "found_in_manuals": False, "confidence": "low",
                "summary": "Not found in the provided manuals.", "likely_causes": [], "diagnostic_steps": [], "citations": []}),
    ])
    response = troubleshoot("Z-999 on the HMI", retriever, llm, settings, allow_general=True)
    assert response.ok and not response.result.found_in_manuals
    assert json.loads(llm.calls[1]["messages"][-1].tool_results[0].content)["found"] is False
    assert response.general_answer and "Z-999" in response.general_answer
    assert response.usage["input_tokens"] >= 300  # 2 loop calls + 1 general call (100 each in FakeLLM)


def test_no_general_answer_when_manuals_cover_it(retriever, settings):
    response = troubleshoot("Axis 1 stopped, display shows E-101", retriever, FakeLLM(), settings, allow_general=True)
    assert response.ok and response.result.found_in_manuals
    assert response.general_answer is None

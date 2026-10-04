import pytest

from app.generation.citations import best_snippet, cited_numbers
from app.generation.llm import FakeLLM, fake_response, inline_refs
from app.generation.prompts import NOT_FOUND, QA_SYSTEM
from app.generation.qa import answer_question
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import ChunkStore
from app.retrieval.vector import VectorIndex


@pytest.fixture(scope="module")
def retriever(built_index, embedder):
    return HybridRetriever(ChunkStore.load(built_index), VectorIndex(built_index, embedder))


def test_cited_numbers():
    assert cited_numbers("A [1]. B [2, 3]. C [1][4].") == [1, 2, 3, 4]
    assert cited_numbers("no citations") == []


def test_best_snippet_picks_relevant_sentence():
    text = "The fan cools the unit. Replace the CR2032 battery every five years. Clean the filter."
    assert best_snippet(text, "battery replacement CR2032").startswith("Replace the CR2032 battery")


def test_prompt_contains_sources_and_rules(retriever):
    llm = FakeLLM(script=[fake_response("Apply lockout/tagout before opening the cabinet [1].")])
    answer_question("What must be done before opening the cabinet?", retriever, llm, k=3)
    call = llm.calls[0]
    assert call["system"] == QA_SYSTEM
    assert NOT_FOUND in call["system"] and "lockout/tagout" in call["system"]
    prompt = call["messages"][0].text
    assert '<source id="1"' in prompt and "<question>" in prompt


def test_answer_maps_citations_to_chunks(retriever):
    llm = FakeLLM(script=[fake_response("Apply lockout/tagout first [1]. Then check the LED [2]. Ignore [9].")])
    result = answer_question("What must be done before opening the cabinet?", retriever, llm, k=3)
    assert [c.n for c in result.citations] == [1, 2]  # [9] is out of range and dropped
    first = retriever.store.get(result.retrieved_chunk_ids[0])
    assert result.citations[0].doc_title == first.doc_title
    assert result.citations[0].page == first.page
    assert result.found_in_manuals
    assert result.usage["input_tokens"] > 0


def test_default_fake_policy_answers_with_citation(retriever):
    result = answer_question("What does fault code E-205 mean?", retriever, FakeLLM(), k=4)
    assert "Encoder signal lost" in result.answer
    assert result.citations and result.citations[0].page == 3
    assert result.found_in_manuals


def test_unanswerable_question_is_refused(retriever):
    result = answer_question("What is the warranty period for the gearbox oil?", retriever, FakeLLM(), k=4)
    assert result.answer.startswith(NOT_FOUND)
    assert not result.found_in_manuals
    assert result.citations == []


def test_general_answer_only_when_manuals_have_nothing(retriever):
    off_topic = "What is the warranty period for the gearbox oil?"
    result = answer_question(off_topic, retriever, FakeLLM(), k=4, allow_general=True)
    assert not result.found_in_manuals
    assert result.answer.startswith(NOT_FOUND)  # the grounded verdict is kept
    assert result.general_answer and "General-knowledge answer" in result.general_answer

    grounded = answer_question("What does fault code E-205 mean?", retriever, FakeLLM(), k=4, allow_general=True)
    assert grounded.found_in_manuals and grounded.general_answer is None


def test_general_answer_is_off_by_default(retriever):
    llm = FakeLLM()
    result = answer_question("What is the warranty period for the gearbox oil?", retriever, llm, k=4)
    assert result.general_answer is None
    assert len(llm.calls) == 1  # no extra model call


def test_general_prompt_is_labelled_and_separate(retriever):
    llm = FakeLLM()
    answer_question("Who are you?", retriever, llm, k=4, allow_general=True)
    system = llm.calls[-1]["system"]
    assert "general knowledge" in system and "same language as the question" in system
    assert "<sources>" not in llm.calls[-1]["messages"][0].text


def test_fallback_llm_uses_backup_when_primary_fails(retriever):
    from app.generation.llm import FallbackLLM, LLMError

    def outage(*_):
        raise LLMError("503 overloaded")

    backup = FakeLLM(script=[fake_response("Apply lockout/tagout first [1].")])
    llm = FallbackLLM(FakeLLM(script=[outage]), backup)
    result = answer_question("What must be done before opening the cabinet?", retriever, llm, k=3)
    assert result.answer.startswith("Apply lockout/tagout")
    assert len(backup.calls) == 1


def test_inline_refs_removes_defs():
    schema = {
        "type": "object",
        "properties": {"c": {"type": "array", "items": {"$ref": "#/$defs/C"}}},
        "$defs": {"C": {"type": "object", "title": "C", "properties": {"x": {"type": "string", "title": "X"}}}},
    }
    flat = inline_refs(schema)
    assert "$defs" not in flat
    assert flat["properties"]["c"]["items"]["properties"]["x"] == {"type": "string"}

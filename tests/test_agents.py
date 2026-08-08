"""Unit-тесты четырёх агентов: контракт выхода, обработка ошибок и вход промпта.

Реальных вызовов LLM здесь нет — get_llm подменяется в модуле каждого агента.
"""

import pytest
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

import agents
from agents import get_llm
from agents.classifier import Classification, Priority, Topic, classification_node
from agents.compliance import COMPLIANCE_MAX_TOKENS, ComplianceIssue, ComplianceResult, Severity, compliance_node
from agents.quality import QUALITY_MAX_TOKENS, QualityChecklist, QualityScore, quality_node
from agents.summarizer import SUMMARY_MAX_TOKENS, Summary, summarization_node

pytestmark = pytest.mark.usefixtures("fake_settings")


class TestClassifier:
    def test_empty_transcript_returns_default_without_llm(self, forbid_llm):
        result = classification_node({"transcript": []})

        assert result == {"classification": {"topic": Topic.OTHER, "priority": Priority.LOW}}

    def test_maps_llm_result_to_state(self, patch_llm, transcript):
        patch_llm("agents.classifier", Classification(topic=Topic.CARDS, priority=Priority.CRITICAL))

        result = classification_node({"transcript": transcript})

        assert result == {"classification": {"topic": Topic.CARDS, "priority": Priority.CRITICAL}}

    def test_passes_dialog_to_prompt(self, patch_llm, transcript):
        call = patch_llm("agents.classifier", Classification(topic=Topic.CARDS, priority=Priority.HIGH))

        classification_node({"transcript": transcript})

        assert call.inputs == {"dialog": transcript}
        assert call.schema is Classification

    def test_uses_default_token_limit(self, patch_llm, transcript):
        call = patch_llm("agents.classifier", Classification(topic=Topic.OTHER, priority=Priority.LOW))

        classification_node({"transcript": transcript})

        assert call.max_completion_tokens == 2000

    def test_llm_failure_becomes_422(self, patch_llm, transcript):
        patch_llm("agents.classifier", error=RuntimeError("upstream is down"))

        with pytest.raises(HTTPException) as exc_info:
            classification_node({"transcript": transcript})

        assert exc_info.value.status_code == 422
        assert "Failed to classify dialog" in exc_info.value.detail

    def test_rejects_unknown_topic(self):
        with pytest.raises(ValidationError):
            Classification(topic="mortgages", priority="low")


class TestQuality:
    def test_empty_transcript_returns_zero_score_without_llm(self, forbid_llm):
        result = quality_node({"transcript": []})

        assert result["quality_score"]["total"] == 0
        assert result["quality_score"]["checklist"] == {
            "greeting": False,
            "need_detection": False,
            "solution_provided": False,
            "farewell": False,
        }
        assert result["quality_score"]["comment"]

    def test_flattens_checklist_to_dict(self, patch_llm, transcript):
        patch_llm(
            "agents.quality",
            QualityScore(
                total=78,
                checklist=QualityChecklist(
                    greeting=True, need_detection=True, solution_provided=True, farewell=False
                ),
                comment="Оператор не попрощался.",
            ),
        )

        result = quality_node({"transcript": transcript})

        assert result == {
            "quality_score": {
                "total": 78,
                "checklist": {
                    "greeting": True,
                    "need_detection": True,
                    "solution_provided": True,
                    "farewell": False,
                },
                "comment": "Оператор не попрощался.",
            }
        }

    def test_receives_classification_from_upstream(self, patch_llm, transcript):
        classification = {"topic": Topic.CARDS, "priority": Priority.HIGH}
        call = patch_llm(
            "agents.quality",
            QualityScore(
                total=50,
                checklist=QualityChecklist(
                    greeting=True, need_detection=True, solution_provided=False, farewell=False
                ),
                comment="Решение не предложено.",
            ),
        )

        quality_node({"transcript": transcript, "classification": classification})

        assert call.inputs == {"dialog": transcript, "classification": classification}
        assert call.max_completion_tokens == QUALITY_MAX_TOKENS

    def test_llm_failure_becomes_422(self, patch_llm, transcript):
        patch_llm("agents.quality", error=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            quality_node({"transcript": transcript})

        assert exc_info.value.status_code == 422
        assert "Failed to evaluate dialog quality" in exc_info.value.detail

    @pytest.mark.parametrize("total", [-1, 101])
    def test_total_is_bounded_to_0_100(self, total):
        checklist = QualityChecklist(
            greeting=True, need_detection=True, solution_provided=True, farewell=True
        )
        with pytest.raises(ValidationError):
            QualityScore(total=total, checklist=checklist, comment="—")


class TestCompliance:
    def test_empty_transcript_passes_without_llm(self, forbid_llm):
        result = compliance_node({"transcript": []})

        assert result == {"compliance": {"passed": True, "issues": []}}

    def test_serializes_issues(self, patch_llm, transcript):
        issue = ComplianceIssue(
            rule="Запрос секретных данных клиента",
            severity=Severity.HIGH,
            quote="Назовите код из SMS",
            comment="Оператор не вправе запрашивать код из SMS.",
        )
        patch_llm("agents.compliance", ComplianceResult(passed=False, issues=[issue]))

        result = compliance_node({"transcript": transcript})

        assert result["compliance"]["passed"] is False
        assert result["compliance"]["issues"] == [issue.model_dump()]

    def test_passed_is_recomputed_when_llm_contradicts_itself(self, patch_llm, transcript):
        """LLM иногда отдаёт passed=true вместе с найденными нарушениями — узел это чинит."""
        issue = ComplianceIssue(
            rule="Гарантия одобрения",
            severity=Severity.MEDIUM,
            quote="Вам точно одобрят",
            comment="Решение принимает банк.",
        )
        patch_llm("agents.compliance", ComplianceResult(passed=True, issues=[issue]))

        result = compliance_node({"transcript": transcript})

        assert result["compliance"]["passed"] is False

    def test_passed_is_recomputed_when_issues_are_empty(self, patch_llm, transcript):
        patch_llm("agents.compliance", ComplianceResult(passed=False, issues=[]))

        result = compliance_node({"transcript": transcript})

        assert result["compliance"] == {"passed": True, "issues": []}

    def test_receives_classification_and_token_limit(self, patch_llm, transcript):
        classification = {"topic": Topic.CREDITS, "priority": Priority.MEDIUM}
        call = patch_llm("agents.compliance", ComplianceResult(passed=True, issues=[]))

        compliance_node({"transcript": transcript, "classification": classification})

        assert call.inputs == {"dialog": transcript, "classification": classification}
        assert call.max_completion_tokens == COMPLIANCE_MAX_TOKENS

    def test_llm_failure_becomes_422(self, patch_llm, transcript):
        patch_llm("agents.compliance", error=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            compliance_node({"transcript": transcript})

        assert exc_info.value.status_code == 422
        assert "Failed to check dialog compliance" in exc_info.value.detail


class TestSummarizer:
    def test_empty_transcript_returns_default_without_llm(self, forbid_llm):
        result = summarization_node({"transcript": []})

        assert result["action_items"] == []
        assert result["summary"]

    def test_maps_llm_result_to_state(self, patch_llm, transcript):
        patch_llm(
            "agents.summarizer",
            Summary(summary="Клиент потерял карту, оператор её заблокировал.", action_items=["Перевыпустить карту"]),
        )

        result = summarization_node({"transcript": transcript})

        assert result == {
            "summary": "Клиент потерял карту, оператор её заблокировал.",
            "action_items": ["Перевыпустить карту"],
        }

    def test_receives_all_upstream_results(self, patch_llm, transcript):
        state = {
            "transcript": transcript,
            "classification": {"topic": Topic.CARDS, "priority": Priority.HIGH},
            "quality_score": {"total": 80},
            "compliance": {"passed": True, "issues": []},
        }
        call = patch_llm("agents.summarizer", Summary(summary="Резюме.", action_items=[]))

        summarization_node(state)

        assert call.inputs == {
            "dialog": transcript,
            "classification": state["classification"],
            "quality_score": state["quality_score"],
            "compliance": state["compliance"],
        }
        assert call.max_completion_tokens == SUMMARY_MAX_TOKENS

    def test_tolerates_missing_upstream_keys(self, patch_llm, transcript):
        call = patch_llm("agents.summarizer", Summary(summary="Резюме.", action_items=[]))

        result = summarization_node({"transcript": transcript})

        assert call.inputs == {
            "dialog": transcript,
            "classification": {},
            "quality_score": {},
            "compliance": {},
        }
        assert result["summary"] == "Резюме."

    def test_llm_failure_becomes_422(self, patch_llm, transcript):
        patch_llm("agents.summarizer", error=RuntimeError("boom"))

        with pytest.raises(HTTPException) as exc_info:
            summarization_node({"transcript": transcript})

        assert exc_info.value.status_code == 422
        assert "Failed to summarize dialog" in exc_info.value.detail


class TestGetLLM:
    def test_builds_deterministic_client(self):
        llm = get_llm(max_completion_tokens=1234)

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "test/model"
        assert llm.temperature == 0.0
        # langchain-openai хранит max_completion_tokens в поле max_tokens.
        assert llm.max_tokens == 1234

    def test_missing_api_key_is_400(self, monkeypatch):
        monkeypatch.setattr(agents.settings, "OPENROUTER_API_KEY", None)

        with pytest.raises(HTTPException) as exc_info:
            get_llm()

        assert exc_info.value.status_code == 400
        assert "OPENROUTER_API_KEY" in exc_info.value.detail

    def test_missing_model_is_400(self, monkeypatch):
        monkeypatch.setattr(agents.settings, "OPENROUTER_MODEL", None)

        with pytest.raises(HTTPException) as exc_info:
            get_llm()

        assert exc_info.value.status_code == 400
        assert "OPENROUTER_MODEL" in exc_info.value.detail

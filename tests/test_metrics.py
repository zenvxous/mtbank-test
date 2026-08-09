"""Тесты Prometheus-метрик.

Счётчики глобальны на процесс, поэтому проверяем не абсолютные значения,
а дельту до/после вызова — иначе тесты зависят от порядка выполнения.
"""

import pytest
from prometheus_client import REGISTRY

from api import metrics


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else value


@pytest.fixture
def enabled_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """METRICS_ENABLED мог быть выключен через .env в корне репозитория."""
    monkeypatch.setattr(metrics.settings, "METRICS_ENABLED", True, raising=False)


def test_metrics_endpoint_exposes_domain_metrics(api_client):
    response = api_client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    for name in (
        "calls_analyzed_total",
        "calls_by_topic_total",
        "call_quality_score",
        "compliance_checks_total",
        "agent_duration_seconds",
    ):
        assert name in body


def test_record_analysis_increments_domain_metrics(enabled_metrics, analyze_result):
    before = {
        "topic": sample("calls_by_topic_total", {"topic": "credits"}),
        "priority": sample("calls_by_priority_total", {"priority": "medium"}),
        "quality_count": sample("call_quality_score_count"),
        "quality_sum": sample("call_quality_score_sum"),
        "greeting": sample("call_quality_checklist_passed_total", {"item": "greeting"}),
        "farewell": sample("call_quality_checklist_passed_total", {"item": "farewell"}),
        "passed": sample("compliance_checks_total", {"result": "passed"}),
    }

    metrics.record_analysis(analyze_result)

    assert sample("calls_by_topic_total", {"topic": "credits"}) == before["topic"] + 1
    assert sample("calls_by_priority_total", {"priority": "medium"}) == before["priority"] + 1
    assert sample("call_quality_score_count") == before["quality_count"] + 1
    assert sample("call_quality_score_sum") == before["quality_sum"] + 78
    # greeting=True попадает в счётчик, farewell=False — нет.
    assert sample("call_quality_checklist_passed_total", {"item": "greeting"}) == before["greeting"] + 1
    assert sample("call_quality_checklist_passed_total", {"item": "farewell"}) == before["farewell"]
    assert sample("compliance_checks_total", {"result": "passed"}) == before["passed"] + 1


def test_record_analysis_counts_issues_by_severity(enabled_metrics, analyze_result):
    before_failed = sample("compliance_checks_total", {"result": "failed"})
    before_high = sample("compliance_issues_total", {"severity": "high"})

    metrics.record_analysis(
        {
            **analyze_result,
            "compliance": {
                "passed": False,
                "issues": [
                    {"rule": "Запрос секретных данных", "severity": "high", "quote": "...", "comment": "..."},
                    {"rule": "Гарантия одобрения", "severity": "medium", "quote": "...", "comment": "..."},
                ],
            },
        }
    )

    assert sample("compliance_checks_total", {"result": "failed"}) == before_failed + 1
    assert sample("compliance_issues_total", {"severity": "high"}) == before_high + 1


def test_record_analysis_survives_malformed_result(enabled_metrics):
    """Сбор метрик не должен ронять /analyze на неожиданной форме данных."""
    metrics.record_analysis({})
    metrics.record_analysis({"classification": None, "quality_score": None, "compliance": None})
    metrics.record_analysis({"compliance": {"passed": False, "issues": ["не словарь"]}})


def test_observe_asr_records_realtime_factor(enabled_metrics):
    before = sample("asr_realtime_factor_count")

    metrics.observe_asr(duration_s=60.0, audio_duration_s=120.0)

    assert sample("asr_realtime_factor_count") == before + 1
    assert sample("asr_transcription_duration_seconds_count") > 0


def test_observe_asr_skips_realtime_factor_without_audio_duration(enabled_metrics):
    """Пустой транскрипт даёт duration=0 — деления на ноль быть не должно."""
    before = sample("asr_realtime_factor_count")

    metrics.observe_asr(duration_s=1.0, audio_duration_s=0.0)

    assert sample("asr_realtime_factor_count") == before


def test_observe_role_assignment_records_method_and_margin(enabled_metrics):
    before_markers = sample("diarization_role_assignment_total", {"method": "markers"})
    before_margin = sample("diarization_role_margin_count")
    before_unknown = sample("diarization_unknown_segments_total")

    metrics.observe_role_assignment("markers", margin=2.5, unknown_segments=3)

    assert sample("diarization_role_assignment_total", {"method": "markers"}) == before_markers + 1
    assert sample("diarization_role_margin_count") == before_margin + 1
    assert sample("diarization_unknown_segments_total") == before_unknown + 3


def test_observe_role_assignment_skips_zero_unknown_segments(enabled_metrics):
    before = sample("diarization_unknown_segments_total")

    metrics.observe_role_assignment("fallback_order", margin=0.0, unknown_segments=0)

    assert sample("diarization_unknown_segments_total") == before


def test_observe_agent_skipped_does_not_record_duration(enabled_metrics):
    before_runs = sample("agent_runs_total", {"agent": "classifier", "outcome": "skipped"})
    before_duration = sample("agent_duration_seconds_count", {"agent": "classifier"})

    metrics.observe_agent("classifier", "skipped", 0.0)

    assert sample("agent_runs_total", {"agent": "classifier", "outcome": "skipped"}) == before_runs + 1
    assert sample("agent_duration_seconds_count", {"agent": "classifier"}) == before_duration


def test_helpers_are_noop_when_disabled(monkeypatch: pytest.MonkeyPatch, analyze_result):
    monkeypatch.setattr(metrics.settings, "METRICS_ENABLED", False, raising=False)
    before = sample("calls_analyzed_total", {"status": "success"})

    metrics.record_analysis(analyze_result)
    metrics.observe_call("success", 1.0)
    metrics.observe_asr(1.0, 2.0)
    metrics.observe_diarization(1.0)
    metrics.observe_role_assignment("markers", 1.0, 1)
    metrics.record_diarization_failure()
    metrics.observe_agent("classifier", "completed", 1.0)

    assert sample("calls_analyzed_total", {"status": "success"}) == before


def test_analyze_endpoint_feeds_metrics(enabled_metrics, api_client):
    before_calls = sample("calls_analyzed_total", {"status": "success"})
    before_topic = sample("calls_by_topic_total", {"topic": "credits"})

    with open("test_data/deposit_question.ogg", "rb") as f:
        response = api_client.post("/analyze", files={"file": ("deposit_question.ogg", f, "audio/ogg")})

    assert response.status_code == 200
    assert sample("calls_analyzed_total", {"status": "success"}) == before_calls + 1
    assert sample("calls_by_topic_total", {"topic": "credits"}) == before_topic + 1


def test_rejected_upload_counts_as_error(enabled_metrics, api_client):
    before = sample("calls_analyzed_total", {"status": "error"})

    response = api_client.post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")})

    assert response.status_code == 400
    assert sample("calls_analyzed_total", {"status": "error"}) == before + 1

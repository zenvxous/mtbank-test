"""Prometheus-метрики пайплайна анализа звонков.

Метрики живут в дефолтном REGISTRY процесса `api`. Multiprocess-режим
prometheus_client не используется: uvicorn запускается одним воркером
(см. CMD в Dockerfile), поэтому /metrics отдаёт полное состояние процесса.

Состояние счётчиков не переживает рестарт `api` — историю хранит TSDB
Prometheus, а rate()/increase() корректно обрабатывают обнуление счётчика.

Все хелперы этого модуля не должны ронять запрос: сбор метрик обёрнут в
try/except, ошибка пишется в лог и проглатывается.
"""

import structlog
from prometheus_client import Counter, Histogram

from api.config import settings

log = structlog.get_logger(__name__)

CHECKLIST_ITEMS = ("greeting", "need_detection", "solution_provided", "farewell")

DURATION_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)

calls_analyzed_total = Counter(
    "calls_analyzed_total",
    "Количество обработанных звонков",
    ["status"],
)

calls_by_topic_total = Counter(
    "calls_by_topic_total",
    "Количество звонков по тематике обращения",
    ["topic"],
)

calls_by_priority_total = Counter(
    "calls_by_priority_total",
    "Количество звонков по приоритету обращения",
    ["priority"],
)

call_quality_score = Histogram(
    "call_quality_score",
    "Оценка качества обслуживания, 0-100",
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
)

call_quality_checklist_passed_total = Counter(
    "call_quality_checklist_passed_total",
    "Количество звонков, где пункт чеклиста оператора выполнен",
    ["item"],
)

compliance_checks_total = Counter(
    "compliance_checks_total",
    "Результаты проверки комплаенса",
    ["result"],
)

compliance_issues_total = Counter(
    "compliance_issues_total",
    "Найденные нарушения регламента по критичности",
    ["severity"],
)

call_analysis_duration_seconds = Histogram(
    "call_analysis_duration_seconds",
    "Полное время обработки звонка: ASR + диаризация + агенты",
    buckets=DURATION_BUCKETS,
)

asr_transcription_duration_seconds = Histogram(
    "asr_transcription_duration_seconds",
    "Время транскрибации faster-whisper",
    buckets=DURATION_BUCKETS,
)

asr_audio_duration_seconds = Histogram(
    "asr_audio_duration_seconds",
    "Длительность обработанной аудиозаписи",
    buckets=DURATION_BUCKETS,
)

asr_realtime_factor = Histogram(
    "asr_realtime_factor",
    "RTF: время транскрибации, делённое на длительность записи",
    buckets=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)

diarization_duration_seconds = Histogram(
    "diarization_duration_seconds",
    "Время диаризации pyannote",
    buckets=DURATION_BUCKETS,
)

diarization_failures_total = Counter(
    "diarization_failures_total",
    "Количество падений диаризации (звонок обрабатывается без ролей)",
)

diarization_role_assignment_total = Counter(
    "diarization_role_assignment_total",
    "Способ определения роли оператора: по содержанию реплик или порядковый fallback",
    ["method"],
)

diarization_role_margin = Histogram(
    "diarization_role_margin",
    "Отрыв кластера-оператора от следующего по смысловому score",
    buckets=(0, 0.25, 0.5, 1, 2, 4, 8),
)

diarization_unknown_segments_total = Counter(
    "diarization_unknown_segments_total",
    "Сегменты, которым не удалось сопоставить спикера",
)

agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Время работы LLM-агента",
    ["agent"],
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)

agent_runs_total = Counter(
    "agent_runs_total",
    "Запуски LLM-агентов по исходу",
    ["agent", "outcome"],
)


def record_analysis(result: dict) -> None:
    """Разложить финальный state графа агентов по доменным метрикам."""
    if not settings.METRICS_ENABLED:
        return

    try:
        classification = result.get("classification") or {}
        topic = classification.get("topic")
        if topic:
            calls_by_topic_total.labels(topic=str(topic)).inc()

        priority = classification.get("priority")
        if priority:
            calls_by_priority_total.labels(priority=str(priority)).inc()

        quality = result.get("quality_score") or {}
        total = quality.get("total")
        if isinstance(total, int | float):
            call_quality_score.observe(float(total))

        checklist = quality.get("checklist") or {}
        for item in CHECKLIST_ITEMS:
            if checklist.get(item):
                call_quality_checklist_passed_total.labels(item=item).inc()

        compliance = result.get("compliance") or {}
        issues = compliance.get("issues") or []
        compliance_checks_total.labels(result="passed" if compliance.get("passed") else "failed").inc()
        for issue in issues:
            severity = issue.get("severity") if isinstance(issue, dict) else None
            compliance_issues_total.labels(severity=str(severity or "unknown")).inc()
    except Exception:
        log.warning("metrics_record_failed", metric="analysis", exc_info=True)


def observe_call(status: str, duration_s: float) -> None:
    if not settings.METRICS_ENABLED:
        return

    try:
        calls_analyzed_total.labels(status=status).inc()
        call_analysis_duration_seconds.observe(duration_s)
    except Exception:
        log.warning("metrics_record_failed", metric="call", exc_info=True)


def observe_asr(duration_s: float, audio_duration_s: float) -> None:
    if not settings.METRICS_ENABLED:
        return

    try:
        asr_transcription_duration_seconds.observe(duration_s)
        if audio_duration_s > 0:
            asr_audio_duration_seconds.observe(audio_duration_s)
            asr_realtime_factor.observe(duration_s / audio_duration_s)
    except Exception:
        log.warning("metrics_record_failed", metric="asr", exc_info=True)


def observe_diarization(duration_s: float) -> None:
    if not settings.METRICS_ENABLED:
        return

    try:
        diarization_duration_seconds.observe(duration_s)
    except Exception:
        log.warning("metrics_record_failed", metric="diarization", exc_info=True)


def observe_role_assignment(method: str, margin: float, unknown_segments: int) -> None:
    """Зафиксировать, как определилась роль оператора.

    Доля `method="fallback_order"` — прямой сигнал, что смысловая эвристика
    не срабатывает на реальном трафике и роли снова назначаются по порядку.
    """
    if not settings.METRICS_ENABLED:
        return

    try:
        diarization_role_assignment_total.labels(method=method).inc()
        diarization_role_margin.observe(margin)
        if unknown_segments > 0:
            diarization_unknown_segments_total.inc(unknown_segments)
    except Exception:
        log.warning("metrics_record_failed", metric="role_assignment", exc_info=True)


def record_diarization_failure() -> None:
    if not settings.METRICS_ENABLED:
        return

    try:
        diarization_failures_total.inc()
    except Exception:
        log.warning("metrics_record_failed", metric="diarization_failure", exc_info=True)


def observe_agent(agent: str, outcome: str, duration_s: float) -> None:
    if not settings.METRICS_ENABLED:
        return

    try:
        agent_runs_total.labels(agent=agent, outcome=outcome).inc()
        if outcome != "skipped":
            agent_duration_seconds.labels(agent=agent).observe(duration_s)
    except Exception:
        log.warning("metrics_record_failed", metric="agent", agent=agent, exc_info=True)

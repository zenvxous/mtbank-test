"""Структурные тесты пайплайна: проводка графа и HTTP-слой.

Быстрые и офлайновые. Прогон всего пайплайна на настоящей LLM живёт
в tests/test_pipeline_integration.py.
"""

import io
import wave

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langgraph.graph import END, START

from agents.classifier import Classification, Priority, Topic
from agents.compliance import ComplianceResult
from agents.graph import build_graph, run_analysis
from agents.quality import QualityChecklist, QualityScore
from agents.summarizer import Summary
from asr.transcriber import Segment, Transcript


@pytest.fixture
def mocked_agents(patch_llm):
    """Все четыре агента отвечают детерминированно; отдаёт записи их вызовов."""
    return {
        "classifier": patch_llm(
            "agents.classifier", Classification(topic=Topic.CARDS, priority=Priority.HIGH)
        ),
        "quality": patch_llm(
            "agents.quality",
            QualityScore(
                total=82,
                checklist=QualityChecklist(
                    greeting=True, need_detection=True, solution_provided=True, farewell=True
                ),
                comment="Оператор отработал по чеклисту.",
            ),
        ),
        "compliance": patch_llm("agents.compliance", ComplianceResult(passed=True, issues=[])),
        "summarizer": patch_llm(
            "agents.summarizer",
            Summary(summary="Клиент заблокировал карту.", action_items=["Перевыпустить карту"]),
        ),
    }


class TestGraphWiring:
    def test_topology_matches_supervisor_design(self):
        graph = build_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        assert (START, "classifier") in edges
        assert ("classifier", "quality") in edges
        assert ("classifier", "compliance") in edges
        assert ("quality", "summarizer") in edges
        assert ("compliance", "summarizer") in edges
        assert ("summarizer", END) in edges

    def test_state_flows_between_nodes(self, fake_settings, mocked_agents, transcript):
        """Классификация доезжает до quality/compliance, а их результаты — до суммаризатора."""
        run_analysis(transcript)

        classification = {"topic": Topic.CARDS, "priority": Priority.HIGH}
        assert mocked_agents["quality"].inputs["classification"] == classification
        assert mocked_agents["compliance"].inputs["classification"] == classification

        summarizer_inputs = mocked_agents["summarizer"].inputs
        assert summarizer_inputs["classification"] == classification
        assert summarizer_inputs["quality_score"]["total"] == 82
        assert summarizer_inputs["compliance"] == {"passed": True, "issues": []}

    def test_empty_transcript_runs_whole_graph_without_llm(self, forbid_llm):
        state = run_analysis([])

        assert state["classification"] == {"topic": Topic.OTHER, "priority": Priority.LOW}
        assert state["quality_score"]["total"] == 0
        assert state["compliance"] == {"passed": True, "issues": []}
        assert state["action_items"] == []
        assert state["summary"]

    def test_agent_failure_propagates(self, fake_settings, patch_llm, transcript):
        patch_llm("agents.classifier", error=RuntimeError("upstream is down"))

        with pytest.raises(HTTPException) as exc_info:
            run_analysis(transcript)

        assert exc_info.value.status_code == 422

    def test_returns_full_contract(self, fake_settings, mocked_agents, transcript):
        state = run_analysis(transcript)

        assert set(state) >= {
            "transcript",
            "classification",
            "quality_score",
            "compliance",
            "summary",
            "action_items",
        }
        assert state["transcript"] == transcript


def _wav_bytes(seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buffer.getvalue()


class StubTranscriber:
    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.calls: list[str] = []

    def transcribe(self, audio_path: str) -> Transcript:
        self.calls.append(audio_path)
        return Transcript(segments=list(self.segments), language="ru", duration=24.0)


class StubDiarizer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def assign_speakers(self, segments: list[Segment], audio_file_path: str) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        for i, segment in enumerate(segments):
            segment.speaker = "Оператор" if i % 2 == 0 else "Клиент"


@pytest.fixture
def api_client(monkeypatch, analyze_result):
    """TestClient без реального lifespan: Whisper и pyannote не поднимаем."""
    from api import main as api_main

    transcriber = StubTranscriber(
        [
            Segment(start=0.0, end=4.2, text="Добрый день, МТБанк, меня зовут Анна."),
            Segment(start=4.5, end=8.1, text="Здравствуйте, хочу узнать про кредит."),
        ]
    )
    diarizer = StubDiarizer()

    api_main.app.state.transcriber = transcriber
    api_main.app.state.diarizer = diarizer

    analysis_calls: list[list[dict]] = []

    def _run_analysis(dialog: list[dict]) -> dict:
        analysis_calls.append(dialog)
        return {**analyze_result, "transcript": dialog}

    monkeypatch.setattr(api_main, "run_analysis", _run_analysis)

    # TestClient без контекстного менеджера не запускает lifespan — Whisper и pyannote
    # не грузятся, работают подставленные выше стабы.
    client = TestClient(api_main.app)
    client.transcriber = transcriber
    client.diarizer = diarizer
    client.analysis_calls = analysis_calls

    yield client

    api_main.app.state.transcriber = None
    api_main.app.state.diarizer = None


class TestAnalyzeAPI:
    def test_health_reports_loaded_components(self, api_client):
        response = api_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "asr_loaded": True, "diarization_loaded": True}

    def test_analyze_returns_full_contract(self, api_client):
        response = api_client.post(
            "/analyze", files={"file": ("call.wav", _wav_bytes(), "audio/wav")}
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "transcript",
            "classification",
            "quality_score",
            "compliance",
            "summary",
            "action_items",
        }

        # В граф уходит транскрипт в формате Segment.as_dict() с проставленными спикерами.
        dialog = api_client.analysis_calls[0]
        assert [segment["speaker"] for segment in dialog] == ["Оператор", "Клиент"]
        assert set(dialog[0]) == {"speaker", "start", "end", "text"}

    def test_diarization_failure_does_not_break_request(self, api_client):
        api_client.diarizer.error = RuntimeError("pyannote is unavailable")

        response = api_client.post(
            "/analyze", files={"file": ("call.wav", _wav_bytes(), "audio/wav")}
        )

        assert response.status_code == 200
        assert api_client.diarizer.calls == 1

    def test_rejects_unsupported_extension(self, api_client):
        response = api_client.post(
            "/analyze", files={"file": ("notes.txt", b"hello", "text/plain")}
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_rejects_empty_file(self, api_client):
        response = api_client.post("/analyze", files={"file": ("call.wav", b"", "audio/wav")})

        assert response.status_code == 400
        assert response.json()["detail"] == "Uploaded file is empty"

    def test_rejects_oversized_file(self, api_client, monkeypatch):
        from api.config import settings

        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 128)

        response = api_client.post(
            "/analyze", files={"file": ("call.wav", _wav_bytes(seconds=1.0), "audio/wav")}
        )

        assert response.status_code == 413

    def test_rejects_missing_file(self, api_client):
        response = api_client.post("/analyze")

        assert response.status_code == 422

    def test_rejects_file_without_audio_stream(self, api_client):
        response = api_client.post(
            "/analyze", files={"file": ("call.wav", b"not really audio", "audio/wav")}
        )

        assert response.status_code == 422


class TestRequestIdMiddleware:
    def test_echoes_incoming_request_id(self, api_client):
        response = api_client.get("/health", headers={"X-Request-ID": "abc-123"})

        assert response.headers["X-Request-ID"] == "abc-123"

    def test_generates_request_id_when_missing(self, api_client):
        response = api_client.get("/health")

        assert response.headers.get("X-Request-ID")

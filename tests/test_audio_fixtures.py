"""Тесты на реальных файлах из test_data/.

Негативные кейсы гоняются через настоящий HTTP-слой и настоящий PyAV-декодер
(ASR подменён стабом), поэтому проверяют ровно то, что произойдёт в проде при
битом или пустом файле. Позитивные файлы проверяются на соответствие
требованиям README к тестовой выборке.

Аудио генерируется скриптом scripts/make_test_data.py и лежит в репозитории.
"""

import subprocess

import pytest

from tests.conftest import TEST_DATA

NEGATIVE = TEST_DATA / "negative"

# (файл, HTTP-код, фрагмент detail) — ожидания зафиксированы по api/main.py.
NEGATIVE_CASES = [
    ("empty.wav", 400, "Uploaded file is empty"),
    ("not_audio.txt", 400, "Unsupported file type"),
    ("corrupted.ogg", 422, "Failed to decode audio"),
    ("no_audio_stream.wav", 422, "File contains no audio stream"),
]

POSITIVE_FILES = [
    "call_dialog.wav",
    "call_dialog_8k.wav",
    "card_block.mp3",
    "complaint_violation.wav",
    "deposit_question.ogg",
    "short_greeting.mp3",
]

CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".txt": "text/plain"}


def probe(path) -> dict:
    """sample_rate/channels/duration через ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=sample_rate,channels",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.strip().splitlines())
    return {
        "sample_rate": int(values["sample_rate"]),
        "channels": int(values["channels"]),
        "duration": float(values["duration"]),
    }


def upload(client, path):
    content_type = CONTENT_TYPES[path.suffix]
    return client.post("/analyze", files={"file": (path.name, path.read_bytes(), content_type)})


class TestNegativeFixtures:
    @pytest.mark.parametrize(("filename", "status", "detail"), NEGATIVE_CASES)
    def test_bad_audio_is_rejected(self, api_client, filename, status, detail):
        response = upload(api_client, NEGATIVE / filename)

        assert response.status_code == status
        assert detail in response.json()["detail"]

    @pytest.mark.parametrize(("filename", "status", "detail"), NEGATIVE_CASES)
    def test_bad_audio_never_reaches_asr(self, api_client, filename, status, detail):
        """Отбраковка происходит до Whisper — иначе платим за декодирование мусора."""
        upload(api_client, NEGATIVE / filename)

        assert api_client.transcriber.calls == []

    def test_silence_returns_full_contract_without_llm(self, api_client, monkeypatch, forbid_llm):
        """Тишина — валидное аудио: 200, пустой транскрипт, граф не зовёт LLM."""
        from agents.graph import run_analysis
        from api import main as api_main

        # Whisper на тишине возвращает ноль сегментов — воспроизводим это в стабе
        # и пускаем настоящий граф, чтобы проверить его short-circuit.
        api_client.transcriber.segments = []
        monkeypatch.setattr(api_main, "run_analysis", run_analysis)

        response = upload(api_client, NEGATIVE / "silence_30s.wav")

        assert response.status_code == 200
        body = response.json()
        assert body["transcript"] == []
        assert set(body) == {
            "transcript",
            "classification",
            "quality_score",
            "compliance",
            "summary",
            "action_items",
        }

    def test_oversized_real_file_is_rejected(self, api_client, monkeypatch):
        """MAX_UPLOAD_BYTES проверяется на настоящем файле, а не на синтетических байтах."""
        from api.config import settings

        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)

        response = upload(api_client, TEST_DATA / "call_dialog.wav")

        assert response.status_code == 413
        assert api_client.transcriber.calls == []


class TestPositiveFixtures:
    """Требования README к тестовой выборке проверяются автоматически, а не на глаз."""

    @pytest.mark.parametrize("filename", POSITIVE_FILES)
    def test_has_reference_transcript(self, filename):
        reference = (TEST_DATA / filename).with_suffix(".txt")

        assert reference.exists(), f"нет эталона для {filename}"
        text = reference.read_text(encoding="utf-8").strip()
        assert text
        assert all(line.split(":", 1)[0] in {"Оператор", "Клиент"} for line in text.splitlines())

    def test_total_duration_is_at_least_five_minutes(self):
        total = sum(probe(TEST_DATA / name)["duration"] for name in POSITIVE_FILES)

        assert total >= 300, f"суммарно {total:.0f} с, README требует не меньше 300"

    def test_has_telephone_quality_file(self):
        assert probe(TEST_DATA / "call_dialog_8k.wav")["sample_rate"] == 8000

    def test_has_two_speaker_dialog_over_a_minute(self):
        reference = (TEST_DATA / "call_dialog.wav").with_suffix(".txt")
        speakers = {line.split(":", 1)[0] for line in reference.read_text(encoding="utf-8").splitlines()}

        assert speakers == {"Оператор", "Клиент"}
        assert probe(TEST_DATA / "call_dialog.wav")["duration"] >= 60

    def test_covers_all_supported_formats(self):
        from api.main import ALLOWED_EXTENSIONS

        extensions = {(TEST_DATA / name).suffix for name in POSITIVE_FILES}

        assert extensions == set(ALLOWED_EXTENSIONS)

    def test_sample_rates_and_channels_vary(self):
        specs = {(probe(TEST_DATA / name)["sample_rate"], probe(TEST_DATA / name)["channels"]) for name in POSITIVE_FILES}

        assert len({rate for rate, _ in specs}) >= 3, "нужны разные sample rate"
        assert 2 in {channels for _, channels in specs}, "нужен хотя бы один стереофайл"

    @pytest.mark.parametrize("filename", POSITIVE_FILES)
    def test_accepted_by_api(self, api_client, filename):
        response = upload(api_client, TEST_DATA / filename)

        assert response.status_code == 200

"""Тесты Open WebUI pipeline: чистые хелперы форматирования и ветки _run.

Сеть мокается подменой requests.get/requests.post внутри загруженного модуля
(pipeline ходит через requests, не httpx).
"""

import json

import pytest
import requests


class FakeResponse:
    def __init__(
        self,
        content: bytes = b"",
        headers: dict | None = None,
        json_data: dict | None = None,
        raise_for_status: Exception | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self.content = content
        self.headers = headers or {}
        self._json = json_data
        self._raise = raise_for_status
        self._json_error = json_error
        self.closed = False

    def raise_for_status(self) -> None:
        if self._raise is not None:
            raise self._raise

    def json(self) -> dict:
        if self._json_error is not None:
            raise self._json_error
        return self._json or {}

    def iter_content(self, chunk_size: int = 1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.closed = True


class TestFormatHelpers:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [(0, "0 Б"), (512, "512 Б"), (2048, "2 КБ"), (5 * 1024 * 1024, "5.0 МБ")],
    )
    def test_format_size(self, owui_pipeline, size, expected):
        assert owui_pipeline._format_size(size) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00"), (5.4, "00:05"), (65.5, "01:05"), (3601, "60:01"), (None, "--:--"), ("x", "--:--")],
    )
    def test_format_timestamp(self, owui_pipeline, seconds, expected):
        assert owui_pipeline._format_timestamp(seconds) == expected

    @pytest.mark.parametrize(
        ("total", "mark"), [(100, "🟢"), (80, "🟢"), (79, "🟡"), (50, "🟡"), (49, "🔴"), (0, "🔴")]
    )
    def test_format_score_marks(self, owui_pipeline, total, mark):
        rendered = owui_pipeline._format_score(total)

        assert rendered.startswith(mark)
        assert f"{total}/100" in rendered

    def test_format_score_without_number(self, owui_pipeline):
        assert owui_pipeline._format_score(None) == "—"
        assert owui_pipeline._format_score("78") == "—"

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, "1 реплика"),
            (2, "2 реплики"),
            (5, "5 реплик"),
            (11, "11 реплик"),
            (14, "14 реплик"),
            (21, "21 реплика"),
            (114, "114 реплик"),
        ],
    )
    def test_plural_replicas(self, owui_pipeline, count, expected):
        assert owui_pipeline._plural_replicas(count) == expected


class TestMessageParsing:
    def test_extracts_file_tags(self, owui_pipeline):
        content = '<file type="audio" id="abc-1" name="call.wav"/> проанализируй'

        assert owui_pipeline._extract_file_tags(content) == [("abc-1", "call.wav")]

    def test_extracts_multiple_file_tags(self, owui_pipeline):
        content = '<file id="a" name="one.wav"/><file id="b" name="two.mp3"/>'

        assert owui_pipeline._extract_file_tags(content) == [("a", "one.wav"), ("b", "two.mp3")]

    def test_no_file_tags(self, owui_pipeline):
        assert owui_pipeline._extract_file_tags("просто текст") == []

    def test_extracts_url_and_strips_punctuation(self, owui_pipeline):
        content = "Вот запись: https://cdn.example.com/call.mp3."

        assert owui_pipeline._extract_urls(content) == ["https://cdn.example.com/call.mp3"]

    def test_ignores_urls_inside_file_tags(self, owui_pipeline):
        content = '<file id="1" name="call.wav" url="https://openwebui/files/1"/>'

        assert owui_pipeline._extract_urls(content) == []

    def test_detects_openwebui_task_prompt(self, owui_pipeline):
        messages = [{"content": "### Task:\nGenerate a chat title"}]

        assert owui_pipeline._is_task_request(messages, "") is True
        assert owui_pipeline._is_task_request([{"content": "проанализируй"}], "") is False

    def test_task_request_returns_empty_string(self, owui_pipeline):
        result = owui_pipeline.pipe("", "model", [{"content": "### Task: summarize"}], {})

        assert result == ""

    def test_last_message_content_falls_back_to_user_message(self, owui_pipeline):
        messages = [{"content": [{"type": "text", "text": "мультимодальный контент"}]}]

        assert owui_pipeline._last_message_content(messages, "запасной текст") == "запасной текст"
        assert owui_pipeline._last_message_content([], "запасной текст") == ""


class TestFilenameResolution:
    @pytest.mark.parametrize(
        ("disposition", "expected"),
        [
            ('attachment; filename="запись.wav"', "запись.wav"),
            ("attachment; filename=call.mp3", "call.mp3"),
            ("attachment; filename*=UTF-8''%D0%B7%D0%B2%D0%BE%D0%BD%D0%BE%D0%BA.ogg", "звонок.ogg"),
        ],
    )
    def test_prefers_content_disposition(self, owui_pipeline, disposition, expected):
        headers = {"Content-Disposition": disposition, "Content-Type": "audio/mpeg"}

        assert owui_pipeline._resolve_filename("https://cdn.example.com/x", headers) == expected

    def test_falls_back_to_url_path(self, owui_pipeline):
        headers = {"Content-Disposition": 'attachment; filename="report.pdf"'}

        assert owui_pipeline._resolve_filename("https://cdn.example.com/audio/call.wav", headers) == "call.wav"

    def test_falls_back_to_content_type(self, owui_pipeline):
        headers = {"Content-Type": "audio/mpeg; charset=binary"}

        assert owui_pipeline._resolve_filename("https://cdn.example.com/files/xyz123", headers) == "xyz123.mp3"

    def test_unsupported_format_messages(self, owui_pipeline):
        assert "'.flac'" in owui_pipeline._unsupported_format_error(".flac")
        assert "не удалось определить формат" in owui_pipeline._unsupported_format_error("")


class TestReport:
    def test_renders_all_sections(self, owui_pipeline, analyze_result):
        report = owui_pipeline._format_report(analyze_result, "call.wav")

        assert "`call.wav`" in report
        assert "Кредиты" in report and "🟡 Средний" in report
        assert "🟡 78/100" in report
        assert "✅ Приветствие и представление" in report
        assert "❌ Корректное завершение" in report
        assert "✅ Нарушений регламента не обнаружено." in report
        assert "- [ ] Отправить КП на email клиента" in report
        assert "🎧 Транскрипт · 2 реплики" in report
        assert "`01:05` **Клиент:**" in report

    def test_renders_compliance_issues(self, owui_pipeline, analyze_result):
        analyze_result["compliance"] = {
            "passed": False,
            "issues": [
                {
                    "rule": "Запрос секретных данных клиента",
                    "severity": "high",
                    "quote": "Назовите код из SMS",
                    "comment": "Оператор не вправе запрашивать код из SMS.",
                }
            ],
        }

        report = owui_pipeline._format_report(analyze_result, "call.wav")

        assert "❌ Обнаружено нарушений: **1**" in report
        assert "**1. Запрос секретных данных клиента** — 🔴 high" in report
        assert "«Назовите код из SMS»" in report
        assert "❌ нарушений: 1" in report

    def test_empty_result_does_not_crash(self, owui_pipeline):
        report = owui_pipeline._format_report({}, "call.wav")

        assert "Анализ звонка" in report
        assert "| — | — | — | — |" in report


class TestRunBranches:
    def test_rejects_file_and_url_together(self, owui_pipeline):
        content = '<file id="1" name="call.wav"/> и ещё https://cdn.example.com/call.mp3'

        output = "".join(owui_pipeline._run([{"content": content}], ""))

        assert "есть и приложенный файл, и ссылка" in output

    def test_rejects_multiple_files(self, owui_pipeline):
        content = '<file id="a" name="one.wav"/><file id="b" name="two.wav"/>'

        output = "".join(owui_pipeline._run([{"content": content}], ""))

        assert "обнаружено 2 файлов" in output

    def test_rejects_multiple_urls(self, owui_pipeline):
        content = "https://a.example.com/one.wav https://b.example.com/two.wav"

        output = "".join(owui_pipeline._run([{"content": content}], ""))

        assert "обнаружено 2 ссылок" in output

    def test_requires_audio_source(self, owui_pipeline):
        output = "".join(owui_pipeline._run([{"content": "проанализируй звонок"}], ""))

        assert "файл не найден в сообщении" in output

    def test_loader_error_is_returned_as_is(self, owui_pipeline):
        content = '<file id="1" name="notes.txt"/>'

        output = "".join(owui_pipeline._run([{"content": content}], ""))

        assert "формат '.txt' не поддерживается" in output

    def test_happy_path_streams_statuses_and_report(self, owui_pipeline, monkeypatch, analyze_result):
        monkeypatch.setattr(
            owui_pipeline, "_load_attached_file", lambda *_: ("call.wav", b"x" * 2048, "audio/wav")
        )
        monkeypatch.setattr(owui_pipeline, "_send_to_analyze", lambda *_: analyze_result)

        chunks = list(owui_pipeline._run([{"content": '<file id="1" name="call.wav"/>'}], ""))
        output = "".join(chunks)

        assert len(chunks) > 1, "отчёт должен отдаваться по частям, а не одним куском"
        assert "Получаю файл `call.wav`" in output
        assert "Файл получен: `call.wav`, 2 КБ." in output
        assert "Анализ звонка" in output

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (requests.RequestException("connection reset"), "Ошибка при отправке файла на анализ"),
            (ValueError("no json"), "некорректный ответ API анализа (не JSON)"),
        ],
    )
    def test_analyze_errors_are_reported(self, owui_pipeline, monkeypatch, error, expected):
        monkeypatch.setattr(
            owui_pipeline, "_load_attached_file", lambda *_: ("call.wav", b"x", "audio/wav")
        )

        def _raise(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(owui_pipeline, "_send_to_analyze", _raise)

        output = "".join(owui_pipeline._run([{"content": '<file id="1" name="call.wav"/>'}], ""))

        assert expected in output

    def test_http_error_includes_response_body(self, owui_pipeline, monkeypatch):
        monkeypatch.setattr(
            owui_pipeline, "_load_attached_file", lambda *_: ("call.wav", b"x", "audio/wav")
        )

        response = requests.Response()
        response.status_code = 422
        response._content = b'{"detail":"File contains no audio stream"}'

        def _raise(*_args, **_kwargs):
            raise requests.HTTPError("422 Client Error", response=response)

        monkeypatch.setattr(owui_pipeline, "_send_to_analyze", _raise)

        output = "".join(owui_pipeline._run([{"content": '<file id="1" name="call.wav"/>'}], ""))

        assert "File contains no audio stream" in output

    def test_broken_report_falls_back_to_raw_json(self, owui_pipeline, monkeypatch, analyze_result):
        monkeypatch.setattr(
            owui_pipeline, "_load_attached_file", lambda *_: ("call.wav", b"x", "audio/wav")
        )
        monkeypatch.setattr(owui_pipeline, "_send_to_analyze", lambda *_: analyze_result)

        def _boom(*_args, **_kwargs):
            raise KeyError("unexpected shape")

        monkeypatch.setattr(owui_pipeline, "_format_report", _boom)

        output = "".join(owui_pipeline._run([{"content": '<file id="1" name="call.wav"/>'}], ""))

        assert "```json" in output
        assert json.loads(output.split("```json")[1].split("```")[0])["summary"]


class TestDownloads:
    def test_loads_attached_file(self, owui_module, owui_pipeline, monkeypatch):
        monkeypatch.setattr(
            owui_module.requests,
            "get",
            lambda *_a, **_kw: FakeResponse(content=b"audio", headers={"Content-Type": "audio/wav"}),
        )

        assert owui_pipeline._load_attached_file("id-1", "call.wav") == ("call.wav", b"audio", "audio/wav")

    def test_reports_empty_attached_file(self, owui_module, owui_pipeline, monkeypatch):
        monkeypatch.setattr(owui_module.requests, "get", lambda *_a, **_kw: FakeResponse(content=b""))

        assert "пустой" in owui_pipeline._load_attached_file("id-1", "call.wav")

    def test_reports_download_error(self, owui_module, owui_pipeline, monkeypatch):
        def _raise(*_a, **_kw):
            raise requests.ConnectionError("host unreachable")

        monkeypatch.setattr(owui_module.requests, "get", _raise)

        assert "Ошибка при скачивании файла" in owui_pipeline._load_attached_file("id-1", "call.wav")

    def test_rejects_unsupported_url_extension_before_download(self, owui_module, owui_pipeline, monkeypatch):
        def _fail(*_a, **_kw):
            raise AssertionError("скачивание не должно начинаться")

        monkeypatch.setattr(owui_module.requests, "get", _fail)

        assert "'.flac'" in owui_pipeline._load_file_by_url("https://cdn.example.com/call.flac")

    def test_loads_file_by_url(self, owui_module, owui_pipeline, monkeypatch):
        monkeypatch.setattr(
            owui_module.requests,
            "get",
            lambda *_a, **_kw: FakeResponse(content=b"audio-bytes", headers={"Content-Type": "audio/mpeg"}),
        )

        assert owui_pipeline._load_file_by_url("https://cdn.example.com/call.mp3") == (
            "call.mp3",
            b"audio-bytes",
            "audio/mpeg",
        )

    def test_read_capped_rejects_oversized_download(self, owui_module, owui_pipeline, monkeypatch):
        monkeypatch.setattr(owui_module, "MAX_DOWNLOAD_BYTES", 8)

        with pytest.raises(ValueError, match="больше"):
            owui_pipeline._read_capped(FakeResponse(content=b"x" * 64))

    def test_send_to_analyze_posts_multipart(self, owui_module, owui_pipeline, monkeypatch, analyze_result):
        captured: dict = {}

        def _post(url, files=None, timeout=None):
            captured.update(url=url, files=files, timeout=timeout)
            return FakeResponse(json_data=analyze_result)

        monkeypatch.setattr(owui_module.requests, "post", _post)

        result = owui_pipeline._send_to_analyze("call.wav", b"audio", "audio/wav")

        assert result == analyze_result
        assert captured["url"].endswith("/analyze")
        assert captured["files"]["file"] == ("call.wav", b"audio", "audio/wav")
        assert captured["timeout"] == owui_module.ANALYZE_TIMEOUT

import json
import os
import re
from collections.abc import Generator
from urllib.parse import unquote, urlparse

import requests
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}
# Аудиофайл можно не прикладывать, а прислать прямую ссылку на него: тогда
# имя файла берётся из Content-Disposition, из пути URL или из Content-Type.
CONTENT_TYPE_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "application/ogg": ".ogg",
}
URL_RE = re.compile(r"https?://[^\s<>\"'\])}]+")
FILE_TAG_RE = re.compile(r"<file\b[^>]*/>")
CONTENT_DISPOSITION_RE = re.compile(
    r"filename\*=UTF-8''(?P<encoded>[^;]+)|filename=\"(?P<quoted>[^\"]+)\"|filename=(?P<bare>[^;]+)",
    re.IGNORECASE,
)
# Ссылка ведёт на произвольный внешний ресурс, поэтому качаем потоком и режем
# по размеру, чтобы не забить память сервера.
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
# Служебные задачи Open WebUI (заголовок чата, теги, поисковые запросы, follow-up)
# уходят в ту же модель обычным /chat/completions и опознаются только по префиксу
# промпта: metadata с типом задачи до pipeline-сервера не доезжает.
TASK_PROMPT_PREFIX = "### Task:"
DOWNLOAD_TIMEOUT = 30
ANALYZE_TIMEOUT = 300

TOPIC_LABELS = {
    "credits": "Кредиты",
    "cards": "Карты",
    "transfers": "Переводы",
    "complaints": "Жалобы",
    "other": "Другое",
}

PRIORITY_LABELS = {
    "low": "🟢 Низкий",
    "medium": "🟡 Средний",
    "high": "🟠 Высокий",
    "critical": "🔴 Критический",
}

SEVERITY_LABELS = {
    "low": "🟢 low",
    "medium": "🟡 medium",
    "high": "🔴 high",
}

CHECKLIST_LABELS = {
    "greeting": "Приветствие и представление",
    "need_detection": "Выявление потребности",
    "solution_provided": "Предложено решение",
    "farewell": "Корректное завершение",
}


class Pipeline:
    class Valves(BaseModel):
        OPENWEBUI_BASE_URL: str = os.getenv("OPENWEBUI_BASE_URL", "http://openwebui:8080")
        OPENWEBUI_API_KEY: str = os.getenv("OPENWEBUI_API_KEY", "")
        ANALYZE_API_URL: str = os.getenv("ANALYZE_API_URL", "http://0.0.0.0:8000")

    def __init__(self):
        self.name = "Audio File Аnalysis Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: list[dict], body: dict
    ) -> str | Generator[str, None, None]:
        if self._is_task_request(messages, user_message):
            return ""
        return self._run(messages, user_message)

    def _run(self, messages: list[dict], user_message: str) -> Generator[str, None, None]:
        """Отдаёт отчёт по частям: анализ идёт минутами, и без промежуточных
        сообщений пользователь всё это время смотрит в пустой ответ."""
        content = self._last_message_content(messages, user_message)
        file_tags = self._extract_file_tags(content)
        urls = self._extract_urls(content)

        if file_tags and urls:
            yield (
                "Ошибка: в сообщении есть и приложенный файл, и ссылка. "
                "Оставьте что-то одно."
            )
            return

        if file_tags:
            if len(file_tags) > 1:
                yield (
                    f"Ошибка: обнаружено {len(file_tags)} файлов, "
                    "а поддерживается только один файл за раз."
                )
                return
            yield self._status(f"Получаю файл `{file_tags[0][1]}`…")
            source = self._load_attached_file(*file_tags[0])
        elif urls:
            if len(urls) > 1:
                yield (
                    f"Ошибка: обнаружено {len(urls)} ссылок, "
                    "а поддерживается только одна ссылка за раз."
                )
                return
            yield self._status(f"Скачиваю файл по ссылке {urls[0]} …")
            source = self._load_file_by_url(urls[0])
        else:
            yield (
                "Ошибка: файл не найден в сообщении. Приложите один аудиофайл "
                "(mp3, wav или ogg) или пришлите прямую ссылку на него."
            )
            return

        if isinstance(source, str):
            yield source
            return

        filename, file_content, content_type = source
        yield self._status(
            f"Файл получен: `{filename}`, {self._format_size(len(file_content))}."
        )
        yield self._status(
            "Отправляю на анализ: транскрибация, разделение по спикерам, "
            "оценка качества и комплаенса. Обычно это занимает пару минут…"
        )

        try:
            result = self._send_to_analyze(filename, file_content, content_type)
        except requests.HTTPError as e:
            detail = e.response.text if e.response is not None else ""
            yield f"Ошибка при отправке файла на анализ: {e}\n{detail}"
            return
        except requests.RequestException as e:
            yield f"Ошибка при отправке файла на анализ: {e}"
            return
        except ValueError:
            yield "Ошибка: некорректный ответ API анализа (не JSON)."
            return

        yield self._status("Анализ завершён, собираю отчёт.")
        yield "\n---\n\n"

        try:
            yield self._format_report(result, filename)
        except Exception:
            yield "```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```"

    def _status(self, text: str) -> str:
        return f"⏳ {text}\n\n"

    def _format_size(self, size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} МБ"
        if size >= 1024:
            return f"{size / 1024:.0f} КБ"
        return f"{size} Б"

    def _format_report(self, result: dict, filename: str) -> str:
        classification = result.get("classification") or {}
        quality = result.get("quality_score") or {}
        compliance = result.get("compliance") or {}
        summary = (result.get("summary") or "").strip()
        action_items = result.get("action_items") or []
        transcript = result.get("transcript") or []

        parts = [
            f"## 📞 Анализ звонка\n`{filename}`",
            self._format_overview(classification, quality, compliance),
        ]

        if summary:
            parts.append(f"### 📝 Кратко\n{summary}")

        parts.append(self._format_quality(quality))
        parts.append(self._format_compliance(compliance))

        if action_items:
            items = "\n".join(f"- [ ] {str(item).strip()}" for item in action_items)
            parts.append(f"### ✅ Что сделать дальше\n{items}")

        if transcript:
            parts.append(self._format_transcript(transcript))

        return "\n\n".join(part for part in parts if part)

    def _format_overview(self, classification: dict, quality: dict, compliance: dict) -> str:
        topic = str(classification.get("topic", "—"))
        priority = str(classification.get("priority", "—"))
        total = quality.get("total")

        if not compliance:
            compliance_cell = "—"
        elif compliance.get("passed"):
            compliance_cell = "✅ пройден"
        else:
            issues_count = len(compliance.get("issues") or [])
            compliance_cell = f"❌ нарушений: {issues_count}" if issues_count else "❌ не пройден"

        return (
            "| Тема | Приоритет | Качество | Комплаенс |\n"
            "| --- | --- | --- | --- |\n"
            f"| {TOPIC_LABELS.get(topic, topic)} "
            f"| {PRIORITY_LABELS.get(priority, priority)} "
            f"| {self._format_score(total)} "
            f"| {compliance_cell} |"
        )

    def _format_score(self, total) -> str:
        if not isinstance(total, int):
            return "—"

        filled = round(max(0, min(total, 100)) / 10)
        bar = "█" * filled + "░" * (10 - filled)
        mark = "🟢" if total >= 80 else "🟡" if total >= 50 else "🔴"
        return f"{mark} {total}/100 `{bar}`"

    def _format_quality(self, quality: dict) -> str:
        if not quality:
            return ""

        lines = ["### 📊 Качество обслуживания", "", self._format_score(quality.get("total")), ""]

        checklist = quality.get("checklist") or {}
        for key, label in CHECKLIST_LABELS.items():
            if key in checklist:
                lines.append(f"- {'✅' if checklist[key] else '❌'} {label}")

        comment = (quality.get("comment") or "").strip()
        if comment:
            lines.append("")
            lines.append(f"> {comment}")

        return "\n".join(lines)

    def _format_compliance(self, compliance: dict) -> str:
        if not compliance:
            return ""

        issues = compliance.get("issues") or []
        if compliance.get("passed") and not issues:
            return "### 🛡️ Комплаенс\n\n✅ Нарушений регламента не обнаружено."

        lines = [f"### 🛡️ Комплаенс\n\n❌ Обнаружено нарушений: **{len(issues)}**", ""]
        for i, issue in enumerate(issues, start=1):
            severity = str(issue.get("severity", "—"))
            rule = str(issue.get("rule", "—")).strip()
            lines.append(f"**{i}. {rule}** — {SEVERITY_LABELS.get(severity, severity)}")

            quote = (issue.get("quote") or "").strip()
            if quote:
                lines.append(f"> «{quote}»")

            comment = (issue.get("comment") or "").strip()
            if comment:
                lines.append(f"\n{comment}")

            lines.append("")

        return "\n".join(lines).rstrip()

    def _format_transcript(self, transcript: list[dict]) -> str:
        lines = [
            "<details>",
            f"<summary>🎧 Транскрипт · {self._plural_replicas(len(transcript))}</summary>",
            "",
        ]

        for segment in transcript:
            speaker = str(segment.get("speaker", "—"))
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"`{self._format_timestamp(segment.get('start'))}` **{speaker}:** {text}")
            lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _plural_replicas(self, count: int) -> str:
        if 11 <= count % 100 <= 14:
            word = "реплик"
        elif count % 10 == 1:
            word = "реплика"
        elif count % 10 in (2, 3, 4):
            word = "реплики"
        else:
            word = "реплик"
        return f"{count} {word}"

    def _format_timestamp(self, seconds) -> str:
        if not isinstance(seconds, (int, float)):
            return "--:--"
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"

    def _is_task_request(self, messages: list[dict], user_message: str) -> bool:
        content = self._last_message_content(messages, user_message)
        return content.lstrip().startswith(TASK_PROMPT_PREFIX)

    def _last_message_content(self, messages: list[dict], user_message: str) -> str:
        content = messages[-1].get("content", "") if messages else ""
        if not isinstance(content, str):
            content = user_message or ""
        return content

    def _extract_file_tags(self, content: str) -> list[tuple[str, str]]:
        return re.findall(
            r'<file[^>]*id="(?P<id>[^"]+)"[^>]*name="(?P<name>[^"]+)"[^>]*/>', content
        )

    def _extract_urls(self, content: str) -> list[str]:
        # Теги <file .../> сами содержат url-подобные атрибуты, поэтому вырезаем их.
        text = FILE_TAG_RE.sub(" ", content)
        return [url.rstrip(".,;:!?") for url in URL_RE.findall(text)]

    def _load_attached_file(self, file_id: str, filename: str) -> tuple[str, bytes, str] | str:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return self._unsupported_format_error(ext)

        try:
            resp = self._fetch_file(file_id)
        except requests.RequestException as e:
            return f"Ошибка при скачивании файла '{filename}': {e}"

        if not resp.content:
            return f"Ошибка: файл '{filename}' пустой."

        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        return filename, resp.content, content_type

    def _load_file_by_url(self, url: str) -> tuple[str, bytes, str] | str:
        # Если расширение видно прямо в ссылке — отсекаем чужой формат до скачивания.
        url_ext = os.path.splitext(unquote(urlparse(url).path))[1].lower()
        if url_ext and url_ext not in SUPPORTED_EXTENSIONS:
            return self._unsupported_format_error(url_ext)

        try:
            resp = self._fetch_url(url)
        except requests.RequestException as e:
            return f"Ошибка при скачивании файла по ссылке '{url}': {e}"

        try:
            with resp:
                content = self._read_capped(resp)
        except ValueError as e:
            return str(e)
        except requests.RequestException as e:
            return f"Ошибка при скачивании файла по ссылке '{url}': {e}"

        if not content:
            return f"Ошибка: по ссылке '{url}' скачан пустой файл."

        content_type = resp.headers.get("Content-Type", "application/octet-stream")
        filename = self._resolve_filename(url, resp.headers)

        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return self._unsupported_format_error(ext)

        return filename, content, content_type

    def _unsupported_format_error(self, ext: str) -> str:
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        if not ext:
            return (
                "Ошибка: не удалось определить формат файла. "
                f"Поддерживаются только: {formats}."
            )
        return f"Ошибка: формат '{ext}' не поддерживается. Поддерживаются только: {formats}."

    def _resolve_filename(self, url: str, headers) -> str:
        disposition = headers.get("Content-Disposition", "")
        match = CONTENT_DISPOSITION_RE.search(disposition)
        if match:
            raw = match.group("encoded") or match.group("quoted") or match.group("bare") or ""
            name = os.path.basename(unquote(raw.strip().strip('"')))
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                return name

        name = os.path.basename(unquote(urlparse(url).path))
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            return name

        # Ссылка без расширения (пресайн, редирект на CDN) — опираемся на Content-Type.
        content_type = headers.get("Content-Type", "").split(";")[0].strip().lower()
        ext = CONTENT_TYPE_EXTENSIONS.get(content_type)
        if ext:
            return f"{os.path.splitext(name)[0] or 'audio'}{ext}"

        return name or "audio"

    def _read_capped(self, resp: requests.Response) -> bytes:
        chunks = []
        size = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    "Ошибка: файл по ссылке больше "
                    f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} МБ."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _fetch_url(self, url: str) -> requests.Response:
        resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp

    def _fetch_file(self, file_id: str) -> requests.Response:
        content_url = f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/files/{file_id}/content"
        resp = requests.get(
            content_url,
            headers={"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"},
            timeout=DOWNLOAD_TIMEOUT,
        )
        resp.raise_for_status()
        return resp

    def _send_to_analyze(self, filename: str, content: bytes, content_type: str) -> dict:
        analyze_url = f"{self.valves.ANALYZE_API_URL}/analyze"
        resp = requests.post(
            analyze_url,
            files={"file": (filename, content, content_type)},
            timeout=ANALYZE_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

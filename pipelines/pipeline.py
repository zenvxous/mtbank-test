import json
import os
import re

import requests
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}
DOWNLOAD_TIMEOUT = 30
ANALYZE_TIMEOUT = 300


class Pipeline:
    class Valves(BaseModel):
        OPENWEBUI_BASE_URL: str = os.getenv("OPENWEBUI_BASE_URL", "http://openwebui:8080")
        OPENWEBUI_API_KEY: str = os.getenv("OPENWEBUI_API_KEY", "")
        ANALYZE_API_URL: str = os.getenv("ANALYZE_API_URL", "http://0.0.0.0:8000")

    def __init__(self):
        self.name = "Audio File Check Pipeline"
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
    ) -> str:
        file_tags = self._extract_file_tags(messages)

        if not file_tags:
            return "Ошибка: файл не найден в сообщении. Приложите один аудиофайл (mp3, wav или ogg)."

        if len(file_tags) > 1:
            return f"Ошибка: обнаружено {len(file_tags)} файлов, а поддерживается только один файл за раз."

        file_id, filename = file_tags[0]

        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return (
                f"Ошибка: формат '{ext}' не поддерживается. "
                f"Поддерживаются только: {', '.join(SUPPORTED_EXTENSIONS)}."
            )

        try:
            resp = self._fetch_file(file_id)
        except requests.RequestException as e:
            return f"Ошибка при скачивании файла '{filename}': {e}"

        file_content = resp.content
        if not file_content:
            return f"Ошибка: файл '{filename}' пустой."

        content_type = resp.headers.get("Content-Type", "application/octet-stream")

        try:
            result = self._send_to_analyze(filename, file_content, content_type)
        except requests.HTTPError as e:
            detail = e.response.text if e.response is not None else ""
            return f"Ошибка при отправке файла на анализ: {e}\n{detail}"
        except requests.RequestException as e:
            return f"Ошибка при отправке файла на анализ: {e}"
        except ValueError:
            return "Ошибка: некорректный ответ API анализа (не JSON)."

        return "```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```"

    def _extract_file_tags(self, messages: list[dict]) -> list[tuple[str, str]]:
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "")
        return re.findall(
            r'<file[^>]*id="(?P<id>[^"]+)"[^>]*name="(?P<name>[^"]+)"[^>]*/>', content
        )

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

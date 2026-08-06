import os
import re

import requests
from pydantic import BaseModel

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}


class Pipeline:
    class Valves(BaseModel):
        OPENWEBUI_BASE_URL: str = os.getenv("OPENWEBUI_BASE_URL", "http://openwebui:8080")
        OPENWEBUI_API_KEY: str = os.getenv("OPENWEBUI_API_KEY", "")

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
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "")

        file_tags = re.findall(
            r'<file[^>]*id="(?P<id>[^"]+)"[^>]*name="(?P<name>[^"]+)"[^>]*/>', content
        )

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

        content_url = f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/files/{file_id}/content"
        try:
            resp = requests.get(
                content_url,
                headers={"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"},
                timeout=30,
            )
            resp.raise_for_status()
            size_bytes = len(resp.content)
            content_type = resp.headers.get("Content-Type", "unknown")
        except requests.RequestException as e:
            return f"Ошибка при скачивании файла '{filename}': {e}"

        return (
            f"Файл получен и прошёл проверку ✅\n"
            f"Имя файла: {filename}\n"
            f"ID файла: {file_id}\n"
            f"Расширение: {ext}\n"
            f"Content-Type: {content_type}\n"
            f"Размер: {size_bytes} байт"
        )

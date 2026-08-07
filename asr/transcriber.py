import os
import threading
from dataclasses import dataclass, field

from fastapi import HTTPException
from faster_whisper import WhisperModel

from api.config import settings

UNKNOWN_SPEAKER = "Неизвестно"

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = UNKNOWN_SPEAKER

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str = settings.ASR_LANGUAGE
    duration: float = 0.0


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = settings.WHISPER_MODEL_SIZE,
        device: str = settings.WHISPER_DEVICE,
        compute_type: str = settings.WHISPER_COMPUTE_TYPE,
        download_root: str | None = settings.WHISPER_DOWNLOAD_ROOT,
    ):
        self.whisper = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        self._lock = threading.Lock()

    def transcribe(self, audio_path: str) -> Transcript:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File {audio_path} not found.")

        with self._lock:
            try:
                segments, info = self.whisper.transcribe(
                    audio_path,
                    beam_size=settings.ASR_BEAM_SIZE,
                    language=settings.ASR_LANGUAGE,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                segments = list(segments)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to transcribe audio: {e}") from e

        return Transcript(
            segments=[
                Segment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                )
                for segment in segments
            ],
            language=info.language,
            duration=round(info.duration, 2),
        )

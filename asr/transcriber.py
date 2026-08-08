import os
import threading
import time
from dataclasses import dataclass, field

import structlog
from fastapi import HTTPException
from faster_whisper import WhisperModel

from api.config import settings

log = structlog.get_logger(__name__)

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
        log.debug(
            "whisper_model_loading",
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        load_started = time.perf_counter()

        try:
            self.whisper = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
        except Exception:
            log.error("whisper_model_load_failed", model_size=model_size, device=device, exc_info=True)
            raise

        log.info(
            "whisper_model_loaded",
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            duration_ms=round((time.perf_counter() - load_started) * 1000, 2),
        )

        self._lock = threading.Lock()

    def transcribe(self, audio_path: str) -> Transcript:
        if not os.path.exists(audio_path):
            log.error("transcription_failed", reason="file_not_found", audio_path=audio_path)
            raise FileNotFoundError(f"File {audio_path} not found.")

        log.debug(
            "transcription_started",
            audio_path=audio_path,
            language=settings.ASR_LANGUAGE,
            beam_size=settings.ASR_BEAM_SIZE,
        )
        started = time.perf_counter()

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
                log.error(
                    "transcription_failed",
                    reason="whisper_error",
                    audio_path=audio_path,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    exc_info=True,
                )
                raise HTTPException(status_code=422, detail=f"Failed to transcribe audio: {e}") from e

        log.info(
            "transcription_finished",
            segments=len(segments),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            audio_duration_s=round(info.duration, 2),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

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

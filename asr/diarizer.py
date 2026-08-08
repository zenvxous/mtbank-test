import threading
import time
from collections.abc import Iterator
from typing import cast

import structlog
import torch
from fastapi import HTTPException
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core import Segment

from api.config import settings
from asr.transcriber import UNKNOWN_SPEAKER

log = structlog.get_logger(__name__)


class AudioDiarizer:
    def __init__(self):
        hf_token = settings.HF_TOKEN
        if not hf_token:
            log.warning("diarization_pipeline_disabled", reason="hf_token_missing")
            self.diarization_pipeline = None
        else:
            log.debug("diarization_pipeline_loading", model=settings.DIARIZATION_MODEL)
            load_started = time.perf_counter()

            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL, token=hf_token
                )
            except Exception:
                log.error(
                    "diarization_pipeline_load_failed",
                    model=settings.DIARIZATION_MODEL,
                    exc_info=True,
                )
                raise

            if self.diarization_pipeline is None:
                log.error(
                    "diarization_pipeline_load_failed",
                    model=settings.DIARIZATION_MODEL,
                    reason="pipeline_is_none",
                )
                raise HTTPException(status_code=500, detail="Not able to load diarization pipeline.")
            self.diarization_pipeline.to(torch.device("cpu"))

            log.info(
                "diarization_pipeline_loaded",
                model=settings.DIARIZATION_MODEL,
                device="cpu",
                duration_ms=round((time.perf_counter() - load_started) * 1000, 2),
            )

        self._lock = threading.Lock()

    def assign_speakers(self, segments: list, audio_file_path: str):
        if self.diarization_pipeline is None:
            log.error("diarization_failed", reason="pipeline_not_initialized")
            raise HTTPException(status_code=500, detail="Diarization pipeline is not initialized.")

        log.debug("diarization_started", segments=len(segments), audio_path=audio_file_path)
        started = time.perf_counter()

        with self._lock:
            try:
                diarization = cast(DiarizeOutput, self.diarization_pipeline(audio_file_path))
            except Exception:
                log.error(
                    "diarization_failed",
                    reason="pipeline_error",
                    audio_path=audio_file_path,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    exc_info=True,
                )
                raise

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            speaker = UNKNOWN_SPEAKER

            if diarization:
                max_intersection = 0.0
                tracks = cast(
                    Iterator[tuple[Segment, object, str]],
                    diarization.speaker_diarization.itertracks(yield_label=True),
                )
                for turn, _, spk in tracks:
                    intersection = min(end_time, turn.end) - max(start_time, turn.start)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        speaker = spk

            if speaker == "SPEAKER_00":
                speaker = "Оператор"
            elif speaker == "SPEAKER_01":
                speaker = "Клиент"

            segment.speaker = speaker

        speaker_counts: dict[str, int] = {}
        for segment in segments:
            speaker_counts[segment.speaker] = speaker_counts.get(segment.speaker, 0) + 1

        log.info(
            "diarization_finished",
            segments=len(segments),
            speakers=speaker_counts,
            unknown_segments=speaker_counts.get(UNKNOWN_SPEAKER, 0),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

        return segments

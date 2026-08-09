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
from api.metrics import observe_role_assignment
from asr.roles import assign_roles
from asr.transcriber import UNKNOWN_SPEAKER

log = structlog.get_logger(__name__)

# Сегмент Whisper иногда не пересекается ни с одним turn диаризации (пауза,
# обрезка по VAD). Подтягиваем его к ближайшему turn, если тот рядом, — иначе
# сегмент уходит в промпты как «Неизвестно» и выпадает из оценки оператора.
NEAREST_TURN_MAX_GAP_S = 3.0


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

        # 1. Сопоставляем сегменты с анонимными кластерами pyannote.
        cluster_labels: list[str | None] = []
        recovered = 0

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            label: str | None = None

            if diarization:
                max_intersection = 0.0
                nearest_gap = float("inf")
                nearest_label: str | None = None
                tracks = cast(
                    Iterator[tuple[Segment, object, str]],
                    diarization.speaker_diarization.itertracks(yield_label=True),
                )
                for turn, _, spk in tracks:
                    intersection = min(end_time, turn.end) - max(start_time, turn.start)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        label = spk
                    if intersection <= 0:
                        gap = max(turn.start - end_time, start_time - turn.end)
                        if gap < nearest_gap:
                            nearest_gap = gap
                            nearest_label = spk

                if label is None and nearest_label is not None and nearest_gap <= NEAREST_TURN_MAX_GAP_S:
                    label = nearest_label
                    recovered += 1

            cluster_labels.append(label)

        # 2. Решаем по содержанию реплик, какой кластер — оператор.
        cluster_texts: dict[str, list[str]] = {}
        for segment, label in zip(segments, cluster_labels, strict=True):
            if label is not None:
                cluster_texts.setdefault(label, []).append(segment.text)

        assignment = assign_roles(cluster_texts)

        # 3. Раскладываем роли по сегментам. Сырые метки SPEAKER_NN наружу
        # не выходят: у сегмента может быть только роль либо UNKNOWN_SPEAKER.
        for segment, label in zip(segments, cluster_labels, strict=True):
            segment.speaker = assignment.mapping.get(label, UNKNOWN_SPEAKER) if label else UNKNOWN_SPEAKER

        speaker_counts: dict[str, int] = {}
        for segment in segments:
            speaker_counts[segment.speaker] = speaker_counts.get(segment.speaker, 0) + 1

        unknown_segments = speaker_counts.get(UNKNOWN_SPEAKER, 0)
        observe_role_assignment(assignment.method, assignment.margin, unknown_segments)

        log.info(
            "diarization_finished",
            segments=len(segments),
            speakers=speaker_counts,
            unknown_segments=unknown_segments,
            recovered_segments=recovered,
            method=assignment.method,
            margin=assignment.margin,
            cluster_scores=assignment.scores,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

        return segments

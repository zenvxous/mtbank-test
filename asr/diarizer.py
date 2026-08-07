import threading
from collections.abc import Iterator
from typing import cast

import torch
from fastapi import HTTPException
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from pyannote.core import Segment

from api.config import settings


class AudioDiarizer:
    def __init__(self):
        hf_token = settings.HF_TOKEN
        if not hf_token:
            print("HF_TOKEN is not set in the environment variables.")
            self.diarization_pipeline = None
        else:
            self.diarization_pipeline = Pipeline.from_pretrained(
                settings.DIARIZATION_MODEL, token=hf_token
            )
            if self.diarization_pipeline is None:
                raise HTTPException(status_code=500, detail="Not able to load diarization pipeline.")
            self.diarization_pipeline.to(torch.device("cpu"))

        self._lock = threading.Lock()

    def assign_speakers(self, segments: list, audio_file_path: str):
        if self.diarization_pipeline is None:
            raise HTTPException(status_code=500, detail="Diarization pipeline is not initialized.")

        with self._lock:
            diarization = cast(DiarizeOutput, self.diarization_pipeline(audio_file_path))

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            speaker = "Неизвестно"

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

        return segments

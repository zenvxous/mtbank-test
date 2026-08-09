"""Тесты AudioDiarizer.assign_speakers.

Настоящий pyannote не поднимаем: экземпляр создаётся через `object.__new__`,
а вместо пайплайна подставляется фейк, отдающий заранее заданные turn'ы.
Так проверяется вся склейка «сегменты Whisper → кластеры → роли».
"""

from dataclasses import dataclass

import pytest

from asr.diarizer import NEAREST_TURN_MAX_GAP_S, AudioDiarizer
from asr.roles import CLIENT_LABEL, OPERATOR_LABEL
from asr.transcriber import UNKNOWN_SPEAKER, Segment

OPERATOR_LINES = [
    "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?",
    "Конечно, подскажите, пожалуйста, какая сумма вас интересует?",
    "Спасибо за обращение в МТБанк, хорошего дня!",
]

CLIENT_LINES = [
    "Здравствуйте, хочу узнать про условия по кредиту наличными.",
    "У меня есть карточка ваша, мне нужно уточнить платёж.",
    "Нет, всё понятно, спасибо.",
]


@dataclass
class FakeTurn:
    start: float
    end: float


class FakeDiarization:
    def __init__(self, tracks: list[tuple[FakeTurn, str]]) -> None:
        self.speaker_diarization = self
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for turn, label in self._tracks:
            yield turn, None, label


class FakePipeline:
    def __init__(self, tracks: list[tuple[FakeTurn, str]]) -> None:
        self.diarization = FakeDiarization(tracks)
        self.calls: list[str] = []

    def __call__(self, audio_file_path: str) -> FakeDiarization:
        self.calls.append(audio_file_path)
        return self.diarization


def make_diarizer(tracks: list[tuple[FakeTurn, str]]) -> AudioDiarizer:
    """AudioDiarizer без загрузки модели: нужен только assign_speakers."""
    import threading

    diarizer = object.__new__(AudioDiarizer)
    diarizer.diarization_pipeline = FakePipeline(tracks)
    diarizer._lock = threading.Lock()
    return diarizer


def build_call(first_speaker_lines: list[str], second_speaker_lines: list[str]):
    """Собрать чередующийся звонок: сегменты, turn'ы и метки кластеров."""
    segments: list[Segment] = []
    tracks: list[tuple[FakeTurn, str]] = []
    cursor = 0.0

    for index in range(len(first_speaker_lines) + len(second_speaker_lines)):
        lines = first_speaker_lines if index % 2 == 0 else second_speaker_lines
        label = "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01"
        text = lines[index // 2]
        start, end = cursor, cursor + 4.0
        segments.append(Segment(start=start, end=end, text=text))
        tracks.append((FakeTurn(start=start, end=end), label))
        cursor = end + 0.5

    return segments, tracks


def test_client_speaking_first_is_not_labelled_operator():
    """Регрессия на исходный баг: SPEAKER_00 больше не «Оператор» по умолчанию."""
    segments, tracks = build_call(CLIENT_LINES, OPERATOR_LINES)

    make_diarizer(tracks).assign_speakers(segments, "call.wav")

    assert [segment.speaker for segment in segments] == [
        CLIENT_LABEL,
        OPERATOR_LABEL,
        CLIENT_LABEL,
        OPERATOR_LABEL,
        CLIENT_LABEL,
        OPERATOR_LABEL,
    ]


def test_operator_speaking_first_still_works():
    segments, tracks = build_call(OPERATOR_LINES, CLIENT_LINES)

    make_diarizer(tracks).assign_speakers(segments, "call.wav")

    assert segments[0].speaker == OPERATOR_LABEL
    assert segments[1].speaker == CLIENT_LABEL


def test_third_cluster_becomes_client_and_never_leaks_raw_label():
    segments, tracks = build_call(OPERATOR_LINES, CLIENT_LINES)
    extra_start = segments[-1].end + 1.0
    segments.append(Segment(start=extra_start, end=extra_start + 3.0, text="А я вообще мимо проходил."))
    tracks.append((FakeTurn(start=extra_start, end=extra_start + 3.0), "SPEAKER_02"))

    make_diarizer(tracks).assign_speakers(segments, "call.wav")

    assert segments[-1].speaker == CLIENT_LABEL
    assert {segment.speaker for segment in segments} == {OPERATOR_LABEL, CLIENT_LABEL}


class TestUnknownSpeakerRecovery:
    """Сегмент без пересечения подтягивается к ближайшему turn, если тот рядом."""

    def build(self, gap: float):
        segments, tracks = build_call(OPERATOR_LINES, CLIENT_LINES)
        last_end = segments[-1].end
        orphan_start = last_end + gap
        segments.append(Segment(start=orphan_start, end=orphan_start + 2.0, text="Да, конечно."))
        return segments, tracks

    def test_nearby_orphan_gets_speaker_of_closest_turn(self):
        segments, tracks = self.build(NEAREST_TURN_MAX_GAP_S / 2)

        make_diarizer(tracks).assign_speakers(segments, "call.wav")

        # Ближайший turn — последняя реплика клиента.
        assert segments[-1].speaker == CLIENT_LABEL

    def test_distant_orphan_stays_unknown(self):
        segments, tracks = self.build(NEAREST_TURN_MAX_GAP_S + 5.0)

        make_diarizer(tracks).assign_speakers(segments, "call.wav")

        assert segments[-1].speaker == UNKNOWN_SPEAKER


def test_uninitialized_pipeline_raises():
    from fastapi import HTTPException

    diarizer = make_diarizer([])
    diarizer.diarization_pipeline = None

    with pytest.raises(HTTPException) as excinfo:
        diarizer.assign_speakers([], "call.wav")

    assert excinfo.value.status_code == 500

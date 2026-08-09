"""Тесты смыслового определения ролей спикеров.

Работают на голом тексте: аудио, pyannote и моки не нужны. Ключевая проверка —
результат не зависит от того, какая метка кластера досталась оператору, то есть
регрессия на прежнее поведение «SPEAKER_00 всегда оператор».
"""

import re

import pytest

from asr.roles import (
    CLIENT_LABEL,
    METHOD_FALLBACK_ORDER,
    METHOD_MARKERS,
    OPERATOR_LABEL,
    assign_roles,
    normalize,
)
from tests.test_audio_fixtures import TEST_DATA

# Эталоны в test_data лежат в формате «Роль: текст» — тот же префикс разбирает
# SPEAKER_PREFIX в scripts/eval_wer.py.
SPEAKER_PREFIX = re.compile(r"^(Оператор|Клиент)\s*:\s*")

DIALOG_FIXTURES = ("call_dialog", "complaint_violation", "deposit_question", "card_block")


def load_reference(name: str) -> dict[str, list[str]]:
    """Разложить эталонный транскрипт на реплики по истинным ролям."""
    utterances: dict[str, list[str]] = {}
    text = (TEST_DATA / f"{name}.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        match = SPEAKER_PREFIX.match(line)
        if match:
            utterances.setdefault(match.group(1), []).append(line[match.end() :])
    return utterances


def test_normalize_strips_punctuation_and_yo():
    assert normalize("Всё  понятно, спасибо!") == "все понятно спасибо"


class TestReferenceDialogs:
    """Роль определяется содержанием, а не тем, какая метка досталась кластеру."""

    @pytest.mark.parametrize("name", DIALOG_FIXTURES)
    @pytest.mark.parametrize("operator_label", ["A", "B"])
    def test_operator_found_regardless_of_cluster_label(self, name, operator_label):
        reference = load_reference(name)
        client_label = "B" if operator_label == "A" else "A"
        clusters = {
            operator_label: reference["Оператор"],
            client_label: reference["Клиент"],
        }

        result = assign_roles(clusters)

        assert result.method == METHOD_MARKERS
        assert result.mapping[operator_label] == OPERATOR_LABEL
        assert result.mapping[client_label] == CLIENT_LABEL

    @pytest.mark.parametrize("name", DIALOG_FIXTURES)
    def test_scores_are_reported_for_every_cluster(self, name):
        reference = load_reference(name)

        result = assign_roles({"S0": reference["Оператор"], "S1": reference["Клиент"]})

        assert set(result.scores) == {"S0", "S1"}
        assert result.margin > 0


def test_client_monologue_is_not_labelled_operator():
    """short_greeting.txt — единственный спикер, и это клиент."""
    reference = load_reference("short_greeting")
    assert "Оператор" not in reference

    result = assign_roles({"SPEAKER_00": reference["Клиент"]})

    assert result.mapping == {"SPEAKER_00": CLIENT_LABEL}


def test_extra_clusters_all_become_clients():
    reference = load_reference("call_dialog")
    clients = reference["Клиент"]
    clusters = {
        "SPEAKER_00": clients[: len(clients) // 2],
        "SPEAKER_01": clients[len(clients) // 2 :],
        "SPEAKER_02": reference["Оператор"],
    }

    result = assign_roles(clusters)

    assert result.mapping["SPEAKER_02"] == OPERATOR_LABEL
    assert result.mapping["SPEAKER_00"] == CLIENT_LABEL
    assert result.mapping["SPEAKER_01"] == CLIENT_LABEL


def test_empty_input_yields_empty_mapping():
    result = assign_roles({})

    assert result.mapping == {}
    assert result.method == METHOD_FALLBACK_ORDER


def test_indistinguishable_clusters_fall_back_to_order():
    """Без смыслового сигнала возвращаемся к прежнему порядковому поведению."""
    clusters = {"SPEAKER_00": ["Ага"], "SPEAKER_01": ["Угу"]}

    result = assign_roles(clusters)

    assert result.method == METHOD_FALLBACK_ORDER
    assert result.mapping["SPEAKER_00"] == OPERATOR_LABEL
    assert result.mapping["SPEAKER_01"] == CLIENT_LABEL

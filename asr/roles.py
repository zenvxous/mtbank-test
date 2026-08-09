import re
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)

OPERATOR_LABEL = "Оператор"
CLIENT_LABEL = "Клиент"

METHOD_MARKERS = "markers"
METHOD_FALLBACK_ORDER = "fallback_order"

MARGIN_THRESHOLD = 0.35

WEIGHT_STRONG = 3.0
WEIGHT_WEAK = 1.0
WEIGHT_PRONOUNS = 2.0
WEIGHT_LENGTH = 0.5

STRONG_OPERATOR_MARKERS = (
    r"мтбанк",
    r"меня зовут",
    r"чем могу (?:помочь|быть полезн)",
    r"оставайтесь на линии",
    r"ваш звонок записывается",
    r"спасибо за обращение",
    r"(?:минуточк|секундочк)",
    r"уточните[^.?!]{0,30}пожалуйста",
    r"подскажите[^.?!]{0,30}пожалуйста",
    r"для вас действу",
    r"оформ(?:лю|им) заявк",
    r"провер(?:ю|им) информацию",
    r"могу (?:предложить|отправить)",
)

STRONG_CLIENT_MARKERS = (
    r"у меня (?:есть )?(?:карт|счет|кредит|вклад)",
    r"мо(?:я|ей|ю|ей) карт",
    r"мой счет",
    r"хочу (?:узнать|уточнить|оформить|пожаловаться)",
    r"хотел(?:а)? бы",
    r"мне нужно",
    r"я не совершал",
    r"с моей карты",
    r"у меня проблема",
)

WEAK_OPERATOR_MARKERS = (
    r"заявк",
    r"тариф",
    r"условия по",
)

WEAK_CLIENT_MARKERS = (
    r"а можно",
    r"не понимаю",
    r"почему",
)

SECOND_PERSON = re.compile(r"\b(?:вы|вам|вас|вами|ваш\w*)\b")
FIRST_PERSON = re.compile(r"\b(?:я|мне|меня|мной|мо[йяеию]\w*)\b")

_PUNCTUATION = re.compile(r"[^\w\s]+")
_WHITESPACE = re.compile(r"\s+")


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


_STRONG_OPERATOR = _compile(STRONG_OPERATOR_MARKERS)
_STRONG_CLIENT = _compile(STRONG_CLIENT_MARKERS)
_WEAK_OPERATOR = _compile(WEAK_OPERATOR_MARKERS)
_WEAK_CLIENT = _compile(WEAK_CLIENT_MARKERS)


@dataclass
class RoleAssignment:
    """Результат назначения ролей кластерам диаризации."""

    mapping: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    method: str = METHOD_FALLBACK_ORDER


def normalize(text: str) -> str:
    """Привести реплику к виду, на котором стабильно матчатся маркеры."""
    lowered = text.lower().replace("ё", "е")
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", lowered)).strip()


def _marker_score(text: str) -> float:
    """Разница операторских и клиентских лексических маркеров в тексте кластера.

    Считаем число сработавших паттернов, а не число вхождений: иначе кластер,
    десять раз повторивший «карта», перевесил бы содержательные маркеры.
    """
    score = 0.0
    for pattern in _STRONG_OPERATOR:
        if pattern.search(text):
            score += WEIGHT_STRONG
    for pattern in _STRONG_CLIENT:
        if pattern.search(text):
            score -= WEIGHT_STRONG
    for pattern in _WEAK_OPERATOR:
        if pattern.search(text):
            score += WEIGHT_WEAK
    for pattern in _WEAK_CLIENT:
        if pattern.search(text):
            score -= WEIGHT_WEAK
    return score


def _pronoun_score(text: str) -> float:
    """Асимметрия «вы» против «я»: оператор обращается, клиент рассказывает о себе.

    Признак языковой, а не сценарный, поэтому работает и на репликах вне скрипта.
    """
    second = len(SECOND_PERSON.findall(text))
    first = len(FIRST_PERSON.findall(text))
    total = second + first
    if total == 0:
        return 0.0
    return (second - first) / total


def score_cluster(utterances: list[str]) -> tuple[float, float]:
    """Вернуть (содержательный score кластера, среднюю длину реплики в словах).

    Маркеры нормируются на число реплик: иначе оператором объявлялся бы просто
    тот, кто дольше говорил.
    """
    if not utterances:
        return 0.0, 0.0

    normalized = [normalize(utterance) for utterance in utterances]
    joined = " ".join(normalized)

    markers = _marker_score(joined) / len(normalized) ** 0.5
    pronouns = _pronoun_score(joined) * WEIGHT_PRONOUNS
    avg_words = sum(len(utterance.split()) for utterance in normalized) / len(normalized)

    return markers + pronouns, avg_words


def assign_roles(cluster_texts: dict[str, list[str]]) -> RoleAssignment:
    """Назначить ролям кластеров диаризации метки «Оператор» / «Клиент».

    Оператором становится кластер с максимальным score, остальные — клиенты.
    Если отрыв лидера меньше `MARGIN_THRESHOLD`, содержательного сигнала не
    хватает и мы откатываемся на порядковый выбор (лексикографически
    минимальная метка), помечая это в `method`.
    """
    if not cluster_texts:
        return RoleAssignment(method=METHOD_FALLBACK_ORDER)

    raw: dict[str, tuple[float, float]] = {
        label: score_cluster(utterances) for label, utterances in cluster_texts.items()
    }

    max_length = max((length for _, length in raw.values()), default=0.0)
    scores = {
        label: score + (WEIGHT_LENGTH * length / max_length if max_length > 0 else 0.0)
        for label, (score, length) in raw.items()
    }

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    leader, leader_score = ranked[0]

    if len(ranked) > 1:
        margin = leader_score - ranked[1][1]
        operator: str | None = leader
    else:
        margin = abs(leader_score)
        operator = leader if leader_score > 0 else None

    method = METHOD_MARKERS
    if margin < MARGIN_THRESHOLD:
        operator = min(cluster_texts)
        method = METHOD_FALLBACK_ORDER
        log.warning(
            "role_assignment_low_confidence",
            margin=round(margin, 3),
            scores={label: round(value, 3) for label, value in scores.items()},
        )

    mapping = {label: OPERATOR_LABEL if label == operator else CLIENT_LABEL for label in cluster_texts}

    return RoleAssignment(
        mapping=mapping,
        scores={label: round(value, 3) for label, value in scores.items()},
        margin=round(margin, 3),
        method=method,
    )

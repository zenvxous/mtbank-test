"""Интеграционный тест всего пайплайна на реальной LLM.

LLM здесь НЕ мокается: агенты ходят в OpenRouter под ключом из .env.
Проверяем контракт и смысл ответа, а не конкретные формулировки — модель
недетерминирована даже при temperature=0.

Запуск:  uv run pytest -m integration
Пропуск: uv run pytest -m "not integration"
"""

import pytest

from agents.classifier import Priority, Topic
from agents.compliance import Severity
from agents.graph import run_analysis
from api.config import settings
from api.main import AnalyzeResponse

pytestmark = pytest.mark.integration

CHECKLIST_KEYS = {"greeting", "need_detection", "solution_provided", "farewell"}


@pytest.fixture(scope="module", autouse=True)
def require_llm_credentials():
    if not settings.OPENROUTER_API_KEY or not settings.OPENROUTER_MODEL:
        pytest.skip("OPENROUTER_API_KEY/OPENROUTER_MODEL не заданы — реальный прогон пропущен")


@pytest.fixture(scope="module")
def clean_call_transcript() -> list[dict]:
    """Корректный звонок: оператор представился, выявил потребность, решил вопрос, попрощался."""
    return [
        {"speaker": "Оператор", "start": 0.0, "end": 5.1, "text": "Добрый день, МТБанк, меня зовут Анна. Чем могу помочь?"},
        {
            "speaker": "Клиент",
            "start": 5.4,
            "end": 12.7,
            "text": "Здравствуйте. Я потерял свою карту вчера вечером, боюсь, что ей кто-то воспользуется.",
        },
        {
            "speaker": "Оператор",
            "start": 13.0,
            "end": 22.5,
            "text": "Понимаю вас. Скажите, пожалуйста, вы хотите заблокировать карту и заказать перевыпуск?",
        },
        {"speaker": "Клиент", "start": 22.8, "end": 26.4, "text": "Да, заблокируйте и сделайте новую, пожалуйста."},
        {
            "speaker": "Оператор",
            "start": 26.7,
            "end": 38.2,
            "text": (
                "Карту заблокировал, операции по ней больше невозможны. "
                "Заявку на перевыпуск оформил, новая карта будет готова в вашем отделении в течение пяти рабочих дней."
            ),
        },
        {"speaker": "Клиент", "start": 38.5, "end": 42.0, "text": "Отлично, спасибо большое за помощь."},
        {
            "speaker": "Оператор",
            "start": 42.3,
            "end": 48.0,
            "text": "Пожалуйста. Если появятся вопросы, звоните нам в любое время. Спасибо за обращение, всего доброго!",
        },
    ]


@pytest.fixture(scope="module")
def violating_call_transcript() -> list[dict]:
    """Звонок с нарушениями регламента: гарантия одобрения и запрос кода из SMS."""
    return [
        {"speaker": "Оператор", "start": 0.0, "end": 2.4, "text": "Да, слушаю."},
        {
            "speaker": "Клиент",
            "start": 2.7,
            "end": 9.5,
            "text": "Здравствуйте, я хотел бы узнать условия по кредиту на ремонт квартиры.",
        },
        {
            "speaker": "Оператор",
            "start": 9.8,
            "end": 18.6,
            "text": "Вам точно одобрят кредит под 9 процентов годовых, это я вам гарантирую на сто процентов.",
        },
        {
            "speaker": "Оператор",
            "start": 18.9,
            "end": 27.3,
            "text": "Чтобы оформить прямо сейчас, назовите мне код из СМС, который вам пришёл, и полный номер вашей карты.",
        },
        {"speaker": "Клиент", "start": 27.6, "end": 31.0, "text": "Э-э, а это точно нужно?"},
        {"speaker": "Оператор", "start": 31.2, "end": 36.8, "text": "Решайте прямо сейчас, потом такого предложения не будет."},
    ]


@pytest.fixture(scope="module")
def clean_call_state(clean_call_transcript) -> dict:
    """Один реальный прогон графа на модуль: четыре агента, четыре вызова модели."""
    return run_analysis(clean_call_transcript)


@pytest.fixture(scope="module")
def violating_call_state(violating_call_transcript) -> dict:
    return run_analysis(violating_call_transcript)


class TestRealPipelineContract:
    def test_state_contains_all_contract_keys(self, clean_call_state, clean_call_transcript):
        assert set(clean_call_state) >= {
            "transcript",
            "classification",
            "quality_score",
            "compliance",
            "summary",
            "action_items",
        }
        assert clean_call_state["transcript"] == clean_call_transcript

    def test_result_validates_against_api_response_model(self, clean_call_state):
        """Реальный выход графа должен подходить под ответ /analyze из README."""
        AnalyzeResponse.model_validate(clean_call_state)

    def test_classification_uses_known_enum_values(self, clean_call_state):
        classification = clean_call_state["classification"]

        assert classification["topic"] in set(Topic)
        assert classification["priority"] in set(Priority)

    def test_quality_score_is_well_formed(self, clean_call_state):
        quality = clean_call_state["quality_score"]

        assert isinstance(quality["total"], int)
        assert 0 <= quality["total"] <= 100
        assert set(quality["checklist"]) == CHECKLIST_KEYS
        assert all(isinstance(value, bool) for value in quality["checklist"].values())
        assert quality["comment"].strip()

    def test_compliance_invariant_holds(self, clean_call_state):
        compliance = clean_call_state["compliance"]

        assert isinstance(compliance["passed"], bool)
        assert isinstance(compliance["issues"], list)
        assert compliance["passed"] == (not compliance["issues"])

    def test_summary_and_action_items_are_usable(self, clean_call_state):
        assert clean_call_state["summary"].strip()
        assert isinstance(clean_call_state["action_items"], list)
        assert all(isinstance(item, str) and item.strip() for item in clean_call_state["action_items"])


class TestRealPipelineSemantics:
    def test_lost_card_call_is_classified_as_cards(self, clean_call_state):
        assert clean_call_state["classification"]["topic"] == Topic.CARDS

    def test_correct_call_gets_greeting_and_farewell(self, clean_call_state):
        checklist = clean_call_state["quality_score"]["checklist"]

        assert checklist["greeting"] is True
        assert checklist["farewell"] is True

    def test_violations_are_detected(self, violating_call_state):
        compliance = violating_call_state["compliance"]

        assert compliance["passed"] is False
        assert compliance["issues"], "нарушения регламента должны быть найдены"

        for issue in compliance["issues"]:
            assert set(issue) == {"rule", "severity", "quote", "comment"}
            assert issue["severity"] in set(Severity)
            assert issue["quote"].strip()

    def test_violating_call_scores_lower_than_clean_one(self, clean_call_state, violating_call_state):
        assert violating_call_state["quality_score"]["total"] < clean_call_state["quality_score"]["total"]

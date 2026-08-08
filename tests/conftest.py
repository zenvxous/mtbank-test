"""Общие фикстуры тестов.

Ключевой приём: узлы агентов импортируют `get_llm` к себе в модуль
(`from agents import ... get_llm`), поэтому в unit-тестах патчить нужно
именно `agents.<модуль>.get_llm`, а не `agents.get_llm`.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from api.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent

AGENT_MODULES = ("agents.classifier", "agents.quality", "agents.compliance", "agents.summarizer")


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Фиктивные креды LLM: в корне лежит реальный .env, полагаться на него нельзя.

    Не autouse — интеграционные тесты работают с настоящим ключом из окружения.
    """
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "test/model", raising=False)


class LLMCall:
    """Запись одного обращения к LLM: чем сконфигурировали и с чем вызвали цепочку."""

    def __init__(self, max_completion_tokens: int | None) -> None:
        self.max_completion_tokens = max_completion_tokens
        self.schema: Any = None
        self.inputs: dict[str, Any] | None = None


class FakeLLM:
    """Заглушка ChatOpenAI: отдаёт готовый pydantic-объект или бросает исключение."""

    def __init__(self, result: Any, call: LLMCall, error: Exception | None = None) -> None:
        self._result = result
        self._call = call
        self._error = error

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> RunnableLambda:
        self._call.schema = schema

        def _invoke(inputs: Any) -> Any:
            # На вход узлу приходит ChatPromptValue, но нам важны переменные шаблона,
            # которые узел передал в chain.invoke — их пишет patch_llm ниже.
            if self._error is not None:
                raise self._error
            return self._result

        return RunnableLambda(_invoke)


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch):
    """Подменяет get_llm в модуле агента и возвращает запись о вызове."""

    def _patch(module_name: str, result: Any = None, error: Exception | None = None) -> LLMCall:
        module = sys.modules[module_name] if module_name in sys.modules else importlib.import_module(module_name)
        call = LLMCall(max_completion_tokens=None)

        def _get_llm(max_completion_tokens: int = 2000) -> FakeLLM:
            call.max_completion_tokens = max_completion_tokens
            return FakeLLM(result, call, error)

        monkeypatch.setattr(module, "get_llm", _get_llm)

        # Переменные шаблона видны только на входе в цепочку, поэтому перехватываем
        # ChatPromptTemplate.invoke внутри модуля агента.
        original_invoke = module.ChatPromptTemplate.invoke

        def _invoke(self, input_: Any, *args: Any, **kwargs: Any) -> Any:
            call.inputs = input_
            return original_invoke(self, input_, *args, **kwargs)

        monkeypatch.setattr(module.ChatPromptTemplate, "invoke", _invoke)
        return call

    return _patch


@pytest.fixture
def forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Любое обращение к LLM в тесте — ошибка (проверяем short-circuit на пустом транскрипте)."""

    def _boom(*_args: Any, **_kwargs: Any):
        raise AssertionError("LLM не должна вызываться")

    for module_name in AGENT_MODULES:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "get_llm", _boom)


@pytest.fixture
def transcript() -> list[dict[str, Any]]:
    """Короткий корректный диалог: оператор представился, выявил потребность, решил вопрос."""
    return [
        {"speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "Добрый день, МТБанк, меня зовут Анна. Чем могу помочь?"},
        {"speaker": "Клиент", "start": 4.5, "end": 9.8, "text": "Здравствуйте. Я потерял карту, её нужно срочно заблокировать."},
        {
            "speaker": "Оператор",
            "start": 10.0,
            "end": 17.4,
            "text": "Понял вас. Карту блокирую прямо сейчас, перевыпуск займёт до пяти рабочих дней.",
        },
        {"speaker": "Клиент", "start": 17.6, "end": 20.1, "text": "Спасибо, всё понятно."},
        {"speaker": "Оператор", "start": 20.3, "end": 24.0, "text": "Спасибо за обращение, хорошего дня!"},
    ]


@pytest.fixture
def violating_transcript() -> list[dict[str, Any]]:
    """Диалог с явными нарушениями: запрос кода из SMS и гарантия одобрения кредита."""
    return [
        {"speaker": "Оператор", "start": 0.0, "end": 3.0, "text": "Слушаю вас."},
        {"speaker": "Клиент", "start": 3.2, "end": 8.0, "text": "Здравствуйте, хочу узнать про кредит на ремонт."},
        {
            "speaker": "Оператор",
            "start": 8.2,
            "end": 16.0,
            "text": "Вам точно одобрят кредит под 9% годовых, это я вам гарантирую.",
        },
        {
            "speaker": "Оператор",
            "start": 16.2,
            "end": 22.5,
            "text": "Назовите, пожалуйста, код из SMS, который вам сейчас пришёл.",
        },
    ]


@pytest.fixture
def analyze_result() -> dict[str, Any]:
    """Ответ /analyze в формате контракта README — вход для форматирования отчёта."""
    return {
        "transcript": [
            {"speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "Добрый день, МТБанк, меня зовут Анна."},
            {"speaker": "Клиент", "start": 65.5, "end": 70.1, "text": "Здравствуйте, хочу узнать про кредит."},
        ],
        "classification": {"topic": "credits", "priority": "medium"},
        "quality_score": {
            "total": 78,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": False,
            },
            "comment": "Оператор вежлив, но не попрощался.",
        },
        "compliance": {"passed": True, "issues": []},
        "summary": "Клиент обратился по вопросу кредита.",
        "action_items": ["Отправить КП на email клиента"],
    }


@pytest.fixture(scope="session")
def owui_module():
    """Загружает pipelines/pipeline.py по пути.

    Обычный import не годится: рядом лежит каталог pipelines/pipeline/ с valves.json,
    который затеняет модуль как namespace-пакет.
    """
    path = REPO_ROOT / "pipelines" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("owui_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["owui_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def owui_pipeline(owui_module):
    return owui_module.Pipeline()

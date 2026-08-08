import time
from enum import StrEnum
from typing import cast

import structlog
from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field

from agents import AnalyzeState, get_llm

log = structlog.get_logger(__name__)


class Topic(StrEnum):
    CREDITS = "credits"
    CARDS = "cards"
    TRANSFERS = "transfers"
    COMPLAINTS = "complaints"
    OTHER = "other"

class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Classification(BaseModel):
    topic: Topic = Field(description="Основная тематика обращения")
    priority: Priority = Field(description="Приоритет обработки обращения")

SYSTEM_PROMPT = """
Ты — классификатор обращений клиентов банка.

Проанализируй текст диалога и верни результат строго в формате JSON.
Верни только один валидный JSON-объект без Markdown, пояснений и дополнительного текста:

Строгая JSON-схема результата:
{{
  "topic": "credits | cards | transfers | complaints | other",
  "priority": "low | medium | high | critical"
}}

Определи:
1. topic — одну основную тему, то есть предмет главной проблемы или запроса клиента.
2. priority — операционный приоритет обработки.

<topics>
credits:
- кредит, ипотека, рассрочка;
- платёж, просрочка, реструктуризация;
- процентная ставка, задолженность, кредитный лимит.

cards:
- выпуск, перевыпуск, доставка, активация или блокировка карты;
- PIN, бесконтактная оплата, Apple Pay, Google Pay;
- снятие наличных, операция или списание по карте;
- потеря, кража или компрометация карты.

transfers:
- внутренний или международный  перевод;
- реквизиты, получатель, назначение, комиссия;
- задержка, отмена или ошибочный перевод.

complaints:
- основной запрос клиента — зафиксировать жалобу, претензию
  или оспорить качество сервиса, работу сотрудника либо решение банка;
- грубость, длительное ожидание, отказ в обслуживании,
  требование официального ответа или претензии.

other:
- обращение не относится к категориям выше;
- в диалоге недостаточно информации для определения предметной темы.
</topics>

<topic_selection>
- Анализируй смысл всего диалога, а не отдельные слова.
- Выбирай только одну тему: тему главной проблемы или цели клиента.
- Если клиент недоволен проблемой с картой, кредитом или переводом,
  но не просит оформить жалобу как основное действие, выбирай предметную тему:
  cards, credits или transfers.
- Выбирай complaints, если регистрация жалобы, претензии или оценка работы
  банка — главная цель обращения.
- Риск мошенничества или безопасности влияет на priority, но не заменяет topic.
  Например: «карту украли» -> topic=cards, priority=critical.
- Не выдумывай факты, которых нет в диалоге.
</topic_selection>

<priority_selection>
critical:
- предполагаемое мошенничество;
- несанкционированная операция, которую клиент не совершал;
- взлом аккаунта, утечка или компрометация данных;
- потеря или кража карты с риском доступа к деньгам;
- угроза жизни или физической безопасности.

high:
- просрочка по кредиту;
- клиент лишён доступа к деньгам или важному банковскому сервису;
- перевод или платёж не поступил и клиент сообщает о срочной финансовой проблеме;
- ошибочное списание, двойное списание или существенная финансовая потеря;
- серьёзная жалоба с риском эскалации.

medium:
- требуется проверка операции или статуса;
- проблема требует исправления, но нет явного немедленного риска;
- уточнение условий, комиссий, сроков или порядка действий.

low:
- справочная консультация;
- информация о продукте;
- стандартная процедура без проблемы, потери денег или срочности.

Правила:
- Слово «срочно» само по себе не повышает priority.
- Если признаков critical или high нет, выбирай medium или low.
- При недостатке фактов не завышай приоритет.
</priority_selection>

<examples>
Диалог: «С карты списали 15 000 рублей, я эту покупку не совершал».
Результат: topic=cards, priority=critical.

Диалог: «Почему перевод по номеру телефона всё ещё не пришёл получателю? Отправил час назад».
Результат: topic=transfers, priority=medium.

Диалог: «Перевод не дошёл, из-за этого я не могу оплатить лечение сегодня».
Результат: topic=transfers, priority=high.

Диалог: «Сотрудник в чате грубит. Хочу оставить официальную претензию».
Результат: topic=complaints, priority=high.

Диалог: «Как заказать дополнительную карту для ребёнка?».
Результат: topic=cards, priority=low.
</examples>

Правила:
1. Анализируй смысл всего диалога, а не отдельные ключевые слова.
2. Если в диалоге несколько тем, выбери тему основной проблемы клиента.
3. Не выдумывай отсутствующие факты.
"""

def classification_node(state: AnalyzeState) -> dict:
    dialog = state.get("transcript", [])
    if not dialog:
        log.warning("agent_skipped", agent="classifier", reason="empty_transcript")
        return {"classification": {"topic": Topic.OTHER, "priority": Priority.LOW}}

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("Диалог: {dialog}"),
        ]
    )

    log.info("agent_started", agent="classifier", segments=len(dialog))
    started = time.perf_counter()

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(Classification)

        chain = prompt | structured_llm
        result = cast(Classification, chain.invoke({"dialog": dialog}))
        log.info(
            "agent_completed",
            agent="classifier",
            topic=result.topic,
            priority=result.priority,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

        return {"classification": {"topic": result.topic, "priority": result.priority}}
    except Exception as e:
        log.error(
            "agent_failed",
            agent="classifier",
            segments=len(dialog),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            exc_info=True,
        )
        raise HTTPException(status_code=422, detail=f"Failed to classify dialog: {e}") from e

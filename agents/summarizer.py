import logging
from typing import cast

from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field

from agents import AnalyzeState, get_llm

logger = logging.getLogger(__name__)

SUMMARY_MAX_TOKENS = 4000


class Summary(BaseModel):
    summary: str = Field(description="Краткое резюме разговора, от 3 до 5 предложений")
    action_items: list[str] = Field(description="Список конкретных задач по итогам звонка, может быть пустым")

SYSTEM_PROMPT = """
Ты — аналитик контакт-центра банка. Ты составляешь краткую сводку по звонку
для супервайзера и для следующего оператора, который будет вести этого клиента.

Проанализируй диалог и результаты предыдущих агентов, затем верни результат
строго в формате JSON.
Верни только один валидный JSON-объект без Markdown, пояснений и дополнительного текста:

Строгая JSON-схема результата:
{{
  "summary": "резюме разговора из 3-5 предложений",
  "action_items": ["конкретная задача", "..."]
}}

Помимо диалога тебе передаются результаты трёх агентов:
- classification — тема обращения и приоритет;
- quality_score — оценка работы оператора по чеклисту;
- compliance — найденные нарушения регламента.
Используй их как контекст: приоритет и нарушения влияют на то, какие задачи попадут
в action_items. Не пересказывай сами оценки в summary — там описывается суть разговора.

<summary_rules>
- Ровно от 3 до 5 предложений, обычный связный текст на русском языке.
- Без Markdown, без списков, без заголовков, без эмодзи.
- Структура: с чем обратился клиент -> что выяснилось -> что сделал оператор -> чем закончился разговор.
- Только факты из диалога. Не додумывай суммы, сроки, имена и обстоятельства.
- Если разговор оборван или содержит мало информации, честно отрази это в резюме.
</summary_rules>

<action_items_rules>
- Каждый пункт — одно конкретное действие: глагол в инфинитиве + объект.
  Например: «Отправить клиенту условия кредита на email», «Перевыпустить карту и уведомить клиента».
- Включай только те задачи, которые прямо следуют из диалога:
  обещания оператора, нерешённые вопросы клиента, необходимые проверки.
- Если compliance нашёл нарушения высокой критичности, добавь задачу для супервайзера,
  например «Провести разбор звонка с оператором по факту запроса кода из SMS».
- Не превращай в задачи то, что уже сделано во время разговора.
- Не выдумывай задачи ради заполнения списка. Пустой список — допустимый и правильный
  результат, если по итогам звонка делать нечего.
- Максимум 6 пунктов, самые важные сначала.
</action_items_rules>

<examples>
Диалог: клиент сообщил о списании 15 000 рублей, которого не совершал; оператор
заблокировал карту и оформил заявление на оспаривание операции.
summary: 4 предложения о сути обращения, блокировке карты и оформленном заявлении.
action_items: ["Оспорить операцию на 15 000 рублей и сообщить клиенту результат",
"Перевыпустить заблокированную карту"].

Диалог: клиент спросил, как заказать дополнительную карту для ребёнка; оператор
объяснил порядок и условия, клиент поблагодарил.
summary: 3 предложения о справочном характере обращения и данном ответе.
action_items: [].
</examples>

Правила:
1. Пиши только на русском языке.
2. Не выдумывай факты, которых нет в диалоге.
3. Соблюдай ограничение 3-5 предложений в summary.
"""

def summarization_node(state: AnalyzeState) -> dict:
    dialog = state.get("transcript", [])
    if not dialog:
        logger.warning("Summarizer agent: пустой транскрипт, резюме не составлено")
        return {"summary": "Транскрипт пуст, резюме составить невозможно.", "action_items": []}

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "Классификация обращения: {classification}\n"
                "Оценка качества: {quality_score}\n"
                "Результат compliance-проверки: {compliance}\n\n"
                "Диалог: {dialog}"
            ),
        ]
    )

    logger.info("Summarizer agent: старт, сегментов=%d", len(dialog))

    try:
        llm = get_llm(max_completion_tokens=SUMMARY_MAX_TOKENS)
        structured_llm = llm.with_structured_output(Summary)

        chain = prompt | structured_llm
        result = cast(
            Summary,
            chain.invoke(
                {
                    "dialog": dialog,
                    "classification": state.get("classification", {}),
                    "quality_score": state.get("quality_score", {}),
                    "compliance": state.get("compliance", {}),
                }
            ),
        )
        logger.info("Summarizer agent: резюме готово, action_items=%d", len(result.action_items))

        return {"summary": result.summary, "action_items": result.action_items}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to summarize dialog: {e}") from e

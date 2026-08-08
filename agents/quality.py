import logging
from typing import cast

from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field

from agents import AnalyzeState, get_llm

logger = logging.getLogger(__name__)

QUALITY_MAX_TOKENS = 3000


class QualityChecklist(BaseModel):
    greeting: bool = Field(description="Оператор поздоровался и представился")
    need_detection: bool = Field(description="Оператор выявил потребность клиента")
    solution_provided: bool = Field(description="Оператор предложил решение или чёткие дальнейшие шаги")
    farewell: bool = Field(description="Оператор корректно завершил разговор")

class QualityScore(BaseModel):
    total: int = Field(ge=0, le=100, description="Итоговая оценка качества работы оператора от 0 до 100")
    checklist: QualityChecklist = Field(description="Результат по каждому пункту чеклиста")
    comment: str = Field(description="Краткое обоснование оценки, 1-2 предложения")

SYSTEM_PROMPT = """
Ты — супервайзер контакт-центра банка. Ты оцениваешь качество работы оператора по звонку.

Проанализируй диалог и верни результат строго в формате JSON.
Верни только один валидный JSON-объект без Markdown, пояснений и дополнительного текста:

Строгая JSON-схема результата:
{{
  "total": 0-100,
  "checklist": {{
    "greeting": true | false,
    "need_detection": true | false,
    "solution_provided": true | false,
    "farewell": true | false
  }},
  "comment": "краткое обоснование оценки"
}}

Оценивай только реплики со speaker = "Оператор".
Реплики клиента — это контекст, за них оператор не отвечает.
Реплики со speaker = "Неизвестно" учитывай по смыслу: если по содержанию видно,
что это говорит оператор, засчитывай пункт.

<checklist>
greeting — приветствие:
- оператор поздоровался в начале разговора;
- назвал банк и/или своё имя.
Засчитывай true, если есть приветствие. Отсутствие имени снижает оценку через total,
но само по себе не делает пункт false.

need_detection — выявление потребности:
- оператор задал уточняющие вопросы;
- переформулировал или подтвердил запрос клиента;
- выяснил детали, необходимые для решения (продукт, сумма, сроки, обстоятельства).
Засчитывай false, если оператор сразу начал говорить, не разобравшись в проблеме.

solution_provided — решение:
- оператор дал ответ по существу, предложил вариант решения
  или назвал конкретные дальнейшие шаги (заявка, перевод на профильного специалиста, сроки).
Засчитывай false, если оператор ушёл от ответа или оставил клиента без результата.

need_detection и solution_provided — ключевые пункты, они весят больше остальных.

farewell — прощание:
- оператор подвёл итог, спросил, есть ли ещё вопросы, попрощался или поблагодарил за обращение.
Засчитывай false, если разговор оборван без завершающей реплики оператора.
</checklist>

<scoring>
Ориентировочные веса пунктов при подсчёте total:
- greeting — до 20 баллов;
- need_detection — до 30 баллов;
- solution_provided — до 35 баллов;
- farewell — до 15 баллов.

Начисляй баллы за выполненные пункты и корректируй итог в пределах веса пункта
по качеству исполнения: формальное или неполное выполнение — неполный балл.
Дополнительно учитывай вежливость, ясность речи и отсутствие лишних пауз,
но не выходи за общие рамки весов.

Правила:
- total должен быть согласован с чеклистом: если все пункты false, total близок к 0;
  если все пункты выполнены качественно, total близок к 100.
- Если транскрипт короткий или обрывочный, не занижай оценку агрессивно:
  оценивай то, что реально прозвучало.
</scoring>

<examples>
Диалог: оператор поздоровался и представился, уточнил сумму и срок кредита,
объяснил условия, оформил заявку, попрощался.
Результат: все пункты true, total ≈ 95.

Диалог: оператор поздоровался, но сразу зачитал условия продукта,
не выяснив запрос клиента, и завершил разговор без прощания.
Результат: greeting=true, need_detection=false, solution_provided=true, farewell=false, total ≈ 50.

Диалог: оператор без приветствия грубо ответил «мы этим не занимаемся» и отключился.
Результат: все пункты false, total ≈ 5.
</examples>

Правила:
1. Оценивай только то, что есть в диалоге, не додумывай реплики оператора.
2. Не завышай оценку из-за вежливого тона, если задача клиента не решена.
3. comment пиши на русском языке, 1-2 предложения, по существу.
"""

def quality_node(state: AnalyzeState) -> dict:
    dialog = state.get("transcript", [])
    if not dialog:
        logger.warning("Quality agent: пустой транскрипт, возвращаю нулевую оценку")
        return {
            "quality_score": {
                "total": 0,
                "checklist": {
                    "greeting": False,
                    "need_detection": False,
                    "solution_provided": False,
                    "farewell": False,
                },
                "comment": "Транскрипт пуст, оценить работу оператора невозможно.",
            }
        }

    classification = state.get("classification", {})

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("Классификация обращения: {classification}\n\nДиалог: {dialog}"),
        ]
    )

    logger.info("Quality agent: старт, сегментов=%d", len(dialog))

    try:
        llm = get_llm(max_completion_tokens=QUALITY_MAX_TOKENS)
        structured_llm = llm.with_structured_output(QualityScore)

        chain = prompt | structured_llm
        result = cast(QualityScore, chain.invoke({"dialog": dialog, "classification": classification}))

        quality_score = {
            "total": result.total,
            "checklist": result.checklist.model_dump(),
            "comment": result.comment,
        }
        logger.info("Quality agent: результат %s", quality_score)

        return {"quality_score": quality_score}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to evaluate dialog quality: {e}") from e

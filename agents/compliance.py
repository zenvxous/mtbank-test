import logging
from enum import StrEnum
from typing import cast

from fastapi import HTTPException
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel, Field

from agents import AnalyzeState, get_llm

logger = logging.getLogger(__name__)

COMPLIANCE_MAX_TOKENS = 4000


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ComplianceIssue(BaseModel):
    rule: str = Field(description="Какое правило нарушено, кратко")
    severity: Severity = Field(description="Критичность нарушения")
    quote: str = Field(description="Дословная цитата из реплики оператора")
    comment: str = Field(description="Пояснение, почему это нарушение")

class ComplianceResult(BaseModel):
    passed: bool = Field(description="Проверка пройдена: нарушений не обнаружено")
    issues: list[ComplianceIssue] = Field(description="Список найденных нарушений, пустой если их нет")

SYSTEM_PROMPT = """
Ты — комплаенс-контролёр банка. Ты проверяешь звонок оператора контакт-центра
на соответствие требованиям регламента.

Проанализируй диалог и верни результат строго в формате JSON.
Верни только один валидный JSON-объект без Markdown, пояснений и дополнительного текста:

Строгая JSON-схема результата:
{{
  "passed": true | false,
  "issues": [
    {{
      "rule": "краткое название нарушенного правила",
      "severity": "low | medium | high",
      "quote": "дословная цитата из диалога",
      "comment": "почему это нарушение"
    }}
  ]
}}

Проверяй только реплики со speaker = "Оператор".
Реплики клиента нарушением быть не могут — клиент может грубить или называть свои данные,
это не вина оператора. Реплики со speaker = "Неизвестно" проверяй, только если
по содержанию очевидно, что говорит оператор.

<forbidden_phrases>
Запрещённые фразы и поведение оператора:
- грубость, оскорбления, пренебрежительный или раздражённый тон, повышение голоса;
- обещание гарантированного результата: «одобрение 100%», «вам точно одобрят»,
  «гарантированная доходность», «вы ничего не потеряете»;
- давление на клиента, навязывание продукта после отказа, срочность в манипулятивной форме
  («решайте прямо сейчас, потом не будет»);
- запрос полного номера карты, CVV/CVC-кода, ПИН-кода, пароля или кода из SMS;
- предложение обойти процедуру банка, дать «неофициальный» совет;
- разглашение данных клиента третьим лицам или обсуждение других клиентов;
- перекладывание вины на клиента, отказ в обслуживании без основания.
severity: грубость и запрос секретных данных — high; обещания и давление — medium.
</forbidden_phrases>

<required_disclaimers>
Обязательные элементы, отсутствие которых является нарушением:
- оператор представился: назвал банк и своё имя в начале разговора (severity: low);
- при обсуждении кредита, рассрочки или кредитного лимита оператор оговорил,
  что окончательное решение принимает банк и требуется рассмотрение заявки (severity: medium);
- при предложении продукта оператор упомянул, что действуют условия договора
  и что с ними нужно ознакомиться (severity: low);
- при операциях с деньгами или изменении данных оператор идентифицировал клиента
  корректным способом, а не запросом секретных реквизитов (severity: medium).

Проверяй обязательный элемент только тогда, когда для него есть повод в диалоге.
Если про кредит не говорили — отсутствие кредитной оговорки не нарушение.
</required_disclaimers>

<offer_correctness>
Корректность предложений:
- оператор не должен называть точную ставку, сумму или срок как окончательные
  без оговорки «предварительно» / «по результатам рассмотрения» (severity: medium);
- оператор не должен ссылаться на несуществующие акции, условия или продукты (severity: high);
- оператор не должен обещать сроки, которые банк не контролирует
  («деньги придут через минуту», «вопрос решится сегодня») (severity: medium);
- информация в разных репликах оператора не должна противоречить сама себе (severity: medium).
</offer_correctness>

<examples>
Диалог: оператор представился, обсуждали только блокировку карты, говорил вежливо.
Результат: passed=true, issues=[].

Диалог: оператор говорит «Назовите, пожалуйста, код из SMS, который вам пришёл».
Результат: passed=false, одно нарушение, rule="Запрос секретных данных клиента",
severity=high, quote="Назовите, пожалуйста, код из SMS, который вам пришёл".

Диалог: оператор говорит «Вам точно одобрят кредит под 9% годовых».
Результат: passed=false, два нарушения: гарантия одобрения (high)
и точная ставка без оговорки о рассмотрении заявки (medium).
</examples>

Правила:
1. passed = true тогда и только тогда, когда список issues пуст.
2. quote — строго дословный фрагмент реплики из диалога, а не пересказ.
   Если точную цитату привести нельзя, нарушение не фиксируй.
3. Не выдумывай нарушения при недостатке данных: короткий или обрывочный транскрипт
   сам по себе нарушением не является.
4. Одно нарушение — один элемент issues, не дублируй одно и то же правило.
5. rule и comment пиши на русском языке, кратко.
"""

def compliance_node(state: AnalyzeState) -> dict:
    dialog = state.get("transcript", [])
    if not dialog:
        logger.warning("Compliance agent: пустой транскрипт, проверка пропущена")
        return {"compliance": {"passed": True, "issues": []}}

    classification = state.get("classification", {})

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("Классификация обращения: {classification}\n\nДиалог: {dialog}"),
        ]
    )

    logger.info("Compliance agent: старт, сегментов=%d", len(dialog))

    try:
        llm = get_llm(max_completion_tokens=COMPLIANCE_MAX_TOKENS)
        structured_llm = llm.with_structured_output(ComplianceResult)

        chain = prompt | structured_llm
        result = cast(ComplianceResult, chain.invoke({"dialog": dialog, "classification": classification}))

        issues = [issue.model_dump() for issue in result.issues]
        compliance = {"passed": not issues, "issues": issues}
        logger.info("Compliance agent: passed=%s, нарушений=%d", compliance["passed"], len(issues))

        return {"compliance": compliance}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to check dialog compliance: {e}") from e

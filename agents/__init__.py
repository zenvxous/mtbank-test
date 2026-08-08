from typing import Any, TypedDict

import structlog
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from api.config import settings

log = structlog.get_logger(__name__)


class AnalyzeState(TypedDict):
    transcript: list[dict[str, Any]]
    classification: dict[str, Any]
    quality_score: dict[str, Any]
    compliance: dict[str, Any]
    summary: str
    action_items: list[str]

def get_llm(max_completion_tokens: int = 2000):
    openrouter_api_key = settings.OPENROUTER_API_KEY
    if not openrouter_api_key:
        log.error("llm_misconfigured", missing="OPENROUTER_API_KEY")
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY is not set in the environment variables.")
    openrouter_model = settings.OPENROUTER_MODEL
    if not openrouter_model:
        log.error("llm_misconfigured", missing="OPENROUTER_MODEL")
        raise HTTPException(status_code=400, detail="OPENROUTER_MODEL is not set in the environment variables.")

    log.debug("llm_created", model=openrouter_model, max_completion_tokens=max_completion_tokens)

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=SecretStr(openrouter_api_key),
        model=openrouter_model,
        temperature=0.0,
        max_completion_tokens=max_completion_tokens,
    )

import time

import structlog
from langgraph.graph import END, START, StateGraph

from agents import AnalyzeState
from agents.classifier import classification_node
from agents.compliance import compliance_node
from agents.quality import quality_node
from agents.summarizer import summarization_node

log = structlog.get_logger(__name__)


def build_graph():
    workflow = StateGraph(AnalyzeState)
    workflow.add_node("classifier", classification_node)
    workflow.add_node("quality", quality_node)
    workflow.add_node("compliance", compliance_node)
    workflow.add_node("summarizer", summarization_node)

    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "quality")
    workflow.add_edge("classifier", "compliance")
    workflow.add_edge("quality", "summarizer")
    workflow.add_edge("compliance", "summarizer")
    workflow.add_edge("summarizer", END)

    return workflow.compile()

analyze_graph = build_graph()

def run_analysis(transcript: list) -> dict:
    initial_state = AnalyzeState(
        transcript=transcript,
        classification={},
        quality_score={},
        compliance={},
        summary="",
        action_items=[],
    )

    log.info("analysis_graph_started", segments=len(transcript))
    started = time.perf_counter()

    try:
        final_state = analyze_graph.invoke(initial_state)
    except Exception:
        log.error(
            "analysis_graph_failed",
            segments=len(transcript),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            exc_info=True,
        )
        raise

    log.info(
        "analysis_graph_completed",
        segments=len(transcript),
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )

    return final_state

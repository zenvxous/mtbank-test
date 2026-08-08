from langgraph.graph import END, START, StateGraph

from agents import AnalyzeState
from agents.classifier import classification_node
from agents.compliance import compliance_node
from agents.quality import quality_node
from agents.summarizer import summarization_node


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

    final_state = analyze_graph.invoke(initial_state)

    return final_state

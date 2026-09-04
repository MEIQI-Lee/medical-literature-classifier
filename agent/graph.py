from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.schemas import (
    Candidate,
    ConfidencePolicy,
    ErrorInfo,
    ModelOutput,
    ParsedDocument,
    PreparedText,
    StepInfo,
    WarningInfo,
)


@dataclass(frozen=True)
class InputRequest:
    kind: str
    payload: Any
    filename: str = ""
    top_k: Any = None
    confidence_threshold: Any = None


class WorkflowState(TypedDict, total=False):
    request_id: str
    request: InputRequest
    on_step: Callable[[StepInfo], None] | None
    steps: list[StepInfo]
    warnings: list[WarningInfo]
    error: ErrorInfo | None
    document: ParsedDocument
    prepared: PreparedText
    output: ModelOutput
    candidates: list[Candidate]
    confidence_status: str
    display_mode: str
    message: str
    top_k: int
    confidence_policy: ConfidencePolicy | None


def build_graph(nodes: dict[str, Callable]):
    """Explicit routes are easy to inspect, test, and explain in the presentation."""
    builder = StateGraph(WorkflowState)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "validate_input")

    def route_input(state: WorkflowState):
        if state.get("error"):
            return "generate_result"
        return "parse_document" if state["request"].kind == "file" else "read_text"

    builder.add_conditional_edges(
        "validate_input", route_input, ["parse_document", "read_text", "generate_result"]
    )
    for source, target in (
        ("read_text", "prepare_text"),
        ("parse_document", "prepare_text"),
        ("prepare_text", "classify_text"),
        ("classify_text", "select_candidates"),
        ("accept_prediction", "query_category"),
        ("review_candidates", "query_category"),
        ("query_category", "generate_result"),
    ):
        builder.add_conditional_edges(
            source,
            lambda state, next_node=target: "generate_result" if state.get("error") else next_node,
            list(dict.fromkeys([target, "generate_result"])),
        )

    def route_confidence(state: WorkflowState):
        if state.get("error"):
            return "generate_result"
        return (
            "accept_prediction" if state["confidence_status"] == "normal" else "review_candidates"
        )

    builder.add_conditional_edges(
        "select_candidates",
        route_confidence,
        ["accept_prediction", "review_candidates", "generate_result"],
    )
    builder.add_edge("generate_result", END)
    # Request state stays in memory for this invocation; no checkpoint database is necessary.
    return builder.compile()

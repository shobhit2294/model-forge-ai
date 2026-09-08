"""LangGraph orchestration boundary for the AI Engineering Agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class WorkflowState(TypedDict, total=False):
    request: dict[str, Any]
    profile: dict[str, Any]
    quality: dict[str, Any]
    research: dict[str, Any]
    experiments: list[dict[str, Any]]
    decision: dict[str, Any]
    deployment: dict[str, Any]
    monitoring: dict[str, Any]
    loop_count: int
    next_action: str


def intake_node(state: WorkflowState) -> WorkflowState:
    return {"loop_count": state.get("loop_count", 0), "next_action": "data"}


def data_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "research"}


def research_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "model"}


def model_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "decision"}


def decision_node(state: WorkflowState) -> WorkflowState:
    decision = state.get("decision", {})
    approved = decision.get("status") == "approved"
    loop_count = state.get("loop_count", 0)
    if approved:
        return {"next_action": "deploy", "loop_count": loop_count}
    if loop_count < 1:
        return {"next_action": "model", "loop_count": loop_count + 1}
    return {"next_action": "blocked", "loop_count": loop_count}


def deployment_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "monitor"}


def monitoring_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "complete"}


def blocked_node(state: WorkflowState) -> WorkflowState:
    return {"next_action": "complete"}


def _route_after_intake(state: WorkflowState) -> str:
    return "data"


def _route_after_decision(state: WorkflowState) -> str:
    return state.get("next_action", "blocked")


def build_workflow_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("agent_intake", intake_node)
    graph.add_node("agent_data", data_node)
    graph.add_node("agent_research", research_node)
    graph.add_node("agent_model", model_node)
    graph.add_node("agent_decision", decision_node)
    graph.add_node("agent_deploy", deployment_node)
    graph.add_node("agent_monitor", monitoring_node)
    graph.add_node("agent_blocked", blocked_node)
    graph.add_edge(START, "agent_intake")
    graph.add_conditional_edges("agent_intake", _route_after_intake, {"data": "agent_data"})
    graph.add_edge("agent_data", "agent_research")
    graph.add_edge("agent_research", "agent_model")
    graph.add_edge("agent_model", "agent_decision")
    graph.add_conditional_edges("agent_decision", _route_after_decision, {"model": "agent_model", "deploy": "agent_deploy", "blocked": "agent_blocked"})
    graph.add_edge("agent_deploy", "agent_monitor")
    graph.add_edge("agent_monitor", END)
    graph.add_edge("agent_blocked", END)
    return graph.compile()
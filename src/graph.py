from langgraph.graph import StateGraph, END
from src.agents import (
    AgentState, parser_agent, retrieve_agent, 
    solver_agent, verifier_agent, explainer_agent
)

def build_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("parser", parser_agent)
    workflow.add_node("retriever", retrieve_agent)
    workflow.add_node("solver", solver_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("explainer", explainer_agent)

    # Add Edges
    workflow.set_entry_point("parser")
    
    # Conditional edge after parser
    def check_clarification(state):
        if state["parsed_problem"].get("needs_clarification"):
            return END 
        return "retriever"

    workflow.add_conditional_edges(
        "parser",
        check_clarification,
        {END: END, "retriever": "retriever"}
    )

    workflow.add_edge("retriever", "solver")
    workflow.add_edge("solver", "verifier")

    def check_correctness(state):
        if state["is_correct"]:
            return "explainer"
        else:
            return END 

    workflow.add_conditional_edges(
        "verifier",
        check_correctness,
        {"explainer": "explainer", END: END}
    )

    workflow.add_edge("explainer", END)

    return workflow.compile()
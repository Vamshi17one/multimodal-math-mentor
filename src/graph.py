from langgraph.graph import StateGraph, END
from src.agents import (
    AgentState, parser_agent, router_agent, 
    python_solver_agent, rag_solver_agent, 
    verifier_agent, explainer_agent
)

def build_graph():
    workflow = StateGraph(AgentState)

    
    workflow.add_node("parser", parser_agent)
    workflow.add_node("router", router_agent)
    workflow.add_node("python_solver", python_solver_agent) 
    workflow.add_node("rag_solver", rag_solver_agent)       
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("explainer", explainer_agent)

    
    workflow.set_entry_point("parser")

    
    def check_clarification(state):
        if state["parsed_problem"].get("needs_clarification"):
            return END
        return "router"

    workflow.add_conditional_edges(
        "parser",
        check_clarification,
        {END: END, "router": "router"}
    )

    
    def route_workflow(state):
        if state["problem_category"] == "calculation":
            return "python_solver"
        else:
            return "rag_solver"

    workflow.add_conditional_edges(
        "router",
        route_workflow,
        {
            "python_solver": "python_solver",
            "rag_solver": "rag_solver"
        }
    )

    
    workflow.add_edge("python_solver", "verifier")
    workflow.add_edge("rag_solver", "verifier")

    
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
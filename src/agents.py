import sys
import io
import contextlib
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.rag import get_retriever

# ... [AgentState class remains the same] ...
class AgentState(TypedDict):
    raw_input: str
    input_type: str 
    parsed_problem: dict
    problem_category: Literal["calculation", "conceptual"]
    retrieved_docs: List[str]
    code_snippet: str
    code_output: str
    solution_plan: str
    final_answer: str
    is_correct: bool
    explanation: str
    messages: List[str]

# REMOVED GLOBAL LLM INSTANTIATION
# llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)

def get_llm():
    """Helper to get LLM instance with latest API key."""
    return ChatOpenAI(
        model=Config.MODEL_NAME, 
        temperature=0, 
        api_key=Config.get_openai_key()
    )

# ... [Pydantic Models remain the same] ...
class ParsedProblem(BaseModel):
    problem_text: str = Field(description="Clean math problem text")
    topic: str = Field(description="Math topic e.g., Calculus, Algebra")
    needs_clarification: bool = Field(description="True if input is ambiguous")

class RouterDecision(BaseModel):
    category: Literal["calculation", "conceptual"] = Field(
        description="Choose 'calculation' for explicit math requiring computation. Choose 'conceptual' for definitions, theorems, or proofs."
    )
    reasoning: str = Field(description="Why this category was chosen.")

class Verification(BaseModel):
    is_correct: bool = Field(description="Is the solution mathematically sound?")
    critique: str = Field(description="Specific critique of logic, units, or domains.")


def execute_python_math(code: str) -> str:
    # ... [Implementation remains the same] ...
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__, "math": __import__("math"), "numpy": __import__("numpy")})
        return output.getvalue()
    except Exception as e:
        return f"Error: {e}"


async def parser_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Math Parser. Extract the core math problem."),
        ("user", "{raw_input}")
    ])
    
    # Instantiate LLM dynamically
    llm = get_llm()
    structured_llm = llm.with_structured_output(ParsedProblem)
    
    try:
        result = await (prompt | structured_llm).ainvoke({"raw_input": state["raw_input"]})
        return {
            "parsed_problem": result.dict(), 
            "messages": [f"Parser: Extracted topic '{result.topic}'"]
        }
    except Exception as e:
        return {"parsed_problem": {"needs_clarification": True}, "messages": ["Parser: Error parsing input."]}


async def router_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Math Router. Route the problem to the correct solver."),
        ("user", "Problem: {problem}\n\nIs this a heavy calculation/algebra problem (Calculation) or a theoretical/definition question (Conceptual)?")
    ])
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouterDecision)
    
    problem = state["parsed_problem"]["problem_text"]
    decision = await (prompt | structured_llm).ainvoke({"problem": problem})
    
    return {
        "problem_category": decision.category,
        "messages": [f"Router: Routing to {decision.category} solver."]
    }


async def python_solver_agent(state: AgentState):
    problem = state["parsed_problem"]["problem_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python Math Engineer. Write valid Python code to solve the problem. Print the final answer."),
        ("user", "Problem: {problem}\n\nWrite the Python code.")
    ])
    
    llm = get_llm()
    chain = prompt | llm
    res = await chain.ainvoke({"problem": problem})
    code = res.content.replace("```python", "").replace("```", "").strip()
    
    output = execute_python_math(code)
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Format the code output into a math answer."),
        ("user", "Problem: {problem}\nCode: {code}\nOutput: {output}")
    ])
    final_res = await (final_prompt | llm).ainvoke({"problem": problem, "code": code, "output": output})
    
    return {
        "code_snippet": code,
        "code_output": output,
        "final_answer": final_res.content,
        "messages": ["Solver: Executed Python calculation."]
    }


async def rag_solver_agent(state: AgentState):
    query = state["parsed_problem"]["problem_text"]
    
    # get_retriever now handles dynamic embedding creation internally
    retriever = get_retriever()
    docs = await retriever.ainvoke(query)
    
    formatted_docs = [f"[Source: {d.metadata.get('source', 'KB')}]\n{d.page_content}" for d in docs]
    context = "\n\n".join(formatted_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Theoretical Math Tutor. Use the context to explain and solve."),
        ("user", "Context: {context}\n\nProblem: {query}")
    ])
    
    llm = get_llm()
    res = await (prompt | llm).ainvoke({"context": context, "query": query})
    
    return {
        "retrieved_docs": formatted_docs,
        "final_answer": res.content,
        "messages": [f"Solver: Used RAG with {len(docs)} sources."]
    }


async def verifier_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Strict Math Verifier. Check for calculation errors, unit mismatches, and domain violations (e.g., dividing by zero)."),
        ("user", "Problem: {problem}\nProposed Answer: {answer}\nMethod: {category}")
    ])
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(Verification)
    
    res = await (prompt | structured_llm).ainvoke({
        "problem": state["parsed_problem"]["problem_text"],
        "answer": state["final_answer"],
        "category": state.get("problem_category", "general")
    })
    
    return {
        "is_correct": res.is_correct,
        "messages": [f"Verifier: Solution is {'Correct' if res.is_correct else 'Incorrect'}."]
    }


async def explainer_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Tutor. Explain the solution step-by-step using Markdown. Do not use LaTeX."),
        ("user", "Problem: {problem}\nSolution: {solution}\n\nExplain it simply.")
    ])
    
    llm = get_llm()
    res = await (prompt | llm).ainvoke({
        "problem": state["parsed_problem"]["problem_text"],
        "solution": state["final_answer"]
    })
    return {"explanation": res.content}
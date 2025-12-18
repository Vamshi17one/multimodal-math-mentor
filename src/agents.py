from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.rag import get_retriever


class AgentState(TypedDict):
    raw_input: str
    input_type: str 
    parsed_problem: dict
    retrieved_docs: List[str]
    solution_plan: str
    final_answer: str
    is_correct: bool
    explanation: str
    messages: List[str]

llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)


class ParsedProblem(BaseModel):
    problem_text: str = Field(description="Clean math problem text")
    topic: str = Field(description="Math topic e.g., Calculus, Algebra")
    needs_clarification: bool = Field(description="True if input is ambiguous or nonsensical")

class Verification(BaseModel):
    is_correct: bool = Field(description="Is the solution mathematically sound?")
    critique: str = Field(description="Critique of the solution logic")


async def parser_agent(state: AgentState):
    """Parses raw text into structured JSON asynchronously using Structured Output."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Math Parser. Convert input into a structured format."),
        ("user", "{raw_input}")
    ])
    
    structured_llm = llm.with_structured_output(ParsedProblem)
    chain = prompt | structured_llm
    
    try:
        result = await chain.ainvoke({"raw_input": state["raw_input"]})
        return {
            "parsed_problem": result.dict(), 
            "messages": ["Parser: Successfully parsed input."]
        }
    except Exception as e:
        return {
            "parsed_problem": {"needs_clarification": True}, 
            "messages": [f"Parser: Failed to parse. Error: {str(e)}"]
        }

async def retrieve_agent(state: AgentState):
    """RAG Step: Fetches formulas/concepts asynchronously and cites sources."""
    query = state["parsed_problem"]["problem_text"]
    retriever = get_retriever()
    
    docs = await retriever.ainvoke(query)
    
    formatted_docs = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        content = d.page_content
        formatted_docs.append(f"[Source: {source}]\n{content}")
    
    return {
        "retrieved_docs": formatted_docs, 
        "messages": [f"Retrieved {len(docs)} chunks from KB."]
    }

async def solver_agent(state: AgentState):
    """Solves the problem using context."""
    
    context = "\n\n".join(state["retrieved_docs"])
    problem = state["parsed_problem"]["problem_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a JEE Math Tutor. Solve the problem step-by-step using the context provided."),
        ("user", "Context (use this to guide your solution):\n{context}\n\nProblem: {problem}\n\nProvide a solution plan and the final answer.")
    ])
    chain = prompt | llm
    
    response = await chain.ainvoke({"context": context, "problem": problem})
    return {"final_answer": response.content, "messages": ["Solver: Generated solution."]}

async def verifier_agent(state: AgentState):
    """Critique the solution using Structured Output."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Math Professor. Verify the student's solution."),
        ("user", "Problem: {problem}\nProposed Solution: {solution}")
    ])
    
    structured_llm = llm.with_structured_output(Verification)
    chain = prompt | structured_llm
    
    problem = state["parsed_problem"]["problem_text"]
    solution = state["final_answer"]
    
    result = await chain.ainvoke({
        "problem": problem, 
        "solution": solution
    })
    
    return {"is_correct": result.is_correct, "messages": [f"Verifier: Correctness = {result.is_correct}"]}

async def explainer_agent(state: AgentState):
    """Formats the final output for the student with LaTeX support."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        <Role>
        You are a helpful tutor. Explain the solution clearly to a student.
        </Role>
        <formatting_rules>
            <general_format>
                <allowed>Markdown with LaTeX.</allowed>
            </general_format>
            <math_notation>
                <directive>Use LaTeX for ALL mathematical expressions.</directive>
                <inline_math>Wrap inline equations in single dollar signs, e.g., $x^2$.</inline_math>
                <block_math>Wrap main equations in double dollar signs, e.g., $$ \\frac{{a}}{{b}} $$.</block_math>
                <examples>
                    <correct>The area is calculated as $$ A = \pi r^2 $$.</correct>
                    <correct>Since $x > 0$, we proceed.</correct>
                </examples>
            </math_notation>
        </formatting_rules>"""),
        ("user", "Solution: {solution}\n\nExplain this step-by-step with nice formatting.")
    ])
    chain = prompt | llm
    
    res = await chain.ainvoke({"solution": state["final_answer"]})
    return {"explanation": res.content}
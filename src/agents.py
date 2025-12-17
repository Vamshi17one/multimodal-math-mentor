from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import Config
from src.rag import get_retriever

# --- 1. State Definition ---
class AgentState(TypedDict):
    raw_input: str
    input_type: str  # text, image, audio
    parsed_problem: dict
    retrieved_docs: List[str]
    solution_plan: str
    final_answer: str
    is_correct: bool
    explanation: str
    messages: List[str] # Trace log

llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)

# --- 2. Pydantic Models for Structured Output ---

class ParsedProblem(BaseModel):
    problem_text: str = Field(description="Clean math problem text")
    topic: str = Field(description="Math topic e.g., Calculus, Algebra")
    needs_clarification: bool = Field(description="True if input is ambiguous")

class Verification(BaseModel):
    is_correct: bool = Field(description="Is the solution mathematically sound?")
    critique: str = Field(description="Critique of the solution logic")

# --- 3. Agents ---

def parser_agent(state: AgentState):
    """Parses raw text into structured JSON."""
    parser = PydanticOutputParser(pydantic_object=ParsedProblem)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Math Parser. Convert input into a structured format.\n{format_instructions}"),
        ("user", "{raw_input}")
    ])
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"raw_input": state["raw_input"], "format_instructions": parser.get_format_instructions()})
        return {"parsed_problem": result.dict(), "messages": ["Parser: Successfully parsed input."]}
    except:
        # Fallback if parsing fails
        return {"parsed_problem": {"needs_clarification": True}, "messages": ["Parser: Failed to parse."]}

def retrieve_agent(state: AgentState):
    """RAG Step: Fetches formulas/concepts."""
    query = state["parsed_problem"]["problem_text"]
    retriever = get_retriever()
    docs = retriever.invoke(query)
    doc_texts = [d.page_content for d in docs]
    return {"retrieved_docs": doc_texts, "messages": [f"Retrieved {len(docs)} documents."]}

def solver_agent(state: AgentState):
    """Solves the problem using context."""
    context = "\n".join(state["retrieved_docs"])
    problem = state["parsed_problem"]["problem_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a JEE Math Tutor. Solve the problem step-by-step using the context provided."),
        ("user", "Context: {context}\n\nProblem: {problem}\n\nProvide a solution plan and the final answer.")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "problem": problem})
    return {"final_answer": response.content, "messages": ["Solver: Generated solution."]}

def verifier_agent(state: AgentState):
    """Critique the solution."""
    parser = PydanticOutputParser(pydantic_object=Verification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Senior Math Professor. Verify the student's solution.\n{format_instructions}"),
        ("user", "Problem: {problem}\nProposed Solution: {solution}")
    ])
    chain = prompt | llm | parser
    
    problem = state["parsed_problem"]["problem_text"]
    solution = state["final_answer"]
    
    result = chain.invoke({
        "problem": problem, 
        "solution": solution,
        "format_instructions": parser.get_format_instructions()
    })
    
    return {"is_correct": result.is_correct, "messages": [f"Verifier: Correctness = {result.is_correct}"]}

def explainer_agent(state: AgentState):
    """Formats the final output for the student."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        <Role>
        You are a helpful tutor. Explain the solution clearly to a student.
        </Role>
        <formatting_rules>
            <general_format>
                <allowed>Pure Markdown only.</allowed>
                <forbidden>HTML tags (e.g., &lt;div&gt;, &lt;br&gt;).</forbidden>
            </general_format>
                <mathematical_formulae>
                    Identify and verify standard trigonometric identities and equations. Provide only correct and well-known formulas in plain text without any LaTeX or special rendering. If the formula is incorrect or improperly formatted, provide corrected versions.
                </mathematical_formulae>

                <math_notation>
                    <directive>ABSOLUTELY NO LaTeX or special rendering code.</directive>
                    <forbidden_symbols>$, $$, \, \frac, \sqrt, \times, \sin</forbidden_symbols>
                    <required_style>Plain text readable by humans.</required_style>
                    <required_usage>use π symbol for pi.</required_usage>
                    <examples>
                        <correct>3x + 5 = 0</correct>
                        <correct>1/2</correct>
                        <correct>x^2</correct>
                        <correct>π</correct>
                    </examples>
                </math_notation>
        </formatting_rules>"""),
        ("user", "Solution: {solution}\n\nExplain this simply.")
    ])
    chain = prompt | llm
    res = chain.invoke({"solution": state["final_answer"]})
    return {"explanation": res.content}
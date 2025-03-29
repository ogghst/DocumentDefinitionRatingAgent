import traceback
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document

from models import CheckItem, CheckResult
from llm_setup import initialize_llm, analysis_parser

# Global LLM instance
llm = initialize_llm()

# Prompt templates
analysis_prompt_template = """
You are a meticulous Project Compliance Analyst. Your task is to analyze the provided Document Context to determine if a specific Checklist Item is met for a given project phase.

**Checklist Item Details:**
- ID: {check_id}
- Name: {check_name}
- Description: {check_description}
- Phase: {check_phase}

**Document Context:**
--- START CONTEXT ---
{context}
--- END CONTEXT ---

{additional_info}

**Analysis Instructions:**
1.  **Understand the Goal:** Read the Checklist Item Description carefully. What specific condition needs to be confirmed?
2.  **Examine Context:** Search the Document Context for explicit statements or strong evidence directly related to the checklist item's condition.
3.  **Determine Status (`is_met`):**
    *   If the context clearly and unambiguously confirms the condition is met, set `is_met` to `true`.
    *   If the context clearly confirms the condition is *not* met, or if the context lacks any relevant information or is too vague to make a determination, set `is_met` to `false`.
4.  **Assess Reliability (`reliability`):** Provide a confidence score (0-100) based *only* on the provided context:
    *   90-100: Direct, explicit confirmation/denial in the context. No ambiguity.
    *   70-89: Strong evidence suggesting confirmation/denial, but requires minimal interpretation.
    *   50-69: Context provides related information, but it's indirect or requires significant interpretation. Plausible but not certain.
    *   0-49: Context is irrelevant, contradictory, very vague, or completely missing information about the check item.
5.  **Extract Sources (`sources`):** Quote the *exact* sentences or short passages (max 2-3 relevant sentences per source) from the Document Context that are the primary evidence for your `is_met` decision and `reliability` score. If no specific sentences provide direct evidence, provide an empty list `[]`.
6.  **Explain Reasoning (`analysis_details`):** Briefly explain *why* you reached the `is_met` conclusion and `reliability` score, referencing the evidence (or lack thereof) in the context.

**IMPORTANT: Do NOT include the check_item field in your response. Only provide is_met, reliability, sources, and analysis_details.**

**Output Format:**
{format_instructions}
"""

question_prompt_template = """
You are a meticulous Project Compliance Analyst. You're reviewing a checklist item but need additional information to make a confident determination.

**Checklist Item Details:**
- ID: {check_id}
- Name: {check_name}
- Description: {check_description}
- Phase: {check_phase}

**Current Analysis Status:**
- Current reliability score: {current_reliability}%
- Current determination: {is_met_status}
- Current reasoning: {analysis_details}

**Document Evidence:**
{sources_summary}

Based on the above information, identify 1-3 specific questions that would help clarify whether this checklist item is met or not. Focus on questions that:
1. Target the precise information gaps in the document
2. Would significantly increase the reliability of your assessment
3. Are directly relevant to determining if the checklist item is satisfied

Format your response as a numbered list of questions only. Do not provide any additional text or explanations.
"""

def format_docs(docs: List[Document]) -> str:
    """Helper to join document contents."""
    return "\n\n".join(f"Source Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

def create_rag_chain():
    """Create and return the RAG chain for analyzing checks."""
    analysis_prompt = PromptTemplate(
        template=analysis_prompt_template,
        input_variables=["check_id", "check_name", "check_description", "check_phase", "context", "additional_info"],
        partial_variables={"format_instructions": analysis_parser.get_format_instructions()}
    )

    print(f"Prompt template: {analysis_prompt.template}")
    print(f"Input variables: {analysis_prompt.input_variables}")
    print(f"Partial variables: {analysis_prompt.partial_variables}")    
    print(f"Format instructions: {analysis_parser.get_format_instructions()}")
    
    # This chain preserves streaming capabilities through the entire chain
    # The streaming happens via the callback manager that will be provided in the RunnableConfig
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["retriever"].invoke(x["check_description"]))
        )
        | analysis_prompt
        | llm  # streaming=True is already set in the llm instance
        | analysis_parser
    )
    
    return rag_chain

def create_streaming_chain():
    """Create a streaming-only chain for token visibility."""
    analysis_prompt = PromptTemplate(
        template=analysis_prompt_template,
        input_variables=["check_id", "check_name", "check_description", "check_phase", "context", "additional_info"],
        partial_variables={"format_instructions": analysis_parser.get_format_instructions()}
    )
    
    # This chain only includes prompt + LLM without parsing
    # Streaming happens via the callback manager provided in the RunnableConfig
    streaming_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["retriever"].invoke(x["check_description"]))
        )
        | analysis_prompt
        | llm  # streaming=True is already set in the llm instance
    )
    
    return streaming_chain

def generate_questions(check_item: CheckItem, result: CheckResult) -> str:
    """Generate follow-up questions to improve reliability."""
    question_prompt = PromptTemplate(
        template=question_prompt_template,
        input_variables=[
            "check_id", "check_name", "check_description", "check_phase", 
            "current_reliability", "is_met_status", "analysis_details", 
            "sources_summary"
        ]
    )
    
    sources_summary = "No relevant sources found in the document." if not result.sources else "\n".join(result.sources)
    is_met_status = "Undetermined" if result.is_met is None else ("Met" if result.is_met else "Not Met")
    
    question_input = {
        "check_id": check_item.id,
        "check_name": check_item.name,
        "check_description": check_item.description,
        "check_phase": check_item.phase,
        "current_reliability": result.reliability,
        "is_met_status": is_met_status,
        "analysis_details": result.analysis_details,
        "sources_summary": sources_summary
    }
    
    return llm.invoke(question_prompt.format(**question_input)).content 
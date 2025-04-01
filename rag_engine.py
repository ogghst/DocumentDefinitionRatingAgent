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
You are a meticulous Project Compliance Analyst. Your task is to analyze the provided Document Context to determine if a specific Checklist Item is met.

**Checklist Item Details:**
- ID: {check_id}
- Name: {check_name}
- Description: {check_description}

**Document Context:**
--- START CONTEXT ---
{context}
--- END CONTEXT ---

{additional_info}

**Analysis Instructions:**
1. First, carefully understand the checklist item name and description. This is what you need to verify from the document.
2. Search the Document Context for evidence related to this specific checklist item.
3. Determine if the checklist item is met based ONLY on the evidence in the document:
   - If there is clear evidence confirming the item is met, set is_met to true
   - If there is evidence the item is NOT met or there's insufficient evidence, set is_met to false
4. Rate your confidence in your determination with a reliability score (0-100):
   - 90-100: Direct, explicit confirmation in the document
   - 70-89: Strong evidence but requires some interpretation
   - 50-69: Related information that suggests an answer but isn't conclusive
   - 0-49: Very little or no relevant information in the document
5. Include direct quotes from the document that support your determination as sources.
6. Explain your reasoning in the analysis_details.

Respond with ONLY the following fields:
1. is_met (true/false): Whether the checklist item requirements are met based on the document
2. reliability (0-100): Your confidence in the determination
3. sources (list of strings): Direct quotes from the document that support your determination
4. analysis_details (string): Your reasoning for the determination

Output JSON format:
{format_instructions}
"""

question_prompt_template = """
You are analyzing a checklist item but need additional information to make a confident determination.

**Checklist Item:**
ID: {check_id}
Name: {check_name}
Description: {check_description}

**Current Status:**
- Reliability: {current_reliability}%
- Current determination: {is_met_status}
- Current reasoning: {analysis_details}

**Available Evidence:**
{sources_summary}

Generate 1-3 specific questions that would help determine if this checklist item is met.
Focus on questions that would address gaps in the available information.
Format your response as a numbered list of questions ONLY - no additional text.
"""

def format_docs(docs: List[Document]) -> str:
    """Helper to join document contents."""
    return "\n\n".join(f"Source Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

def create_rag_chain():
    """Create and return the RAG chain for analyzing checks."""
    # Create a custom format instruction that's simpler
    simple_format_instructions = """
{
  "is_met": bool,  // true if the check item is met, false otherwise
  "reliability": int,  // 0-100 confidence score
  "sources": [string],  // List of quotes from the document that support your determination
  "analysis_details": string  // Your reasoning
}
"""
    
    analysis_prompt = PromptTemplate(
        template=analysis_prompt_template,
        input_variables=["check_id", "check_name", "check_description", "context", "additional_info"],
        partial_variables={"format_instructions": simple_format_instructions}
    )
    
    # Enable this for debugging:
    # print(f"\n* * * Analysis Prompt Template:\n{analysis_prompt.template}")
    # print(f"\n* * * Input Variables: {analysis_prompt.input_variables}")
    # print(f"\n* * * Partial Variables: {analysis_prompt.partial_variables}\n\n")
    
    # This chain preserves streaming capabilities through the entire chain
    # The streaming happens via the callback manager that will be provided in the RunnableConfig
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["retriever"].invoke(x["check_description"]))
        )
        | analysis_prompt
        | llm  # Don't use with_config here, let the config be passed in directly
        | analysis_parser
    )
    
    return rag_chain


def create_hybrid_rag_chain():
    """Create a RAG chain that provides both streaming and structured output."""
    # Create a custom format instruction that's simpler
    simple_format_instructions = """
{
  "is_met": bool,  // true if the check item is met, false otherwise
  "reliability": int,  // 0-100 confidence score
  "sources": [string],  // List of quotes from the document that support your determination
  "analysis_details": string  // Your reasoning
}
"""

    analysis_prompt = PromptTemplate(
        template=analysis_prompt_template,
        input_variables=["check_id", "check_name", "check_description", "context", "additional_info"],
        partial_variables={"format_instructions": simple_format_instructions}
    )
    
    # Base retrieval and prompt formatting
    base_chain = RunnablePassthrough.assign(
        context=lambda x: format_docs(x["retriever"].invoke(x["check_description"]))
    ) | analysis_prompt
    
    # The LLM step will stream tokens through callbacks
    llm_output = base_chain | llm
    
    # Create a chain that returns both raw output and parsed result
    def process_llm_output(llm_response):
        # For LangChain's ChatModels that return a message, extract content
        content = llm_response.content if hasattr(llm_response, "content") else llm_response
        
        # Parse the output into a structured format
        try:
            parsed_result = analysis_parser.parse(content)
            return {
                "raw_output": content,
                "parsed_result": parsed_result
            }
        except Exception as e:
            # Return the error while preserving the raw output
            return {
                "raw_output": content,
                "parsing_error": str(e)
            }
    
    # Final chain combining streaming and parsing
    final_chain = llm_output | process_llm_output
    
    return final_chain

def generate_questions(check_item: CheckItem, result: CheckResult, callback_manager=None) -> str:
    """Generate specific follow-up questions to improve reliability."""
    question_prompt = PromptTemplate(
        template=question_prompt_template,
        input_variables=[
            "check_id", "check_name", "check_description", 
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
        "current_reliability": result.reliability,
        "is_met_status": is_met_status,
        "analysis_details": result.analysis_details,
        "sources_summary": sources_summary
    }
    
    # Set up config to include the callback_manager for streaming if provided
    config = {}
    if callback_manager:
        try:
            config["callbacks"] = [callback_manager]
        except Exception as e:
            print(f"Warning: Failed to set up callback manager for questions: {e}")
    
    # Directly use LLM for question generation with streaming
    formatted_prompt = question_prompt.format(**question_input)
    return llm.invoke(formatted_prompt, config=config).content 
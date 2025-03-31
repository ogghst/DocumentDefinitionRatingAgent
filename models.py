from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated, TypedDict

class CheckItem(BaseModel):
    """Represents a single checklist item for a specific phase."""
    id: int = Field(..., description="Unique identifier for the check.")
    name: Optional[str] = Field(..., description="Short name or title of the check.")
    branch_id: Optional[int] = Field(..., description="Identifier for the check's branch/topic.")
    branch_name: Optional[str] = Field(..., description="Name of the check's branch/topic.")
    description: Optional[str] = Field(..., description="Detailed description of the check to be performed.")
    weight: Optional[int] = Field(..., description="Weight or importance of the check for this phase.")
    phase: Optional[str] = Field(..., description="The project phase this check instance belongs to.")

class CheckResult(BaseModel):
    """Represents the result of analyzing a check item."""
    check_item: Annotated[CheckItem, "The check item this result refers to"] = None
    is_met: Annotated[Optional[bool], "Whether the check is met (true) or not (false)"] = False
    reliability: Annotated[int, "Reliability score of the result (0-100). 0 is the lowest (not reliable), 100 is the highest (very reliable)."] = 0
    sources: Annotated[List[str], "Sources of evidence from the document"] = []
    analysis_details: Annotated[str, "Detailed analysis of the check"] = None
    needs_human_review: Annotated[bool, "Whether human review is needed (true) or not (false)"] = False
    user_input: Annotated[Optional[str], "User input provided during review"] = None

    @field_validator('reliability')
    def check_reliability_range(cls, v):
        if not (0 <= v <= 100):
            # Clamp the value instead of raising an error, as LLMs might slightly exceed bounds
            print(f"Warning: Clamping reliability score {v} to be within [0, 100].")
            return max(0.0, min(100.0, v))
        return v

# Add GraphState here rather than in graph_nodes.py
class GraphState(TypedDict):
    """Represents the state passed between nodes in the LangGraph workflow."""
    # Inputs
    checklist_path: Annotated[str, "Path to the Excel checklist file."]
    document_path: Annotated[str, "Path to the Word quotation document."]
    target_phase: Annotated[str, "The specific project phase to analyze checks for."]

    # Workflow context
    conversation_id: Annotated[Optional[str], "ID of the conversation for websocket communication."] = None
    callback_manager: Annotated[Optional[Any], "Callback manager for websocket communication."] = None

    # Processed data
    checks_for_phase: Annotated[Optional[List[CheckItem]], "List of CheckItem objects relevant to the target_phase."] = None
    retriever: Annotated[Optional[Any], "Retriever built from the document chunks."] = None

    # Results accumulation
    analysis_results: Annotated[Optional[List[CheckResult]], "List to accumulate analysis results for each check."] = None
    final_results: Annotated[Optional[List[Dict]], "Final list of results as dictionaries for JSON output."] = None
    error_message: Annotated[Optional[str], "To capture any errors during processing."] = None 
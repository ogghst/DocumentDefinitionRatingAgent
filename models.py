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
    """Represents the analysis result for a single check item based on document review."""
    check_item: Optional[CheckItem] = Field(None, description="The original checklist item being analyzed. Can be None if parsing fails early.")
    #check_item_id: int = Field(..., description="Unique identifier for the check.")   
    is_met: Optional[bool] = Field(None, description="True if the document confirms the check is met, False if it's not met or evidence is missing/unclear.")
    reliability: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100) that the 'is_met' field is correct based *only* on the provided document context. High score requires direct, unambiguous evidence.")
    sources: List[str] = Field(default_factory=list, description="Specific sentences or passages from the document context that directly support the 'is_met' conclusion and reliability score. Should be exact quotes.")
    analysis_details: str = Field("", description="Brief LLM reasoning or explanation for the conclusion and reliability score, referencing the context.")
    needs_human_review: bool = Field(False, description="True if reliability score is below 50%, indicating need for manual verification.")

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

    # Processed data
    checks_for_phase: Annotated[Optional[List[CheckItem]], "List of CheckItem objects relevant to the target_phase."] = None
    retriever: Annotated[Optional[Any], "Retriever built from the document chunks."] = None

    # Results accumulation
    analysis_results: Annotated[Optional[List[CheckResult]], "List to accumulate analysis results for each check."] = None
    final_results: Annotated[Optional[List[Dict]], "Final list of results as dictionaries for JSON output."] = None
    error_message: Annotated[Optional[str], "To capture any errors during processing."] = None 
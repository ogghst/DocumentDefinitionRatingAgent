import os
import pandas as pd
import json
import traceback
import tiktoken # For token counting
from dotenv import load_dotenv
from typing import List, Dict, Optional, TypedDict, Any

# Langchain & Langgraph Imports
from langchain_community.document_loaders import Docx2txtLoader
# Embeddings/VectorStore no longer strictly needed for core CAG logic
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI # Use ChatOpenAI configured for DeepSeek endpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, OutputParserException
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Pydantic Models for Structure
from pydantic import BaseModel, Field, validator, ValidationError
from typing_extensions import Annotated

# --- Environment Variable Loading ---
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL") # Not needed for CAG core

# --- Basic Validation ---
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
if not DEEPSEEK_API_BASE:
    raise ValueError("DEEPSEEK_API_BASE environment variable not set.")
# if not OLLAMA_EMBED_MODEL: # No longer strictly required
#     print("Warning: OLLAMA_EMBED_MODEL environment variable not set, but not strictly needed for CAG.")

print(f"Using DeepSeek LLM via endpoint: '{DEEPSEEK_API_BASE}'")

# --- Pydantic Models (Identical to RAG version) ---

class CheckItem(BaseModel):
    """Represents a single checklist item for a specific phase."""
    id: int = Field(..., description="Unique identifier for the check.")
    name: str = Field(..., description="Short name or title of the check.")
    branch_id: int = Field(..., description="Identifier for the check's branch/topic.")
    branch_name: str = Field(..., description="Name of the check's branch/topic.")
    description: str = Field(..., description="Detailed description of the check to be performed.")
    weight: int = Field(..., description="Weight or importance of the check for this phase.")
    phase: str = Field(..., description="The project phase this check instance belongs to.")

class CheckResult(BaseModel):
    """Represents the analysis result for a single check item based on document review."""
    check_item: Optional[CheckItem] = Field(None, description="The original checklist item being analyzed. Can be None if parsing fails early.")
    is_met: Optional[bool] = Field(None, description="True if the document confirms the check is met, False if it's not met or evidence is missing/unclear.")
    reliability: float = Field(..., ge=0.0, le=100.0, description="Confidence score (0-100) that the 'is_met' field is correct based *only* on the provided document context. High score requires direct, unambiguous evidence.")
    sources: List[str] = Field(default_factory=list, description="Quote the most relevant sentence(s) or short paragraph(s) from the document that support the conclusion. If evidence is spread out, describe its location (e.g., 'Section 3, paragraph 2').")
    analysis_details: str = Field("", description="Brief LLM reasoning or explanation for the conclusion and reliability score, referencing the context.")
    needs_human_review: bool = Field(False, description="True if reliability score is below 50%, indicating need for manual verification.")

    @validator('reliability')
    def check_reliability_range(cls, v):
        if not (0 <= v <= 100):
            print(f"Warning: Clamping reliability score {v} to be within [0, 100].")
            return max(0.0, min(100.0, v))
        return v

# --- LangGraph State Definition (Modified) ---

class GraphState(TypedDict):
    """Represents the state passed between nodes in the LangGraph workflow (CAG Version)."""
    # Inputs
    checklist_path: Annotated[str, "Path to the Excel checklist file."]
    document_path: Annotated[str, "Path to the Word quotation document."]
    target_phase: Annotated[str, "The specific project phase to analyze checks for."]

    # Processed data
    checks_for_phase: Annotated[Optional[List[CheckItem]], "List of CheckItem objects relevant to the target_phase."] = None
    # Store the full document content instead of a retriever
    document_content: Annotated[Optional[str], "Full content of the document, possibly truncated if exceeding context limit."] = None
    document_token_count: Annotated[Optional[int], "Number of tokens in the loaded document content."] = None
    document_truncated: Annotated[bool, "Flag indicating if the document content was truncated."] = False

    # Results accumulation
    analysis_results: Annotated[Optional[List[CheckResult]], "List to accumulate analysis results for each check."] = None
    final_results: Annotated[Optional[List[Dict]], "Final list of results as dictionaries for JSON output."] = None
    error_message: Annotated[Optional[str], "To capture any errors during processing."] = None

# --- LLM and Parser Initialization ---

# Context Window Limit (Estimate for DeepSeek - adjust if known more precisely)
# DeepSeek models often have large context windows (e.g., 32k, 128k+). Let's use a conservative large value.
# Check DeepSeek documentation for the specific model you use.
# Using 100k as a safety margin below a potential 128k limit.
MAX_CONTEXT_TOKENS = 100000

try:
    # Initialize ChatOpenAI client configured for DeepSeek
    llm = ChatOpenAI(
        model="deepseek-chat", # Or "deepseek-coder"
        temperature=0.0,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        max_tokens=1024, # Max tokens for the *response*, not context window
    )
    print("DeepSeek LLM client initialized successfully.")

    # Initialize tokenizer for context window checking (using tiktoken as a proxy)
    # Use encoding appropriate for models like GPT-4, often similar for others
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    print("Tokenizer (tiktoken for gpt-4) initialized for token counting.")

except Exception as e:
    print(f"ERROR initializing LangChain components: {e}")
    raise

# Initialize the output parser
analysis_parser = PydanticOutputParser(pydantic_object=CheckResult)

# --- Node Functions (Modified for CAG) ---

def load_and_filter_checklist(state: GraphState) -> GraphState:
    """Loads checklist from Excel and filters checks for the target phase. (Identical to RAG version)."""
    print(f"\n--- Node: load_and_filter_checklist ---")
    checklist_path = state['checklist_path']
    target_phase = state['target_phase']
    print(f"Loading Checklist: '{checklist_path}' for Phase: '{target_phase}'")
    try:
        if not os.path.exists(checklist_path):
            raise FileNotFoundError(f"Checklist file not found at: {checklist_path}")

        df = pd.read_excel(checklist_path)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        required_cols = ['id', 'name', 'branchid', 'branchname', 'chk_description', 'weight', 'phase']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Checklist missing required columns (standardized names): {missing}")

        filtered_df = df[df['phase'].astype(str).str.strip().str.lower() == target_phase.strip().str.lower()]
        checks = []
        for _, row in filtered_df.iterrows():
             try:
                 checks.append(CheckItem(
                     id=int(row['id']), name=str(row['name']), branch_id=int(row['branchid']),
                     branch_name=str(row['branchname']), description=str(row['chk_description']),
                     weight=int(row['weight']), phase=str(row['phase'])
                 ))
             except (ValueError, TypeError) as ve:
                 print(f"Warning: Skipping row due to type conversion error: {ve} - Row data: {row.to_dict()}")
                 continue

        print(f"Found {len(checks)} checks for phase '{target_phase}'.")
        state["checks_for_phase"] = checks
        state["error_message"] = None
    except Exception as e:
        print(f"ERROR loading checklist: {e}")
        traceback.print_exc()
        state["error_message"] = f"Failed to load/filter checklist: {e}"
        state["checks_for_phase"] = []
    return state

def load_and_prepare_document(state: GraphState) -> GraphState:
    """Loads the Word document content and prepares it for the context window."""
    print(f"\n--- Node: load_and_prepare_document ---")
    if state.get("error_message"):
        print("Skipping due to previous error.")
        return state

    document_path = state['document_path']
    print(f"Loading Document: '{document_path}'")
    try:
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document file not found at: {document_path}")

        loader = Docx2txtLoader(document_path)
        docs = loader.load() # Loads as a list of Document objects (usually one)

        if not docs or not docs[0].page_content.strip():
             raise ValueError("Document is empty or could not be loaded/parsed.")

        # Combine content from all parts if loader splits (usually doesn't for Docx2txt)
        full_content = "\n\n".join([doc.page_content for doc in docs])

        # Count tokens
        tokens = tokenizer.encode(full_content)
        num_tokens = len(tokens)
        state["document_token_count"] = num_tokens
        state["document_truncated"] = False
        print(f"Document loaded. Original token count: {num_tokens}")

        # Check against context window limit and truncate if necessary
        if num_tokens > MAX_CONTEXT_TOKENS:
            print(f"WARNING: Document token count ({num_tokens}) exceeds limit ({MAX_CONTEXT_TOKENS}). Truncating content.")
            # Truncate tokens and decode back to string
            truncated_tokens = tokens[:MAX_CONTEXT_TOKENS]
            full_content = tokenizer.decode(truncated_tokens)
            state["document_truncated"] = True
            print(f"Document content truncated to approx. {MAX_CONTEXT_TOKENS} tokens.")

        state["document_content"] = full_content
        print("Document content prepared for LLM context.")
        state["error_message"] = None
    except Exception as e:
        print(f"ERROR loading/preparing document: {e}")
        traceback.print_exc()
        state["error_message"] = f"Failed to load or prepare document: {e}"
        state["document_content"] = None
        state["document_token_count"] = None
    return state

def analyze_check_cag(check_item: CheckItem, config: RunnableConfig) -> CheckResult:
    """
    Analyzes a single check item using CAG against the full document content.
    """
    print(f"\n--- Analyzing Check ID: {check_item.id} ({check_item.name}) ---")
    state: GraphState = config['configurable']
    document_content = state.get('document_content')

    # --- Pre-computation Checks ---
    if document_content is None: # Check if content is None or empty string
        print(f"ERROR: Document content not available for check ID {check_item.id}. Skipping analysis.")
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details="Analysis skipped: Document content not loaded or prepared.",
            needs_human_review=True
        )
    if state.get("document_truncated"):
        print("Note: Analyzing based on truncated document content.")

    # --- Define Prompt Template (Adjusted for CAG) ---
    prompt_template = """
    You are a meticulous Project Compliance Analyst. Your task is to analyze the **entire provided Document Content** to determine if a specific Checklist Item is met for a given project phase.

    **Checklist Item Details:**
    - ID: {check_id}
    - Name: {check_name}
    - Description: {check_description}
    - Phase: {check_phase}

    **Full Document Content:**
    --- START DOCUMENT ---
    {context}
    --- END DOCUMENT ---
    {truncation_warning}

    **Analysis Instructions:**
    1.  **Understand the Goal:** Read the Checklist Item Description carefully. What specific condition needs to be confirmed?
    2.  **Examine Full Document:** Search the **entire** Document Content provided above for explicit statements or strong evidence related to the checklist item's condition. Synthesize information if necessary.
    3.  **Determine Status (`is_met`):**
        *   If the document clearly and unambiguously confirms the condition is met, set `is_met` to `true`.
        *   If the document clearly confirms the condition is *not* met, or if the document lacks any relevant information or is too vague/contradictory, set `is_met` to `false`.
    4.  **Assess Reliability (`reliability`):** Provide a confidence score (0-100) based *only* on the provided document content:
        *   90-100: Direct, explicit confirmation/denial found. No ambiguity.
        *   70-89: Strong evidence suggesting confirmation/denial, perhaps requiring combining info from a couple of places.
        *   50-69: Related information found, but it's indirect, requires significant interpretation, or is slightly ambiguous. Plausible but not certain.
        *   0-49: Document content is irrelevant, contradictory, very vague, or completely missing information about the check item.
    5.  **Cite Evidence (`sources`):**
        *   Quote the *most relevant sentence(s) or short paragraph(s)* from the Document Content that are the primary evidence for your decision. List them as 'sources'.
        *   If the evidence is spread out or indirect, *briefly describe where it is found* (e.g., 'Mentioned in Section 3, paragraph 2' or 'Implied by combining statements in Layout and Compliance sections').
        *   If no specific evidence exists, provide an empty list `[]`.
    6.  **Explain Reasoning (`analysis_details`):** Briefly explain *why* you reached the `is_met` conclusion and `reliability` score, referencing the evidence (or lack thereof) in the document.

    **Output Format:**
    Return *only* a valid JSON object matching the following structure. Do not add any text before or after the JSON object.
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["check_id", "check_name", "check_description", "check_phase", "context", "truncation_warning"],
        partial_variables={"format_instructions": analysis_parser.get_format_instructions()}
    )

    # --- Define CAG Chain ---
    # Simplified chain: Pass check details, add full context from state, format prompt, call LLM, parse.
    cag_chain = (
        RunnablePassthrough.assign(
            # Add document content and truncation warning from state
            context=lambda x, config: config['configurable'].get('document_content', ''),
            truncation_warning=lambda x, config: "(Note: Document content was truncated due to length)" if config['configurable'].get('document_truncated') else ""
        )
        | prompt
        | llm
        | analysis_parser
    )

    # --- Invoke Chain and Handle Results ---
    result: Optional[CheckResult] = None
    try:
        # Prepare input dictionary (check item details)
        chain_input = {
            "check_id": check_item.id,
            "check_name": check_item.name,
            "check_description": check_item.description,
            "check_phase": check_item.phase,
        }
        # Invoke the chain, passing the main config which contains the state
        result = cag_chain.invoke(chain_input, config=config)

        result.check_item = check_item # Ensure check item is attached

        # Apply human review logic
        if result.reliability < 50:
            print(f"-> Check ID {check_item.id}: Reliability {result.reliability:.1f}% < 50%. Flagging for human review.")
            result.needs_human_review = True
        else:
            print(f"-> Check ID {check_item.id}: Reliability {result.reliability:.1f}% >= 50%. Looks OK.")
            result.needs_human_review = False

        return result

    except OutputParserException as ope:
        print(f"ERROR parsing LLM output for check ID {check_item.id}: {ope}")
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details=f"Analysis failed: LLM output parsing error. Raw output might be: {ope.llm_output}",
            needs_human_review=True
        )
    except Exception as e:
        print(f"ERROR during CAG analysis for check ID {check_item.id}: {e}")
        traceback.print_exc()
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details=f"Analysis failed due to runtime error: {e}",
            needs_human_review=True
        )

def format_final_output(state: GraphState) -> GraphState:
    """Formats the results list into the final JSON structure and handles errors. (Identical to RAG version)."""
    print(f"\n--- Node: format_final_output ---")

    if error_message := state.get("error_message"):
        print(f"Workflow finished with error: {error_message}")
        state["final_results"] = [{"error": error_message}]
        return state

    analysis_results = state.get("analysis_results")

    if analysis_results is None:
         if not state.get("checks_for_phase"):
             message = f"No checklist items found for the specified phase '{state['target_phase']}'. No analysis performed."
             print(message)
             state["final_results"] = [{"message": message}]
         else:
             print("Warning: Analysis results list is missing, but checks existed.")
             state["final_results"] = [{"warning": "Analysis results are unexpectedly missing."}]
         return state

    if not isinstance(analysis_results, list):
        print(f"Warning: 'analysis_results' is not a list (type: {type(analysis_results)}). Attempting to format anyway.")
        state["final_results"] = [{"error": "Internal state error: analysis_results is not a list."}]
        return state

    output_list = []
    valid_results_count = 0
    human_review_count = 0
    for i, result in enumerate(analysis_results):
        if isinstance(result, CheckResult):
            output_list.append(result.dict(exclude_none=True))
            valid_results_count += 1
            if result.needs_human_review:
                human_review_count += 1
        else:
            print(f"Warning: Item at index {i} in analysis_results is not a CheckResult object: {result}")
            output_list.append({"error": "Invalid result object received from analysis step.", "details": str(result)})

    state["final_results"] = output_list
    print(f"Formatted {valid_results_count} valid results.")

    if valid_results_count > 0:
        if human_review_count > 0:
            print(f"SUMMARY: {human_review_count} out of {valid_results_count} analyzed checks require human review (Reliability < 50%).")
        else:
            print(f"SUMMARY: All {valid_results_count} analyzed checks met the reliability threshold (>= 50%).")
    elif not state.get("checks_for_phase"):
        pass # Message handled above
    else:
        print("SUMMARY: No valid results were generated.")

    # Add info about truncation to final output if it occurred
    if state.get("document_truncated"):
        if isinstance(state["final_results"], list):
             state["final_results"].insert(0, {"warning": f"Document content was truncated to fit context window (approx. {MAX_CONTEXT_TOKENS} tokens). Analysis based on partial document."})
        print(f"Note: Analysis was performed on truncated document content ({state.get('document_token_count')} original tokens).")

    return state

# --- Graph Definition (Modified for CAG) ---

def decide_after_load(state: GraphState) -> str:
    """Determines the next step after document loading."""
    print(f"\n--- Edge: decide_after_load ---")
    if state.get("error_message"):
        print("Decision: Error detected, routing to format_output.")
        return "format_output"

    checks = state.get("checks_for_phase")
    if not checks:
        print("Decision: No checks found for the phase. Routing to format_output.")
        return "format_output"

    print(f"Decision: {len(checks)} checks found. Routing to analyze_checks_map.")
    return "analyze_checks_map"

# --- Build the Graph ---
graph_builder = StateGraph(GraphState)

# Add nodes
graph_builder.add_node("load_filter_checklist", load_and_filter_checklist)
graph_builder.add_node("load_prepare_document", load_and_prepare_document) # Renamed node

# Define the map operation node using the CAG analysis function
analyze_check_node_cag = RunnableLambda(analyze_check_cag, name="AnalyzeSingleCheckCAG")
graph_builder.add_node(
    "analyze_checks_map",
    RunnableLambda(lambda state: state["checks_for_phase"], name="GetChecksForMap")
    .map(analyze_check_node_cag) # Use the CAG analysis function
    | RunnableLambda(lambda mapped_results: {"analysis_results": mapped_results}, name="CollectMapResults")
)

graph_builder.add_node("format_output", format_final_output)

# Define edges
graph_builder.set_entry_point("load_filter_checklist")
graph_builder.add_edge("load_filter_checklist", "load_prepare_document") # Edge to the new loading node

# Conditional edge after loading
graph_builder.add_conditional_edges(
    "load_prepare_document", # From the new loading node
    decide_after_load,       # Using the new decision function
    {
        "format_output": "format_output",
        "analyze_checks_map": "analyze_checks_map"
    }
)

graph_builder.add_edge("analyze_checks_map", "format_output")
graph_builder.add_edge("format_output", END)

# Compile the graph
try:
    app = graph_builder.compile()
    print("\n--- LangGraph (CAG Version) Compiled Successfully ---")
except Exception as compile_error:
    print(f"FATAL: LangGraph compilation failed: {compile_error}")
    traceback.print_exc()
    exit(1)

# --- Main Execution Block (Similar to RAG, uses CAG app) ---
if __name__ == "__main__":

    # --- Create Dummy Files (Use same logic as RAG version) ---
    checklist_file = "project_checklist_demo.xlsx"
    document_file = "quotation_demo.docx"
    # (Include dummy file creation logic here if needed - identical to RAG version)
    if not os.path.exists(checklist_file):
        print(f"\nCreating dummy checklist file: {checklist_file}")
        # ... (pandas DataFrame creation and save) ...
        dummy_data = { /* ... same data as RAG example ... */ }
        try: pd.DataFrame(dummy_data).to_excel(checklist_file, index=False); print("Dummy checklist created.")
        except Exception as fe: print(f"Error creating dummy checklist: {fe}")

    if not os.path.exists(document_file):
         print(f"Creating dummy document file: {document_file}")
         # ... (python-docx document creation and save) ...
         try:
             from docx import Document as DocxDocument
             # ... (code to create docx content identical to RAG example) ...
             doc = DocxDocument(); # Add paragraphs as before
             doc.add_heading("Project Alpha - Quotation Details", level=1); # ... etc ...
             doc.save(document_file); print("Dummy document created.")
         except Exception as de: print(f"Error creating dummy document: {de}")


    # --- Define Inputs for the Graph Run ---
    inputs = GraphState(
        checklist_path=checklist_file,
        document_path=document_file,
        target_phase="Apertura Commessa" # Analyze this specific phase
    )

    print("\n" + "="*50)
    print("--- Starting Checklist CAG Workflow ---")
    print(f"Checklist: {inputs['checklist_path']}")
    print(f"Document: {inputs['document_path']}")
    print(f"Target Phase: {inputs['target_phase']}")
    print("="*50 + "\n")

    # --- Execute the Graph ---
    final_state = None
    try:
        config = RunnableConfig(recursion_limit=25)
        final_state = app.invoke(inputs, config=config)

        # --- Process Final Output ---
        print("\n" + "="*50)
        print("--- Workflow Completed (CAG) ---")
        print("="*50)

        if final_state and final_state.get("final_results"):
            print("\nFinal Analysis Results (JSON):")
            print(json.dumps(final_state["final_results"], indent=2))

            output_filename = f"analysis_results_CAG_{inputs['target_phase'].replace(' ', '_')}.json"
            try:
                with open(output_filename, 'w') as f:
                    json.dump(final_state["final_results"], f, indent=2)
                print(f"\nResults also saved to: {output_filename}")
            except Exception as write_error:
                print(f"\nError saving results to file: {write_error}")

        elif final_state and final_state.get("error_message"):
             print(f"\nWorkflow finished with an error state: {final_state['error_message']}")
        else:
            print("\n--- Workflow finished, but no final results found in the expected format. ---")
            print("Final State:")
            print(final_state)

    except Exception as e:
        print(f"\n--- FATAL: Workflow Execution Failed (CAG) ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()

    print("\n--- End of script ---")
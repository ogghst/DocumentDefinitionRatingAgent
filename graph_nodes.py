import os
import pandas as pd
import traceback
import asyncio
from typing import List, Optional, Dict, Any

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph, END
from typing_extensions import Annotated, TypedDict

from models import CheckItem, CheckResult
from document_processor import load_document, create_retriever
from rag_engine import create_rag_chain, generate_questions, create_streaming_chain
from common import broadcast_message, get_user_input
from models import GraphState

# Node Functions
async def load_and_filter_checklist(state: GraphState) -> GraphState:
    """Loads checklist from Excel and filters checks for the target phase."""
    await broadcast_message(f"\n--- Node: load_and_filter_checklist ---")
    checklist_path = state['checklist_path']
    target_phase = state['target_phase']
    await broadcast_message(f"Loading Checklist: '{checklist_path}' for Phase: '{target_phase}'")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(checklist_path), exist_ok=True)
    
    try:
        if not os.path.exists(checklist_path):
            await broadcast_message(f"WARNING: Checklist file not found at: {checklist_path}")
            await broadcast_message("Creating a demo checklist file.")
            
            # Create a demo checklist
            dummy_data = {
                'ID': [1, 1, 8, 8, 9, 9, 10, 11],
                'Name': ['Layout Validated', 'Layout Validated', 'Customer Approval', 'Customer Approval', 'Charger Location', 'Charger Location', 'Safety Certs', 'Network Ports'],
                'BranchID': [85, 85, 85, 85, 85, 85, 90, 95],
                'BranchName': ['LAYOUT', 'LAYOUT', 'LAYOUT', 'LAYOUT', 'LAYOUT', 'LAYOUT', 'SAFETY', 'IT'],
                'CHK_Description': [
                    'Plant layout validated for clearances (1.5m+), obstacles, doors, and ceiling specifications (4m+).',
                    'Plant layout validated for clearances (1.5m+), obstacles, doors, and ceiling specifications (4m+).',
                    'Customer approval received for the final offer layout drawing.',
                    'Customer approval received for the final offer layout drawing.',
                    'Battery charger location defined and clearly marked (balooned) in the layout drawing.',
                    'Battery charger location defined and clearly marked (balooned) in the layout drawing.',
                    'Relevant safety certifications (e.g., CE marking) for major components are documented.',
                    'Required network ports identified and locations specified in IT plan.'
                ],
                'Weight': [7, 10, 3, 5, 3, 10, 8, 5],
                'Phase': ['Apertura Commessa', 'Lancio', 'Apertura Commessa', 'Lancio', 'Apertura Commessa', 'Rilascio Tecnico', 'Apertura Commessa', 'Apertura Commessa']
            }
            pd.DataFrame(dummy_data).to_excel(checklist_path, index=False)
            await broadcast_message("Demo checklist created.")

        # Continue with loading the file
        df = pd.read_excel(checklist_path)
        # Standardize column names (lowercase, replace spaces) for robustness
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        required_cols = ['id', 'name', 'branchid', 'branchname', 'chk_description', 'weight', 'phase']
        # Check based on standardized names
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Checklist missing required columns (standardized names): {missing}")

        # Filter by phase (case-insensitive, strip whitespace)
        target_phase_lower = target_phase.strip().lower()
        filtered_df = df[df['phase'].astype(str).str.strip().str.lower() == target_phase_lower]

        await broadcast_message(f"Found {len(filtered_df)} rows for phase '{target_phase}'.")
        
        checks = []
        for _, row in filtered_df.iterrows():
             try:
                 checks.append(CheckItem(
                     id=int(row['id']),
                     name=str(row['name']),
                     branch_id=int(row['branchid']),
                     branch_name=str(row['branchname']),
                     description=str(row['chk_description']),
                     weight=int(row['weight']),
                     phase=str(row['phase'])
                 ))
             except (ValueError, TypeError) as ve:
                 await broadcast_message(f"Warning: Skipping row due to type conversion error: {ve} - Row data: {row.to_dict()}")
                 continue

        await broadcast_message(f"Created {len(checks)} CheckItem objects for phase '{target_phase}'.")
        state["checks_for_phase"] = checks
        state["error_message"] = None
    except Exception as e:
        await broadcast_message(f"ERROR loading checklist: {e}")
        await broadcast_message(f"Traceback: {traceback.format_exc()}")
        state["error_message"] = f"Failed to load/filter checklist: {e}"
        state["checks_for_phase"] = []
    
    # Explicitly log the result
    if state["checks_for_phase"]:
        await broadcast_message(f"Successfully loaded {len(state['checks_for_phase'])} checks.")
    else:
        await broadcast_message("No checks were loaded for the specified phase.")
        
    return state

async def load_index_document(state: GraphState) -> GraphState:
    """Loads the Word document, chunks it, creates embeddings, and sets up the retriever."""
    await broadcast_message(f"\n--- Node: load_index_document ---")
    if state.get("error_message"):
        await broadcast_message("Skipping due to previous error.")
        return state

    document_path = state['document_path']
    await broadcast_message(f"Loading & Indexing Document: '{document_path}'")
    try:
        # Load and chunk the document
        documents = load_document(document_path)
        await broadcast_message(f"Split document into {len(documents)} chunks.")
        
        # Create retriever
        await broadcast_message("Creating vector store with embeddings...")
        state["retriever"] = create_retriever(documents)
        await broadcast_message("Document indexed successfully. Retriever is ready.")
        state["error_message"] = None
    except Exception as e:
        await broadcast_message(f"ERROR loading/indexing document: {e}")
        traceback.print_exc()
        state["error_message"] = f"Failed to load or index document: {e}"
        state["retriever"] = None
    return state

async def analyze_check_rag(check_item: CheckItem, config: RunnableConfig) -> CheckResult:
    """
    Analyzes a single check item using RAG against the indexed document.
    Designed to be called via LangGraph's map functionality.
    """
    await broadcast_message(f"\n--- Analyzing Check ID: {check_item.id} ({check_item.name}) ---")
    
    # Access the shared state (including the retriever) from the config
    state: GraphState = config['configurable']
    retriever = state.get('retriever')
    
    # Get conversation ID from the config if available for user input
    conversation_id = None
    callback_manager = None
    
    # Extract conversation_id from metadata
    if config.get('metadata') and isinstance(config['metadata'], dict):
        conversation_id = config['metadata'].get('conversation_id')
        
        # If we have a conversation_id, try to get the callback manager
        if conversation_id:
            try:
                from websocket_server import conversation_callbacks
                callback_manager = conversation_callbacks.get(conversation_id)
                if callback_manager:
                    print(f"Using callback manager for conversation {conversation_id}")
            except (ImportError, Exception) as e:
                print(f"Could not access callback manager: {e}")
    
    # Pre-computation checks
    if not retriever:
        await broadcast_message(f"ERROR: Retriever not available for check ID {check_item.id}. Skipping analysis.")
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details="Analysis skipped: Document retriever not initialized.",
            needs_human_review=True
        )
    
    # Create the RAG chain
    rag_chain = create_rag_chain()
    
    try:
        # Prepare for interactive analysis
        additional_info = ""
        max_attempts = 3
        attempt = 0
        user_provided_input = False  # Track if the user has already provided input
        prev_reliability = 0  # Track previous reliability score
        
        while attempt < max_attempts:
            # Prepare chain input
            chain_input = {
                "check_id": check_item.id,
                "check_name": check_item.name,
                "check_description": check_item.description,
                "check_phase": check_item.phase,
                "additional_info": additional_info,
                "retriever": retriever
            }
            
            # Now run the actual chain with parser for structured results
            result = await rag_chain.ainvoke(chain_input, config=config)
            
            # Add check_item back (it's not included in the LLM output)
            result.check_item = check_item
            
            # Check if reliability has improved from the previous attempt
            reliability_improved = result.reliability > prev_reliability
            
            # Only check for human review if:
            # 1. This is the first attempt, OR
            # 2. User hasn't provided input yet, OR
            # 3. Reliability has significantly improved from the previous attempt
            needs_human_input = (
                result.reliability < 50 and 
                (attempt == 0 or not user_provided_input or reliability_improved)
            )
            
            # If we're not making progress with reliability, or user already provided input 
            # and reliability is still low, don't ask again
            if not needs_human_input:
                # Accept the result even if reliability is low
                # Mark as needing human review in the final results if reliability < 50
                if result.reliability < 50:
                    result.needs_human_review = True
                    await broadcast_message(f"Low reliability for check ID {check_item.id}: {result.reliability}%. Accepting result without further user input.")
                break
                
            # Update previous reliability score
            prev_reliability = result.reliability
            
            # We need human review
            result.needs_human_review = True
            await broadcast_message(f"Low reliability for check ID {check_item.id}: {result.reliability}%")
            
            # Ask if the user wants to continue or provide more info
            questions = generate_questions(check_item, result)
            
            user_prompt = f"""
# Low Reliability Assessment for Check #{check_item.id}

The system cannot confidently determine if this check is met based on the document.

## Check Details:
- ID: {check_item.id}
- Name: {check_item.name}
- Description: {check_item.description}
- Current Reliability: {result.reliability}%

## Current Assessment:
{result.analysis_details}

## To resolve this, please answer one of these specific questions:
{questions}

## Options:
1. Accept the current assessment (low reliability) and continue to the next check
2. Provide specific information to improve the assessment

Enter "1" to accept and continue, or type your answer to the question(s) above:
"""
            
            # Try to use the callback manager for user input first (direct websocket)
            if callback_manager:
                user_response = await callback_manager.get_user_input(user_prompt)
            else:
                # Fallback to standard get_user_input (might use websocket or console)
                user_response = await get_user_input(user_prompt, conversation_id)
            
            # Mark that user provided input
            user_provided_input = True
            
            if user_response in ["1", "skip", "continue", "accept"]:
                await broadcast_message(f"User chose to accept the current result for check ID {check_item.id}")
                # Record the exact user response
                result.user_input = user_response
                break
            elif user_response in ["2", "improve"]:
                await broadcast_message(f"User chose to provide more information...")
                
                # Get the specific information
                info_prompt = "Please provide additional information about this check item:"
                if callback_manager:
                    additional_info = await callback_manager.get_user_input(info_prompt)
                else:
                    additional_info = await get_user_input(info_prompt, conversation_id)
                
                # Record the exact user input without any modification
                result.user_input = additional_info
                
                # Add to the context for the next attempt
                await broadcast_message(f"Retrying analysis with additional information...")
                attempt += 1
                continue
            else:
                # User provided specific information - store exactly as provided
                additional_info = user_response
                result.user_input = user_response
                await broadcast_message(f"Retrying analysis with user information...")
                attempt += 1
                continue
        
        # Final result with proper determination
        return result
    except Exception as e:
        await broadcast_message(f"ERROR during RAG analysis for check ID {check_item.id}: {e}")
        await broadcast_message(f"RAG Chain Error: {str(e)}")
        await broadcast_message(f"Input used: {chain_input}")
        await broadcast_message(f"CheckItem data: {check_item.model_dump()}")
        traceback.print_exc()
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details=f"Analysis failed due to runtime error: {e}",
            needs_human_review=True
        )

async def format_final_output(state: GraphState) -> GraphState:
    """Formats the results list into the final JSON structure, saves to JSON file, and generates PDF report."""
    await broadcast_message(f"\n--- Node: format_final_output ---")

    if error_message := state.get("error_message"):
        await broadcast_message(f"Workflow finished with error: {error_message}")
        state["final_results"] = [{"error": error_message}]
        return state

    analysis_results = state.get("analysis_results")

    if analysis_results is None:
         if not state.get("checks_for_phase"):
             message = f"No checklist items found for the specified phase '{state['target_phase']}'. No analysis performed."
             await broadcast_message(message)
             state["final_results"] = [{"message": message}]
         else:
             await broadcast_message("Warning: Analysis results list is missing, but checks existed.")
             state["final_results"] = [{"warning": "Analysis results are unexpectedly missing."}]
         return state

    if not isinstance(analysis_results, list):
        await broadcast_message(f"Warning: 'analysis_results' is not a list (type: {type(analysis_results)}). Attempting to format anyway.")
        state["final_results"] = [{"error": "Internal state error: analysis_results is not a list."}]
        return state

    # Convert Pydantic models to dictionaries for JSON serialization
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
            await broadcast_message(f"Warning: Item at index {i} in analysis_results is not a CheckResult object: {result}")
            output_list.append({"error": "Invalid result object received from analysis step.", "details": str(result)})

    state["final_results"] = output_list
    await broadcast_message(f"Formatted {valid_results_count} valid results.")

    # Log summary about human review items
    if valid_results_count > 0:
        if human_review_count > 0:
            await broadcast_message(f"SUMMARY: {human_review_count} out of {valid_results_count} analyzed checks require human review (Reliability < 50%).")
        else:
            await broadcast_message(f"SUMMARY: All {valid_results_count} analyzed checks met the reliability threshold (>= 50%).")
    
    # Save results to JSON file
    import json
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the JSON filename
    json_filename = f"output/analysis_results_{timestamp}.json"
    
    # Save the results to the JSON file
    try:
        with open(json_filename, 'w') as f:
            json.dump(state["final_results"], f, indent=2)
        await broadcast_message(f"Results saved to JSON file: {json_filename}")
    except Exception as e:
        await broadcast_message(f"Error saving results to JSON file: {e}")
    
    # Generate PDF report
    try:
        from report_generator import ChecklistReportGenerator
        
        # Create the report generator instance
        report_generator = ChecklistReportGenerator(json_filename)
        
        # Generate the PDF report
        pdf_path = report_generator.generate_pdf()
        
        await broadcast_message(f"PDF report generated: {pdf_path}")
    except Exception as e:
        await broadcast_message(f"Error generating PDF report: {e}")
        
    return state

async def analyze_checks_map_function(state: GraphState) -> GraphState:
    """Takes all checks and maps the analyze function over them with proper state config."""
    await broadcast_message(f"\n--- Node: analyze_checks_map function ---")
    checks = state.get("checks_for_phase", [])
    await broadcast_message(f"Processing {len(checks)} checks in map function...")
    
    if not checks:
        await broadcast_message("No checks to process. Setting empty results list.")
        state["analysis_results"] = []
        return state
    
    # Create a config with the current state for each analysis
    config = {"configurable": state}
    
    # Map the analysis function over each check with proper configuration
    results = []
    for check in checks:
        try:
            analyze_check_node = RunnableLambda(analyze_check_rag, name="AnalyzeSingleCheck")
            result = await analyze_check_node.ainvoke(check, config=config)
            results.append(result)
            await broadcast_message(f"Successfully analyzed check ID {check.id}")
        except Exception as e:
            await broadcast_message(f"Error analyzing check ID {check.id}: {e}")
            # Create a placeholder result for failed checks
            results.append(CheckResult(
                check_item=check, is_met=None, reliability=0.0, sources=[],
                analysis_details=f"Analysis failed: {str(e)}",
                needs_human_review=True
            ))
    
    # Update the state with results
    await broadcast_message(f"Completed analysis of {len(results)} checks.")
    state["analysis_results"] = results
    return state

async def decide_after_indexing(state: GraphState) -> str:
    """Determines the next step after document indexing."""
    await broadcast_message(f"\n--- Edge: decide_after_indexing ---")
    if state.get("error_message"):
        await broadcast_message("Decision: Error detected, routing to format_output.")
        return "format_output"

    checks = state.get("checks_for_phase")
    if not checks:
        await broadcast_message("Decision: No checks found for the phase. Routing to format_output.")
        return "format_output"

    await broadcast_message(f"Decision: {len(checks)} checks found. Routing to analyze_checks_map.")
    return "analyze_checks_map"

# Build and configure the LangGraph
def create_workflow_graph():
    """Create and return the workflow graph."""
    graph_builder = StateGraph(GraphState)
    
    # Add nodes
    graph_builder.add_node("load_filter_checklist", load_and_filter_checklist)
    graph_builder.add_node("load_index_document", load_index_document)
    graph_builder.add_node("analyze_checks_map", analyze_checks_map_function)
    graph_builder.add_node("format_output", format_final_output)
    
    # Define edges
    graph_builder.set_entry_point("load_filter_checklist")
    graph_builder.add_edge("load_filter_checklist", "load_index_document")
    
    # Conditional edge after indexing
    graph_builder.add_conditional_edges(
        "load_index_document",
        decide_after_indexing,
        {
            "format_output": "format_output",
            "analyze_checks_map": "analyze_checks_map"
        }
    )
    
    # After map operation completes, go to format the output
    graph_builder.add_edge("analyze_checks_map", "format_output")
    
    # Final node leads to the end
    graph_builder.add_edge("format_output", END)
    
    # Compile and return the graph
    return graph_builder.compile() 
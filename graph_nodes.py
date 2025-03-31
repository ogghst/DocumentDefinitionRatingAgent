import os
import pandas as pd
import traceback
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph, END
from typing_extensions import Annotated, TypedDict

from models import CheckItem, CheckResult
from document_processor import load_document, create_retriever
from rag_engine import create_hybrid_rag_chain, generate_questions, create_streaming_chain
from common import broadcast_message, get_user_input
from models import GraphState
from report_generator import generate_pdf_report

# Import WebsocketCallbackManager - add this with try/except to handle optional dependency
try:
    from websocket_callbacks import WebsocketCallbackManager
except ImportError:
    # Create a stub class if the real one isn't available
    class WebsocketCallbackManager:
        pass

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
    """Analyze a single check item using RAG, with interactive human review if needed."""
    
    callback_manager = None
    conversation_id = None
    retriever = None
    
    # Extract our callback manager and conversation ID from config metadata
    if "metadata" in config:
        metadata = config["metadata"]
        if "conversation_id" in metadata:
            conversation_id = metadata["conversation_id"]
            
        if "retriever" in metadata:
            retriever = metadata["retriever"]
            
        if "callback_manager" in metadata:
            callback_manager = metadata["callback_manager"]
    
    # If retriever not in metadata, try to get from configurable
    if not retriever and "configurable" in config:
        retriever = config["configurable"].get("retriever")
    
    if not retriever:
        await broadcast_message(f"ERROR: No retriever found in config for check ID {check_item.id}")
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details="Analysis failed: No document retriever available",
            needs_human_review=True
        )
    
    # Create the RAG chain
    hybrid_rag_chain = create_hybrid_rag_chain()
    
    try:
        # Prepare for interactive analysis
        additional_info = ""
        max_attempts = 3
        attempt = 0
        user_provided_input = False  # Track if the user has already provided input
        prev_reliability = 0  # Track previous reliability score
        saved_user_input = None  # Save the user's input across attempts
        
        while attempt < max_attempts:
            # Prepare chain input
            chain_input = {
                "check_id": check_item.id,
                "check_name": check_item.name,
                "check_description": check_item.description,
                "additional_info": additional_info,
                "retriever": retriever
            }
            
            # Now run the actual chain with parser for structured results
            # Set up config to include the callback_manager for streaming
            chain_config = {}
            
            if callback_manager:
                try:
                    # Use directly as a callback handler
                    chain_config["callbacks"] = [callback_manager]
                except Exception as e:
                    await broadcast_message(f"Warning: Failed to set up callback manager: {e}")
                    
            # Pass any metadata needed
            chain_config["metadata"] = {"conversation_id": conversation_id}
            
            # Run the chain with the appropriate config
            result = await hybrid_rag_chain.ainvoke(chain_input, config=chain_config)
            
            # Extract the parsed result from the hybrid output
            if "parsed_result" in result:
                check_result = result["parsed_result"]
                check_result.check_item = check_item  # Add back the check item
            elif "parsing_error" in result:
                # Handle parsing errors
                await broadcast_message(f"Error parsing LLM output: {result['parsing_error']}")
                check_result = CheckResult(
                    check_item=check_item,
                    is_met=False,
                    reliability=0,
                    analysis_details=f"Failed to parse LLM output: {result['parsing_error']}",
                    needs_human_review=True
                )
            
            # Preserve user input across attempts
            if saved_user_input is not None:
                check_result.user_input = saved_user_input
            
            # Check if reliability has improved from the previous attempt
            reliability_improved = check_result.reliability > prev_reliability
            
            # Only check for human review if:
            # 1. Reliability is low, AND
            # 2. Either this is the first attempt OR user hasn't provided input yet OR reliability hasn't improved
            needs_human_input = (
                (check_result.reliability < 50) and 
                (attempt == 0 or not user_provided_input or not reliability_improved)
            )
            
            # If we're not making progress with reliability, or user already provided input 
            # and reliability is still low, don't ask again
            if not needs_human_input:
                # Accept the result even if reliability is low
                # Mark as needing human review in the final results if reliability < 50
                if check_result.reliability < 50:
                    check_result.needs_human_review = True
                    await broadcast_message(f"Low reliability for check ID {check_item.id}: {check_result.reliability}%. Accepting result without further user input.")
                    
                    # Ensure user_input field is set to indicate no input was requested
                    if check_result.user_input is None:
                        check_result.user_input = "No input requested - automated assessment accepted"
                break
                
            # Update previous reliability score
            prev_reliability = check_result.reliability
            
            # We need human review
            check_result.needs_human_review = True
            await broadcast_message(f"Low reliability ({check_result.reliability}%) for check ID {check_item.id}: Requesting human input")
            
            # Ask if the user wants to continue or provide more info
            questions = generate_questions(check_item, check_result, callback_manager)
            
            user_prompt = f"""
# Low Reliability Assessment for Check #{check_item.id}

The system cannot confidently determine if this check is met based on the document.

## Check Details:
- ID: {check_item.id}
- Name: {check_item.name}
- Description: {check_item.description}
- Current Reliability: {check_result.reliability}%

## Current Assessment:
{check_result.analysis_details}

## To resolve this, please answer one of these specific questions:
{questions}

## Options:
1. Accept the current assessment (low reliability) and continue to the next check
2. Provide specific information to improve the assessment

Enter "1" to accept and continue, or type your answer to the question(s) above:
"""
            
            # Try to use the callback manager for user input first (direct websocket)
            if callback_manager:
                try:
                    user_response = await callback_manager.get_user_input(user_prompt)
                except Exception as e:
                    await broadcast_message(f"Error getting user input via callback manager: {e}")
                    user_response = "timeout"
                    check_result.user_input = f"Input request failed: {str(e)}"
            else:
                # Fallback to standard get_user_input (might use websocket or console)
                try:
                    user_response = await get_user_input(user_prompt, conversation_id)
                except Exception as e:
                    await broadcast_message(f"Error getting user input: {e}")
                    user_response = "timeout"
                    check_result.user_input = f"Input request failed: {str(e)}"
            
            # Mark that user provided input
            user_provided_input = True
            
            if user_response in ["1", "skip", "continue", "accept"]:
                await broadcast_message(f"User chose to accept the current result for check ID {check_item.id}")
                # Record the exact user response
                check_result.user_input = user_response
                saved_user_input = user_response
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
                check_result.user_input = additional_info
                saved_user_input = additional_info  # Save it for subsequent attempts
                
                # Add to the context for the next attempt
                await broadcast_message(f"Retrying analysis with additional information...")
                attempt += 1
                continue
            else:
                # User provided specific information - store exactly as provided
                additional_info = user_response
                check_result.user_input = user_response
                saved_user_input = user_response  # Save it for subsequent attempts
                await broadcast_message(f"Retrying analysis with user information...")
                attempt += 1
                continue
        
        # Final result with proper determination
        # Ensure needs_human_review remains true if human input was provided
        if user_provided_input:
            check_result.needs_human_review = True
            
        # IMPORTANT: Never overwrite actual user input with generic messages
        # Only set the generic message if no user input was recorded
        if check_result.user_input is None and check_result.needs_human_review:
            check_result.user_input = "No input requested - automated assessment accepted"
            
        # Add debugging information
        await broadcast_message(f"Final result for check ID {check_item.id} - user_input: '{check_result.user_input}'")
            
        return check_result
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
    """Format the final output and save results to JSON file."""
    try:
        if not state.get("analysis_results"):
            state["error_message"] = "No analysis results to format"
            return state
            
        # Convert results to dictionaries, ensuring all fields are properly handled
        results = []
        for result in state["analysis_results"]:
            result_dict = result.model_dump()
            
            # Ensure needs_human_review is properly set based on whether human input was provided
            if result_dict.get("user_input") is not None and result_dict.get("user_input") != "No input requested - automated assessment accepted":
                result_dict["needs_human_review"] = True
                
            # IMPORTANT: Only set the generic message if no user input was recorded
            # Never overwrite actual user input with generic messages
            if result_dict.get("user_input") is None and result_dict.get("needs_human_review"):
                result_dict["user_input"] = "No input requested - automated assessment accepted"
                
            results.append(result_dict)
            
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        json_path = output_dir / f"analysis_results_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        await broadcast_message(f"Results saved to JSON file: {json_path}")
        
        # Generate PDF report
        pdf_path = output_dir / f"analysis_results_{timestamp}_report.pdf"
        try:
            await generate_pdf_report(results, str(pdf_path))
            await broadcast_message(f"PDF report generated: {pdf_path}")
        except Exception as e:
            await broadcast_message(f"Warning: Failed to generate PDF report: {e}")
            # Continue even if PDF generation fails
        
        state["final_results"] = results
        return state
        
    except Exception as e:
        state["error_message"] = f"Error formatting final output: {str(e)}"
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
    
    # Extract the callback manager and conversation ID if they exist in the state
    callback_manager = state.get("callback_manager")
    conversation_id = state.get("conversation_id")
    
    # Create a base config without callbacks to avoid AsyncCallbackManager conversion errors
    config = {
        "configurable": state,
        "metadata": {
            "conversation_id": conversation_id,
            "retriever": state.get("retriever"),
            "callback_manager": callback_manager  # Pass as metadata instead
        }
    }
    
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
            traceback.print_exc()  # Print full traceback for debugging
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
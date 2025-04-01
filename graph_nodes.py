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
from rag_engine import create_hybrid_rag_chain, generate_questions
# Remove deprecated imports from common
# from common import send_message, get_user_input
from models import GraphState
from report_generator import generate_pdf_report

# Import WebsocketCallbackManager - remains the same
try:
    from websocket_callbacks import WebsocketCallbackManager
except ImportError:
    class WebsocketCallbackManager:
        pass # Stub remains

# Helper function to safely send messages via callback manager
async def safe_send_message(state_or_config: Any, message: str, message_type: str = "rag_progress"):
    """Safely sends a message using the callback manager found in state or config."""
    callback_manager = None
    conversation_id = None

    if isinstance(state_or_config, dict): # Assuming GraphState is a dict
        callback_manager = state_or_config.get("callback_manager")
        conversation_id = state_or_config.get("conversation_id")
    elif hasattr(state_or_config, "metadata") and isinstance(state_or_config.metadata, dict): # Assuming RunnableConfig
        callback_manager = state_or_config.metadata.get("callback_manager")
        conversation_id = state_or_config.metadata.get("conversation_id")
    
    if callback_manager and hasattr(callback_manager, '_send_message'):
        try:
            await callback_manager._send_message(message, message_type)
        except Exception as e:
            print(f"Error sending message via callback manager for conv {conversation_id}: {e}")
    else:
        # Fallback print if no callback manager or method is found
        log_prefix = f"[safe_send_message][{conversation_id or 'UNKNOWN_CONV_ID'}]"
        print(f"{log_prefix} Callback manager not found or invalid. Message not sent: {message}")

# Node Functions
async def load_and_filter_checklist(state: GraphState) -> GraphState:
    """Loads checklist from Excel and filters checks for the target phase."""
    conversation_id = state.get("conversation_id")
    await safe_send_message(state, f"\n--- Node: load_and_filter_checklist ---")
    checklist_path = state['checklist_path']
    target_phase = state['target_phase']
    await safe_send_message(state, f"Loading Checklist: '{checklist_path}' for Phase: '{target_phase}'")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(checklist_path), exist_ok=True)
    
    try:
        if not os.path.exists(checklist_path):
            await safe_send_message(state, f"WARNING: Checklist file not found at: {checklist_path}", "warning")
            await safe_send_message(state, "Creating a demo checklist file.", "info")
            
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
            await safe_send_message(state, "Demo checklist created.", "info")

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

        await safe_send_message(state, f"Found {len(filtered_df)} rows for phase '{target_phase}'.")
        
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
                 await safe_send_message(state, f"Warning: Skipping row due to type conversion error: {ve} - Row data: {row.to_dict()}", "warning")
                 continue

        await safe_send_message(state, f"Created {len(checks)} CheckItem objects for phase '{target_phase}'.")
        state["checks_for_phase"] = checks
        state["error_message"] = None
    except Exception as e:
        await safe_send_message(state, f"ERROR loading checklist: {e}", "error")
        # Log traceback to console, not via websocket
        print(f"[ERROR][{conversation_id}] Traceback (load_and_filter_checklist): {traceback.format_exc()}")
        # await safe_send_message(state, f"Traceback: {traceback.format_exc()}", "error") # Avoid sending full traceback
        state["error_message"] = f"Failed to load/filter checklist: {e}"
        state["checks_for_phase"] = []
    
    # Explicitly log the result
    if state["checks_for_phase"]:
        await safe_send_message(state, f"Successfully loaded {len(state['checks_for_phase'])} checks.")
    else:
        await safe_send_message(state, "No checks were loaded for the specified phase.", "warning")
        
    return state

async def load_index_document(state: GraphState) -> GraphState:
    """Loads the Word document, chunks it, creates embeddings, and sets up the retriever."""
    conversation_id = state.get("conversation_id")
    await safe_send_message(state, f"\n--- Node: load_index_document ---")
    if state.get("error_message"):
        await safe_send_message(state, "Skipping due to previous error.")
        return state

    document_path = state['document_path']
    await safe_send_message(state, f"Loading & Indexing Document: '{document_path}'")
    try:
        # Load and chunk the document
        documents = load_document(document_path)
        await safe_send_message(state, f"Split document into {len(documents)} chunks.")
        
        # Create retriever
        await safe_send_message(state, "Creating vector store with embeddings...")
        state["retriever"] = create_retriever(documents)
        await safe_send_message(state, "Document indexed successfully. Retriever is ready.")
        state["error_message"] = None
    except Exception as e:
        await safe_send_message(state, f"ERROR loading/indexing document: {e}", "error")
        # Log traceback to console
        print(f"[ERROR][{conversation_id}] Traceback (load_index_document): {traceback.format_exc()}")
        state["error_message"] = f"Failed to load or index document: {e}"
        state["retriever"] = None
    return state

async def analyze_check_rag(
    check_item: CheckItem, 
    retriever: Any, # Pass retriever directly
    callback_manager: Optional[WebsocketCallbackManager], # Pass manager directly
    conversation_id: Optional[str] # Pass conversation_id directly
) -> CheckResult:
    """Analyze a single check item using RAG, with interactive human review if needed."""
    
    # Remove config extraction logic - arguments are passed directly
    # callback_manager = None
    # conversation_id = None
    # retriever = None
    # send_callback_func = None
    # request_input_func = None
    # ... (remove config parsing) ...
            
    # Helpers now use the passed callback_manager object's methods
    async def send_node_message(message: str, message_type: str = "rag_progress"):
        # Use the passed callback_manager object
        if callback_manager and hasattr(callback_manager, '_send_message'):
            try:
                await callback_manager._send_message(message, message_type)
            except Exception as e:
                print(f"[analyze_check_rag][{conversation_id or 'UNKNOWN'}] Error sending message: {e}")
        else:
            # Log to console if CBM is unavailable
            print(f"[analyze_check_rag][{conversation_id or 'UNKNOWN'}] CBM unavailable. Msg not sent: {message}")

    async def get_node_input(prompt: str) -> str:
        # Use the passed callback_manager object
        if callback_manager and hasattr(callback_manager, 'get_user_input'):
            try:
                return await callback_manager.get_user_input(prompt)
            except Exception as e:
                await send_node_message(f"Error getting user input via callback manager: {e}", "error")
                return "error: callback failed"
        else:
            # Log to console and return error if CBM unavailable for input
            print(f"[analyze_check_rag][{conversation_id or 'UNKNOWN'}] CBM unavailable. Cannot request input: {prompt}")
            return "error: no callback manager"

    # Check passed arguments directly
    if not retriever:
        # Try to send message using the potentially available CBM
        await send_node_message(f"ERROR: No retriever provided for check ID {check_item.id}", "error")
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details="Failed: Retriever object not provided to analysis function.",
            needs_human_review=True
        )
        
    # Check if CBM was provided (it's Optional)
    if not callback_manager:
         print(f"[analyze_check_rag][{conversation_id or 'UNKNOWN'}] WARNING: Callback manager not provided for check ID {check_item.id}. Input/Output disabled for this check.")
         # Proceed without callbacks, but human review might be needed if reliability is low

    # --- Function body remains largely the same, but uses passed args and helpers --- 
    
    # Send check data to client before analysis (using helper)
    check_data_content = {
        "id": check_item.id,
        "name": check_item.name,
        "description": check_item.description,
        "branch_id": check_item.branch_id,
        "branch_name": check_item.branch_name,
        "weight": check_item.weight,
        "phase": check_item.phase
    }
    await send_node_message(json.dumps(check_data_content), "check_start")
    
    # Create the RAG chain
    hybrid_rag_chain = create_hybrid_rag_chain()
    
    chain_input = {} 
    try:
        # Prepare for interactive analysis
        additional_info = ""
        max_attempts = 3
        attempt = 0
        user_provided_input = False
        prev_reliability = 0
        saved_user_input = None
        check_result = None 

        while attempt < max_attempts:
            # Prepare chain input (uses passed retriever)
            chain_input = {
                "check_id": check_item.id,
                "check_name": check_item.name,
                "check_description": check_item.description,
                "additional_info": additional_info,
                "retriever": retriever # Use directly passed retriever
            }
            
            # RAG Chain config - Use an empty config or pass one if needed by the chain itself
            # We don't need to pass context via config anymore for this node's logic.
            # If the *inner* RAG chain needs callbacks, they must be handled differently (e.g., globally)
            # or passed explicitly if the chain supports it.
            chain_run_config = RunnableConfig(
                # If the hybrid_rag_chain itself needs callbacks for internal LLM calls,
                # we might pass the callback_manager here, but it depends on the chain's design.
                # For now, assume the node-level callbacks are sufficient.
                # callbacks=[callback_manager] if callback_manager else [] 
            )
            
            # Run the chain
            result = await hybrid_rag_chain.ainvoke(chain_input, config=chain_run_config)
            
            if "parsed_result" in result:
                check_result = result["parsed_result"]
                if isinstance(check_result, CheckResult):
                     check_result.check_item = check_item  # Add back the check item
                else:
                     # Handle unexpected result type
                     await send_node_message(f"Error: Unexpected result type from RAG chain: {type(result)}", "error")
                     check_result = CheckResult(check_item=check_item, is_met=False, reliability=0, analysis_details="Internal error: Unexpected RAG result type", needs_human_review=True)

            elif "parsing_error" in result:
                # Handle parsing errors
                await send_node_message(f"Error parsing LLM output: {result['parsing_error']}", "error")
                check_result = CheckResult(
                    check_item=check_item,
                    is_met=False,
                    reliability=0,
                    analysis_details=f"Failed to parse LLM output: {result['parsing_error']}",
                    needs_human_review=True
                )
            else:
                 # Handle case where neither parsed_result nor parsing_error is present
                 await send_node_message("Error: RAG chain returned unexpected output structure.", "error")
                 check_result = CheckResult(check_item=check_item, is_met=False, reliability=0, analysis_details="Internal error: Unexpected RAG output structure", needs_human_review=True)

            # Preserve user input across attempts
            if saved_user_input is not None:
                check_result.user_input = saved_user_input
            
            # Check if reliability has improved from the previous attempt
            current_reliability = check_result.reliability if check_result else 0
            reliability_improved = current_reliability > prev_reliability
            
            # Check if human input is needed
            needs_human_input_check = (
                (current_reliability < 50) and 
                (attempt == 0 or not user_provided_input or not reliability_improved)
            )
            
            # Also need callback manager to be available to ask for input
            can_request_human_input = needs_human_input_check and callback_manager is not None

            if not can_request_human_input:
                # Accept the result if we can't ask for input or don't need to
                if current_reliability < 50:
                    if check_result: check_result.needs_human_review = True
                    await send_node_message(f"Low reliability ({current_reliability}%) for check ID {check_item.id}. Accepting result (cannot/don't need to request input).", "warning")
                    if check_result and check_result.user_input is None:
                        check_result.user_input = "No input requested/possible - automated assessment accepted"
                break 
            
            # --- Request Human Input --- 
            prev_reliability = current_reliability
            if check_result: check_result.needs_human_review = True
            await send_node_message(f"Low reliability ({current_reliability}%) for check ID {check_item.id}. Requesting human input")
            
            # Pass callback_manager to generate_questions if it uses it
            questions = generate_questions(check_item, check_result, callback_manager)
            
            user_prompt = f"""
# Low Reliability Assessment for Check #{check_item.id}

The system cannot confidently determine if this check is met based on the document.

## Check Details:
- ID: {check_item.id}
- Name: {check_item.name}
- Description: {check_item.description}
- Current Reliability: {current_reliability}%

## Current Assessment:
{check_result.analysis_details if check_result else 'N/A'}

## To resolve this, please answer one of these specific questions:
{questions}

## Options:
1. Accept the current assessment (low reliability) and continue to the next check
2. Provide specific information to improve the assessment

Enter "1" to accept and continue, or type your answer to the question(s) above:
"""
            
            # Use the get_node_input helper (checks internally if CBM is valid)
            user_response = await get_node_input(user_prompt)

            if user_response.startswith("error:"): 
                 await send_node_message(f"Input request failed ... Accepting current result.", "warning")
                 if check_result: check_result.user_input = f"Input request failed: {user_response}"
                 saved_user_input = check_result.user_input if check_result else "Input request failed"
                 break

            user_provided_input = True
            
            if user_response.strip() in ["1", "skip", "continue", "accept"]:
                await send_node_message(f"User chose to accept ... Accepting current result.")
                if check_result: check_result.user_input = user_response
                saved_user_input = user_response
                break 
            else:
                additional_info = user_response 
                if check_result: check_result.user_input = user_response
                saved_user_input = user_response
                await send_node_message(f"Retrying analysis with user information...")
                attempt += 1
                continue
        
        # Final result determination (check_result should be set from the loop)
        if not check_result:
             await send_node_message(f"ERROR: No CheckResult object generated ... {e}", "error")
             return CheckResult(check_item=check_item, is_met=None, reliability=0, analysis_details="Internal error: Failed to generate result object", needs_human_review=True)

        # Ensure needs_human_review remains true if human input was provided
        if user_provided_input:
            check_result.needs_human_review = True
            
        # IMPORTANT: Never overwrite actual user input with generic messages
        # Only set the generic message if no user input was recorded AND needs_human_review is true
        if check_result.user_input is None and check_result.needs_human_review:
            check_result.user_input = "No input requested - automated assessment accepted"
            
        # Add debugging information
        await send_node_message(f"Final result for check ID {check_item.id} ... {check_result.user_input}")
            
        return check_result
    except Exception as e:
        await send_node_message(f"ERROR during RAG analysis ... {e}", "error")
        # Log traceback to console
        print(f"[analyze_check_rag][{conversation_id}] ERROR Traceback: {traceback.format_exc()}")
        print(f"[analyze_check_rag][{conversation_id}] Input used: {chain_input}")
        print(f"[analyze_check_rag][{conversation_id}] CheckItem data: {check_item.model_dump()}")
        
        return CheckResult(
            check_item=check_item, is_met=None, reliability=0.0, sources=[],
            analysis_details=f"Analysis failed due to runtime error: {e}",
            needs_human_review=True
        )

async def format_final_output(state: GraphState) -> GraphState:
    """Format the final output and save results to JSON file."""
    conversation_id = state.get("conversation_id") 
    log_prefix = f"[Node][format_final_output][{conversation_id or 'UNKNOWN_CONV_ID'}]"
    
    await safe_send_message(state, "--- Node: format_final_output ---")
    
    try:
        analysis_results = state.get("analysis_results")
        if analysis_results is None: # Check for None explicitly
            await safe_send_message(state, "No analysis results found in state.", "warning")
            state["error_message"] = "No analysis results to format"
            state["final_results"] = [] # Ensure final_results is an empty list
            return state
            
        # Convert results to dictionaries, ensuring all fields are properly handled
        results_list = []
        for result in analysis_results:
             if isinstance(result, CheckResult):
                 result_dict = result.model_dump()
                 # Logic for needs_human_review and user_input remains the same
                 if result_dict.get("user_input") is not None and result_dict.get("user_input") != "No input requested - automated assessment accepted":
                     result_dict["needs_human_review"] = True
                 if result_dict.get("user_input") is None and result_dict.get("needs_human_review"):
                     result_dict["user_input"] = "No input requested - automated assessment accepted"
                 results_list.append(result_dict)
             else:
                  await safe_send_message(state, f"Warning: Found non-CheckResult item in analysis_results: {type(result)}", "warning")
            
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        json_filename = f"analysis_results_{conversation_id}_{timestamp}.json" if conversation_id else f"analysis_results_{timestamp}.json"
        json_path = output_dir / json_filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
            
        print(f"{log_prefix} Results saved to JSON file: {json_path}")
        await safe_send_message(state, f"Results saved to JSON file: {json_path}")
        
        # Generate PDF report
        pdf_filename = f"analysis_report_{conversation_id}_{timestamp}.pdf" if conversation_id else f"analysis_report_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename
        try:
            await generate_pdf_report(results_list, str(pdf_path))
            print(f"{log_prefix} PDF report generated: {pdf_path}")
            await safe_send_message(state, f"PDF report generated: {pdf_path}")
        except Exception as e:
            print(f"{log_prefix} Warning: Failed to generate PDF report: {e}")
            await safe_send_message(state, f"Warning: Failed to generate PDF report: {e}", "warning")
            # Continue even if PDF generation fails
        
        state["final_results"] = results_list
        state["error_message"] = None # Clear previous errors if formatting succeeded
        return state
        
    except Exception as e:
        print(f"{log_prefix} Error during final output formatting: {e}")
        traceback.print_exc() # Print traceback for the formatting error
        state["error_message"] = f"Error formatting final output: {str(e)}"
        # Send error message back if possible
        await safe_send_message(state, state["error_message"], "error")
        state["final_results"] = [] # Ensure final_results is empty on error
        return state

async def analyze_checks_map_function(state: GraphState) -> GraphState:
    """Takes all checks and runs the analyze function sequentially for each."""
    conversation_id = state.get("conversation_id") 
    await safe_send_message(state, f"\n--- Node: analyze_checks_map function (Sequential) ---")
    checks = state.get("checks_for_phase", [])
    await safe_send_message(state, f"Processing {len(checks)} checks sequentially...")
    
    if not checks:
        await safe_send_message(state, "No checks to process.")
        state["analysis_results"] = []
        return state
    
    # Get context directly from state
    retriever_in_state = state.get("retriever")
    callback_manager_in_state = state.get("callback_manager") 
    
    # --- Debugging Log (can remain) --- 
    print(f"[analyze_checks_map][{conversation_id}] Retriever type in state: {type(retriever_in_state)}")
    print(f"[analyze_checks_map][{conversation_id}] CallbackManager type in state: {type(callback_manager_in_state)}")
    # --- End Debugging Log ---

    # Check if retriever is available before starting loop
    if not retriever_in_state:
        await safe_send_message(state, "Error: Retriever not found in state for sequential analysis.", "error")
        state["analysis_results"] = [CheckResult(check_item=c, analysis_details="Internal Error: Map setup failed (no retriever)", needs_human_review=True) for c in checks]
        return state
    
    # --- Sequential Processing Loop --- 
    final_results = []
    for i, check in enumerate(checks):
        await safe_send_message(state, f"Analyzing check {i+1}/{len(checks)}: ID {check.id} - '{check.name}'")
        try:
            # Directly call and await analyze_check_rag for the current check
            result = await analyze_check_rag(
                check_item=check, 
                retriever=retriever_in_state, 
                callback_manager=callback_manager_in_state, 
                conversation_id=conversation_id
            )
            
            if isinstance(result, CheckResult):
                 final_results.append(result)
                 await safe_send_message(state, f"Successfully analyzed check ID {check.id}")
            else:
                 # Handle unexpected return type from the function call
                 await safe_send_message(state, f"Unexpected result type for check ID {check.id}: {type(result)}", "warning")
                 final_results.append(CheckResult(
                      check_item=check, is_met=None, reliability=0.0, sources=[],
                      analysis_details=f"Analysis failed: Unexpected return type {type(result)}",
                      needs_human_review=True
                 ))

        except Exception as error:
            # Handle exceptions raised during the await analyze_check_rag call
            await safe_send_message(state, f"Error analyzing check ID {check.id}: {error}", "error")
            print(f"[analyze_checks_map][{conversation_id}] Error analyzing check ID {check.id}: {error}")
            # Optionally log traceback here if needed
            # traceback.print_exc()
            final_results.append(CheckResult(
                check_item=check, is_met=None, reliability=0.0, sources=[],
                analysis_details=f"Analysis failed during execution: {str(error)}",
                needs_human_review=True
            ))
        # Small delay between checks? Optional, might help with rate limits or resource usage
        # await asyncio.sleep(0.1)
            
    # --- End Sequential Processing Loop ---

    # Update the state with results
    await safe_send_message(state, f"Completed sequential analysis for {len(final_results)} checks.")
    state["analysis_results"] = final_results
    return state

async def decide_after_indexing(state: GraphState) -> str:
    """Determines the next step after document indexing."""
    conversation_id = state.get("conversation_id") 
    log_prefix = f"[Edge][decide_after_indexing][{conversation_id or 'UNKNOWN_CONV_ID'}]"
    
    # Use safe_send_message helper
    await safe_send_message(state, f"--- Edge: decide_after_indexing ---")
        
    if state.get("error_message"):
        print(f"{log_prefix} Decision: Error detected, routing to format_output.")
        await safe_send_message(state, "Decision: Error detected, routing to format_output.")
        return "format_output"

    checks = state.get("checks_for_phase")
    if not checks:
        print(f"{log_prefix} Decision: No checks found for the phase. Routing to format_output.")
        await safe_send_message(state, "Decision: No checks found for the phase. Routing to format_output.")
        return "format_output"

    print(f"{log_prefix} Decision: {len(checks)} checks found. Routing to analyze_checks_map.")
    await safe_send_message(state, f"Decision: {len(checks)} checks found. Routing to analyze_checks_map.")
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
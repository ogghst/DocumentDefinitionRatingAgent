import os
import json
import asyncio
import traceback
import websockets
from datetime import datetime
from typing import Dict, Any
import signal
import threading
import time
import sys

from models import GraphState
from graph_nodes import create_workflow_graph
from websocket_server import (
    start_websocket_server, 
    broadcast_message, 
    add_connection, 
    remove_connection,
    get_active_connections
)
# Import missing function from utils.py
from utils import create_demo_files

async def process_client_request(websocket, request_data):
    """Process a client request to run the checklist analysis."""
    # Create a watchdog specific to this request
    request_watchdog = Watchdog(timeout=600)  # 10 minutes timeout for request processing
    request_watchdog.start()
    
    try:
        print("\n=== STARTING PROCESSING CLIENT REQUEST ===")  # Direct console logging
        await broadcast_message("\n" + "="*50)
        await broadcast_message("--- Starting Checklist RAG Workflow ---")
        await broadcast_message("Processing request...")
        print("Broadcast initial messages")  # Console logging
        
        # Apply defaults or use provided values
        checklist_file = request_data.get("checklist_path", "checklists/project_checklist_demo.xlsx")
        document_file = request_data.get("document_path", "input/quotation_demo.docx")
        target_phase = request_data.get("target_phase", "Apertura Commessa")
        
        print(f"Input parameters: {checklist_file}, {document_file}, {target_phase}")  # Console logging
        
        # Log the file existence for debugging
        print(f"Checking files:")
        print(f"Checklist exists: {os.path.exists(checklist_file)}")
        print(f"Document exists: {os.path.exists(document_file)}")
        
        await broadcast_message(f"Checklist: {checklist_file}")
        await broadcast_message(f"Document: {document_file}")
        await broadcast_message(f"Target Phase: {target_phase}")
        await broadcast_message("="*50 + "\n")
        
        # Create input state
        print("Creating input state")
        inputs = GraphState(
            checklist_path=checklist_file,
            document_path=document_file,
            target_phase=target_phase
        )
        
        # Create and run the graph
        print("Creating workflow graph")
        await broadcast_message("Creating workflow graph...")
        
        try:
            app = create_workflow_graph()
            print("Workflow graph created successfully")
        except Exception as graph_error:
            print(f"ERROR creating workflow graph: {graph_error}")
            traceback.print_exc()
            raise
        
        # Configure and run graph
        from langchain_core.runnables import RunnableConfig
        config = RunnableConfig(recursion_limit=25)
        
        print("Executing analysis workflow - this may take some time...")
        await broadcast_message("Executing analysis workflow...")
        
        try:
            print("Before app.ainvoke")
            final_state = await app.ainvoke(inputs, config=config)
            print("After app.ainvoke - workflow completed")
            print(f"Final state keys: {list(final_state.keys() if final_state else [])}")
        except Exception as invoke_error:
            print(f"ERROR during workflow execution: {invoke_error}")
            traceback.print_exc()
            raise
        
        # This is a critical section - add detailed logging
        print("Processing workflow results")
        if final_state is None:
            print("CRITICAL ERROR: final_state is None")
        else:
            print(f"Final state contains keys: {list(final_state.keys())}")
            for key in final_state.keys():
                if key != "retriever":  # Skip the retriever which could be too large to print
                    value_type = type(final_state[key])
                    value_info = str(final_state[key])[:100] + "..." if isinstance(final_state[key], str) and len(str(final_state[key])) > 100 else final_state[key]
                    print(f"  {key}: ({value_type}) {value_info}")
                    
        # Process results
        await broadcast_message("\n" + "="*50)
        await broadcast_message("--- Workflow Completed ---")
        await broadcast_message("="*50)
        
        if final_state and final_state.get("final_results"):
            print("Has final results - sending to client")
            results_json = json.dumps(final_state["final_results"], indent=2)
            await broadcast_message("\nFinal Analysis Results (JSON):")
            await broadcast_message(results_json)
            
            # Save to file
            output_filename = f"output/analysis_results_{target_phase.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                os.makedirs("output", exist_ok=True)
                with open(output_filename, 'w') as f:
                    json.dump(final_state["final_results"], f, indent=2)
                await broadcast_message(f"\nResults saved to: {output_filename}")
                print(f"Results saved to: {output_filename}")
            except Exception as write_error:
                print(f"Error saving results: {write_error}")
                await broadcast_message(f"\nError saving results to file: {write_error}")
                
        elif final_state and final_state.get("error_message"):
            print(f"Workflow error: {final_state['error_message']}")
            await broadcast_message(f"\nWorkflow finished with an error state: {final_state['error_message']}")
        else:
            print("No final results or error message found")
            await broadcast_message("\n--- Workflow finished, but no final results found in the expected format. ---")
            if final_state:
                print("Logging final state contents to WebSocket")
                await broadcast_message("Final state contents:")
                for key, value in final_state.items():
                    if key != "retriever":  # Skip printing the retriever which could be large
                        await broadcast_message(f"{key}: {value}")
            
        await broadcast_message("\n--- End of workflow ---")
        print("=== FINISHED PROCESSING CLIENT REQUEST ===")
    except Exception as e:
        print(f"\n--- FATAL: Workflow Execution Failed ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        
        await broadcast_message(f"\n--- FATAL: Workflow Execution Failed ---")
        await broadcast_message(f"Error Type: {type(e).__name__}")
        await broadcast_message(f"Error Details: {e}")
        await broadcast_message(f"Traceback: {traceback.format_exc()}")
    finally:
        # Always stop the watchdog
        request_watchdog.stop()

async def websocket_handler(websocket):
    """Handle a websocket connection and process client requests."""
    # Add the connection to our set using the new function
    add_connection(websocket)
    await broadcast_message(f"Client connected. Current connections: {len(get_active_connections())}")
    
    try:
        # Wait for the first message which should contain the run parameters
        initial_message = await websocket.recv()
        params = json.loads(initial_message)
        
        # Process the request
        await process_client_request(websocket, params)
            
    except websockets.exceptions.ConnectionClosed:
        await broadcast_message("Client disconnected unexpectedly")
    except json.JSONDecodeError:
        await broadcast_message("Error: Received invalid JSON")
    except Exception as e:
        await broadcast_message(f"Error processing request: {str(e)}")
        traceback.print_exc()
    finally:
        # Remove the connection when done
        remove_connection(websocket)

# Watchdog class for Windows compatibility
class Watchdog:
    def __init__(self, timeout=300):  # 5 minutes default
        self.timeout = timeout
        self._timer = None
        self._running = False
    
    def start(self):
        if not self._running:
            self._running = True
            self._timer = threading.Timer(self.timeout, self._handle_timeout)
            self._timer.daemon = True  # Allow the program to exit if only the timer is running
            self._timer.start()
            print(f"Watchdog started with {self.timeout} second timeout")
    
    def stop(self):
        if self._running and self._timer:
            self._timer.cancel()
            self._running = False
            print("Watchdog stopped")
    
    def _handle_timeout(self):
        print("\n----- WATCHDOG TIMEOUT -----")
        print(f"WARNING: Operation timed out after {self.timeout} seconds!")
        print("Current thread stack:")
        traceback.print_stack()
        print("All threads stack traces:")
        for th in threading.enumerate():
            print(f"\nThread: {th.name}")
            traceback.print_stack(sys._current_frames()[th.ident])
        print("----- END WATCHDOG TIMEOUT -----\n")
        self._running = False

async def main():
    """Main application entry point."""
    # Create demo files if needed
    create_demo_files()
    
    # Create and start a watchdog (5 minutes timeout)
    watchdog = Watchdog(timeout=300)
    watchdog.start()
    
    try:
        # Start WebSocket server
        server = await websockets.serve(
            websocket_handler, 
            "0.0.0.0", 
            8765
        )
        
        print(f"WebSocket server started on ws://0.0.0.0:8765")
        print("Waiting for client connections...")
        
        # Keep the server running
        await server.wait_closed()
    finally:
        # Stop the watchdog when we're done
        watchdog.stop()

if __name__ == "__main__":
    asyncio.run(main()) 
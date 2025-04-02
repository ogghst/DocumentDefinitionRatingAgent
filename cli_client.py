import asyncio
import json
import requests
import websockets
import argparse
import sys

# --- Configuration ---
BASE_URL = "http://localhost:8765"
WS_URL = "ws://localhost:8765"

async def run_cli_client(checklist_path: str, document_path: str, target_phase: str):
    """Creates a conversation, starts analysis, and handles WebSocket interaction."""

    conversation_id = None
    websocket = None # Define websocket variable in the outer scope

    try:
        # 1. Create Conversation
        print("Creating conversation...")
        create_payload = {
            "title": f"CLI Run - {document_path}",
            "checklist_path": checklist_path,
            "document_path": document_path,
            "target_phase": target_phase
        }
        response = requests.post(f"{BASE_URL}/conversations", json=create_payload)
        response.raise_for_status()  # Raise exception for bad status codes
        conversation_data = response.json()
        conversation_id = conversation_data["id"]
        print(f"Conversation created: {conversation_id}")

        # 2. Connect to WebSocket *before* starting analysis
        ws_uri = f"{WS_URL}/ws/{conversation_id}"
        print(f"Connecting to WebSocket: {ws_uri}")
        websocket = await websockets.connect(ws_uri)
        print("WebSocket connected.")

        # Task to listen for initial connection messages (history, status)
        async def initial_listener():
            try:
                while True: # Listen briefly for initial non-blocking messages
                    message_json = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message = json.loads(message_json)
                    msg_type = message.get("type", "unknown")
                    content = message.get("content", "")
                    if msg_type == "connection_status":
                         print(f"[STATUS] {content}")
                    elif msg_type != "rag_token": # Avoid printing history tokens
                        print(f"[{msg_type.upper()}] {content}")
            except asyncio.TimeoutError:
                 print("Initial messages received (or timeout).")
            except Exception as e:
                print(f"Error during initial listen: {e}")


        # Run initial listener briefly
        await initial_listener()

        # 3. Start Analysis
        print("Starting RAG analysis...")
        start_response = requests.post(f"{BASE_URL}/conversations/{conversation_id}/start")
        try:
            start_response.raise_for_status()
            print("Analysis started successfully.")
        except requests.exceptions.HTTPError as e:
            print(f"Error starting analysis: {e.response.status_code} - {e.response.text}")
            # Optionally try to delete conversation or just exit
            # requests.delete(f"{BASE_URL}/conversations/{conversation_id}")
            return # Exit if analysis couldn't start


        # 4. Main Interaction Loop
        print("\n--- Waiting for Server Messages ---")
        while True:
            try:
                message_json = await websocket.recv()
                message = json.loads(message_json)

                msg_type = message.get("type", "unknown")
                content = message.get("content", "")
                timestamp = message.get("timestamp", "") # Optional timestamp

                # Handle different message types
                if msg_type == "input_request":
                    print(f"[INPUT REQUIRED] {content}")
                    # Use asyncio.to_thread to run input() in a separate thread
                    # This prevents blocking the asyncio event loop
                    user_input = await asyncio.to_thread(input, "Your response: ")
                    response_payload = json.dumps({
                        "type": "input_response",
                        "content": user_input.strip()
                    })
                    await websocket.send(response_payload)
                    print("[RESPONSE SENT]")

                elif msg_type == "rag_token":
                    # Print tokens without newline for progress indication
                    print(content, end='', flush=True)

                elif msg_type == "rag_result":
                    print("--- RAG Result ---")
                    try:
                        # Try pretty printing if content is JSON string
                        result_data = json.loads(content)
                        print(json.dumps(result_data, indent=2))
                    except json.JSONDecodeError:
                        print(content) # Print as raw text if not JSON
                    print("------------------")
                    print("Analysis complete. Closing connection.")
                    break # Exit loop after receiving final result

                elif msg_type == "error":
                    print(f"[ERROR] {content}")
                    # Decide if we should break on error
                    # break

                elif msg_type == "warning":
                     print(f"[WARNING] {content}")

                elif msg_type == "system":
                     print(f"[SYSTEM] {content}")

                elif msg_type == "rag_progress":
                     print(f"[PROGRESS] {content}")

                elif msg_type == "user":
                     # This script doesn't send user messages, but might receive them
                     # if sent via API or another client
                     print(f"[USER MSG] {content}")

                elif msg_type == "connection_status":
                    # Usually received only at the start, but handle anyway
                    print(f"[STATUS] {content}")

                else:
                    # Catch-all for unknown types
                    print(f"[{msg_type.upper()}] {content}")


            except websockets.exceptions.ConnectionClosedOK:
                print("Server closed the connection normally.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed with error: {e}")
                break
            except json.JSONDecodeError:
                print(f"Received non-JSON message: {message_json}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                import traceback
                traceback.print_exc()
                break

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket connection failed: {e}")
    except Exception as e:
        print(f"An error occurred in the client script: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if websocket and not websocket.closed:
            print("Closing WebSocket connection...")
            await websocket.close()
        # Optional: Delete conversation on exit/error?
        # if conversation_id:
        #     print(f"Deleting conversation {conversation_id}...")
        #     try:
        #         requests.delete(f"{BASE_URL}/conversations/{conversation_id}")
        #     except requests.exceptions.RequestException as del_e:
        #         print(f"Failed to delete conversation: {del_e}")
        print("Client finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG workflow via CLI.")
    parser.add_argument(
        "--checklist",
        default="checklists/project_checklist_demo.xlsx",
        help="Path to the checklist file."
    )
    parser.add_argument(
        "--document",
        default="input/quotation_demo.docx",
        help="Path to the document file."
    )
    parser.add_argument(
        "--phase",
        default="Apertura Commessa",
        help="Target phase for the analysis."
    )
    args = parser.parse_args()

    # Basic check if server is reachable (optional but helpful)
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
         print(f"Error: Could not connect to the server at {BASE_URL}.")
         print("Please ensure the websocket_server.py is running.")
         sys.exit(1)

    try:
        asyncio.run(run_cli_client(args.checklist, args.document, args.phase))
    except KeyboardInterrupt:
        print("Exiting due to user interrupt.") 
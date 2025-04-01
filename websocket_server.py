import asyncio
import json
import uuid
from typing import Dict, Set, Optional, List, Any, Callable, Coroutine
import contextvars
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import common utilities carefully
# from common import queue_message, get_user_input # Keep get_user_input temporarily, might be replaced
# Remove direct dependency on common.send_message and active_connections_var
# from common import send_message, get_user_input, active_connections_var, queue_message
from websocket_callbacks import WebsocketCallbackManager

# RAG imports - IMPORTANT: only import these when needed to avoid circular import
from models import GraphState

# Create FastAPI app
app = FastAPI(title="RAG Conversation Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
class Conversation:
    def __init__(self, id: str, title: str):
        self.id = id
        self.title = title
        self.created_at = datetime.now()
        self.connections: Set[WebSocket] = set()
        self.owner_connection: Optional[WebSocket] = None  # Track owner connection
        self.messages: List[Dict[str, Any]] = []
        self.is_active = False
        self.workflow_result = None
        self.checklist_path = ""
        self.document_path = ""
        self.target_phase = ""
        # New attributes for queued message sending and input handling
        self.outgoing_queue: asyncio.Queue = asyncio.Queue()
        self.sender_task: Optional[asyncio.Task] = None
        self.input_request_event: asyncio.Event = asyncio.Event()
        self.input_response: Optional[str] = None

# Store active conversations
conversations: Dict[str, Conversation] = {}

# Remove dictionary for active callback managers - they are now transient within run_rag_workflow
# conversation_callbacks: Dict[str, WebsocketCallbackManager] = {}

# ---- Models for API requests ----
class ConversationRequest(BaseModel):
    title: str
    checklist_path: str = "checklists/project_checklist_demo.xlsx"
    document_path: str = "input/quotation_demo.docx"
    target_phase: str = "Apertura Commessa"

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    is_active: bool

class MessageRequest(BaseModel):
    content: str

# ---- Internal Message Sender Task ----
async def _message_sender_task(conversation_id: str):
    """Reads messages from the conversation's queue and sends them to the owner."""
    print(f"Starting sender task for conversation {conversation_id}")
    conversation = conversations.get(conversation_id)
    if not conversation:
        print(f"Sender task exiting: Conversation {conversation_id} not found.")
        return

    while True:
        try:
            # Wait for a message to appear in the queue
            message_data = await conversation.outgoing_queue.get()

            if message_data is None: # Sentinel value to stop the task
                print(f"Sender task for {conversation_id} received stop signal.")
                break

            # Ensure owner connection is still valid
            owner_ws = conversation.owner_connection
            if not owner_ws:
                print(f"Sender task for {conversation_id}: No owner connection, discarding message: {message_data.get('type')}")
                conversation.outgoing_queue.task_done()
                continue # Skip sending if no owner

            try:
                json_message = json.dumps(message_data)
                await owner_ws.send_text(json_message)
                # Only log non-token messages to console to reduce noise
                if message_data.get("type") != "rag_token":
                     print(f"[SENT][{conversation_id}] Type: {message_data.get('type')}, Content: {str(message_data.get('content'))[:50]}...")
                #else:
                #     print(f"[SENT][{conversation_id}] Type: rag_token") # Indicate token sent
            except WebSocketDisconnect:
                print(f"Sender task for {conversation_id}: Owner disconnected.")
                # Don't break the loop, owner might reconnect via websocket_endpoint logic
            except Exception as e:
                print(f"Sender task for {conversation_id}: Error sending message: {e}")
                # Potentially add message back to queue or handle error differently

            conversation.outgoing_queue.task_done()

        except asyncio.CancelledError:
            print(f"Sender task for {conversation_id} cancelled.")
            break
        except Exception as e:
            print(f"Unexpected error in sender task for {conversation_id}: {e}")
            # Avoid breaking the loop on unexpected errors if possible
            await asyncio.sleep(1) # Prevent tight loop on errors

    print(f"Sender task for conversation {conversation_id} stopped.")


# ---- Connection Management ----
# Remove global active_connections management - it's per-conversation now
# def get_active_connections() -> Set[WebSocket]:
#     return active_connections_var.get()

def add_connection(websocket: WebSocket, conversation_id: str, is_owner: bool = False):
    """Add WebSocket connection to a conversation and manage sender task."""
    if conversation_id not in conversations:
        print(f"Cannot add connection: Conversation {conversation_id} not found.")
        # Optionally close the websocket here
        # await websocket.close(code=1008)
        return

    conv = conversations[conversation_id]
    conv.connections.add(websocket)

    if is_owner or not conv.owner_connection: # Make first connection the owner
        print(f"Setting connection as owner for conversation {conversation_id}")
        conv.owner_connection = websocket
        # Start sender task if not already running for this conversation
        if conv.sender_task is None or conv.sender_task.done():
            conv.sender_task = asyncio.create_task(_message_sender_task(conversation_id))
            print(f"Started sender task for conversation {conversation_id}")

    print(f"Added connection to conversation {conversation_id}. Owner: {conv.owner_connection is not None}, Total connections: {len(conv.connections)}")

def remove_connection(websocket: WebSocket, conversation_id: str):
    """Remove WebSocket connection and manage sender task."""
    if conversation_id not in conversations:
        print(f"Cannot remove connection: Conversation {conversation_id} not found.")
        return

    conv = conversations[conversation_id]
    if websocket in conv.connections:
        conv.connections.remove(websocket)
        print(f"Removed connection from conversation {conversation_id}. Remaining: {len(conv.connections)}")

        if websocket == conv.owner_connection:
            print(f"Owner connection removed for conversation {conversation_id}.")
            conv.owner_connection = None
            # Stop the sender task if the owner leaves
            if conv.sender_task and not conv.sender_task.done():
                print(f"Stopping sender task for conversation {conversation_id} as owner left.")
                conv.sender_task.cancel()
                # Optionally await task cancellation if needed, but might block
                conv.sender_task = None

            # Optionally assign a new owner if others are connected
            # if conv.connections:
            #     new_owner = next(iter(conv.connections))
            #     conv.owner_connection = new_owner
            #     print(f"Assigned new owner for conversation {conversation_id}")
            #     # Restart sender task for the new owner
            #     if conv.sender_task is None or conv.sender_task.done():
            #          conv.sender_task = asyncio.create_task(_message_sender_task(conversation_id))
            #          print(f"Restarted sender task for new owner of {conversation_id}")

        if not conv.connections and conv.sender_task and not conv.sender_task.done():
             # Safety check: stop sender if no connections remain
             print(f"Stopping sender task for {conversation_id} as no connections remain.")
             conv.sender_task.cancel()
             conv.sender_task = None


# ---- Message Broadcasting / Queueing ----
async def send_message(conversation_id: str, message: str, message_type: str = "text"):
    """Queue a message to be sent to the owner of a conversation."""
    if conversation_id not in conversations:
        print(f"Conversation {conversation_id} not found, cannot queue message.")
        return

    conv = conversations[conversation_id]

    # Prepare message data
    message_data = {
        "type": message_type,
        "content": message,
        "timestamp": datetime.now().isoformat()
    }

    # Add to history (except tokens) before queueing
    if message_type != "rag_token":
        conv.messages.append(message_data)

    # Put the message onto the conversation's outgoing queue
    await conv.outgoing_queue.put(message_data)
    # Simple console log indicating queueing
    # print(f"[QUEUED][{conversation_id}] Type: {message_type}, Content: {message[:50]}...")


# ---- RAG Workflow Integration ----
async def run_rag_workflow(conversation_id: str, background_tasks: BackgroundTasks):
    """Run the RAG workflow for a conversation, using queueing for messages."""
    # Import here to avoid circular imports
    from graph_nodes import create_workflow_graph
    from langchain_core.runnables import RunnableConfig

    try:
        if conversation_id not in conversations:
            print(f"Cannot run workflow: Conversation {conversation_id} not found.")
            return
        conversation = conversations[conversation_id]
        conversation.is_active = True

        # Ensure we have an owner connection to receive messages (checked by sender task)
        if not conversation.owner_connection:
            print(f"No owner connection for conversation {conversation_id}, workflow cannot proceed effectively.")
            # Send status message might fail, but try anyway
            await send_message(conversation_id, "Error: Cannot start analysis without an active owner connection.", "error")
            conversation.is_active = False
            return

        # Send initial status message via the queue
        await send_message(conversation_id, "Starting RAG analysis...", "system")

        # Define the callback function to pass to the manager
        async def send_callback(msg_content: str, msg_type: str = "rag_progress"):
            await send_message(conversation_id, msg_content, msg_type)

        # Define the input request function to pass to the manager
        async def request_input_callback(prompt: str) -> str:
            return await request_user_input(conversation_id, prompt)

        # Create the callback manager, passing the *send_callback* and *request_input_callback*
        # NO websocket is passed directly anymore.
        callback_manager = WebsocketCallbackManager(
            conversation_id=conversation_id,
            send_callback=send_callback,
            request_input_callback=request_input_callback # Add this
        )
        # No need to store callback managers globally anymore
        # conversation_callbacks[conversation_id] = callback_manager

        # Tell the client we'll be streaming tokens via the queue
        await send_message(conversation_id, "Token streaming enabled.", "system")

        # Create workflow graph
        app_graph = create_workflow_graph()

        # Create input state, passing the callback manager itself
        # The graph nodes will access the manager via the state if needed,
        # or more likely, directly use the send_callback/request_input_callback
        # passed via RunnableConfig's metadata or state['callback_manager'].
        inputs = GraphState(
            checklist_path=conversation.checklist_path,
            document_path=conversation.document_path,
            target_phase=conversation.target_phase,
            callback_manager=callback_manager, # Keep manager in state for non-mapped nodes if needed
            conversation_id=conversation_id,
            # Pass callbacks explicitly for use in mapped nodes
            send_callback_func=send_callback,
            request_input_func=request_input_callback,
        )

        # Configure and run graph with proper callback handler
        # The manager itself acts as the handler list
        config = RunnableConfig(
            recursion_limit=25,
            callbacks=[callback_manager], # Manager handles callbacks for the overall run
            # Configurable/Metadata might not be needed at top level if passed via state
            # configurable={ ... },
            metadata={
                "conversation_id": conversation_id
                # Remove callbacks from metadata here
            }
        )

        # Run the workflow
        final_state = await app_graph.ainvoke(inputs, config=config)

        # Store the result
        conversation.workflow_result = final_state

        # Send the final result via the queue
        if final_state and final_state.get("final_results"):
            # Ensure results are JSON serializable before sending
            try:
                results_content = json.dumps(final_state["final_results"])
                await send_message(conversation_id, results_content, "rag_result")
            except TypeError as e:
                 print(f"Error serializing final_results: {e}")
                 await send_message(conversation_id, f"Error displaying results: {e}", "error")

        elif final_state and final_state.get("error_message"):
            await send_message(conversation_id, f"Error: {final_state['error_message']}", "error")
        else:
            await send_message(conversation_id, "Analysis completed but no results were found", "warning")

    except Exception as e:
        print(f"Error running RAG workflow for {conversation_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Try to send error message via the queue
        try:
            await send_message(conversation_id, f"Critical Error running RAG workflow: {str(e)}", "error")
        except Exception as send_error:
            print(f"Error sending workflow error message: {send_error}")
    finally:
        # Mark conversation as inactive
        if conversation_id in conversations:
            conversations[conversation_id].is_active = False
        # No callback manager cleanup needed here as it's not stored globally

# ---- User Input Handling ----
async def request_user_input(conversation_id: str, prompt: str) -> str:
    """Requests input from the user via WebSocket and waits for the response."""
    if conversation_id not in conversations:
        print(f"Cannot request input: Conversation {conversation_id} not found.")
        return "error: conversation not found"

    conv = conversations[conversation_id]

    # Reset event and response holder
    conv.input_request_event.clear()
    conv.input_response = None

    # Send the input request message via the queue
    await send_message(conversation_id, prompt, "input_request")

    print(f"Waiting for user input for conversation {conversation_id}...")
    try:
        # Wait for the event to be set by the websocket handler
        await asyncio.wait_for(conv.input_request_event.wait(), timeout=300) # 5 minute timeout
        response = conv.input_response
        print(f"Received input for {conversation_id}: {response}")
        return response if response is not None else "error: no response received"
    except asyncio.TimeoutError:
        print(f"Timeout waiting for user input for conversation {conversation_id}")
        await send_message(conversation_id, "Timeout waiting for user input. Proceeding...", "warning")
        return "timeout"
    except Exception as e:
         print(f"Error waiting for user input event for {conversation_id}: {e}")
         return f"error: {e}"
    finally:
        # Ensure event is clear for next request (though it should be)
        conv.input_request_event.clear()


# ---- API Routes ----
@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"status": "online", "message": "RAG Conversation Server is running"}

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())

    # Create conversation object (queue and event created in __init__)
    conversation = Conversation(id=conversation_id, title=request.title)
    conversation.checklist_path = request.checklist_path
    conversation.document_path = request.document_path
    conversation.target_phase = request.target_phase

    # Store in our dictionary
    conversations[conversation_id] = conversation

    print(f"Created conversation {conversation_id}")

    return {
        "id": conversation_id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "is_active": conversation.is_active
    }

@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations():
    """List all conversations."""
    return [
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "is_active": conv.is_active
        }
        for conv in conversations.values()
    ]

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get conversation details."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "is_active": conv.is_active
    }

@app.post("/conversations/{conversation_id}/start")
async def start_conversation(conversation_id: str, background_tasks: BackgroundTasks):
    """Start RAG analysis for a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id] # Get conv object

    if conv.is_active:
        raise HTTPException(status_code=400, detail="Conversation is already active")

    # Ensure owner is connected before starting
    if not conv.owner_connection:
         raise HTTPException(status_code=400, detail="Owner websocket must be connected to start analysis.")

    # Start the RAG workflow in the background
    # Pass background_tasks if needed by run_rag_workflow itself (currently not used by it)
    background_tasks.add_task(run_rag_workflow, conversation_id, background_tasks)

    return {"status": "started", "conversation_id": conversation_id}

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get all messages in a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversations[conversation_id].messages

@app.post("/conversations/{conversation_id}/messages")
async def post_message(conversation_id: str, message: MessageRequest): # Renamed from send_message to avoid conflict
    """Send a chat message from a user to a conversation (broadcast TO owner for now)."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Use the queueing send_message function
    # This will add it to history and queue it for the sender task
    await send_message(conversation_id, message.content, "user")

    # Note: This currently sends the user message *only* to the owner via the queue.
    # If broadcasting to all connected clients is needed for chat,
    # _message_sender_task or send_message logic needs adjustment.

    return {"status": "queued", "conversation_id": conversation_id}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation, close connections, and stop tasks."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = conversations[conversation_id]

    # Stop sender task first
    if conv.sender_task and not conv.sender_task.done():
        print(f"Stopping sender task for deleted conversation {conversation_id}")
        # Put sentinel value to ask it to stop gracefully
        await conv.outgoing_queue.put(None)
        try:
             await asyncio.wait_for(conv.sender_task, timeout=5.0)
        except asyncio.TimeoutError:
             print(f"Sender task for {conversation_id} did not stop gracefully, cancelling.")
             conv.sender_task.cancel()
        except Exception as e:
             print(f"Error waiting for sender task {conversation_id} to stop: {e}")
             conv.sender_task.cancel() # Force cancel on other errors

    # Close all WebSocket connections
    print(f"Closing connections for deleted conversation {conversation_id}")
    for connection in list(conv.connections): # Iterate over a copy
        try:
            await connection.close(code=1000, reason="Conversation deleted")
        except Exception as e:
            print(f"Error closing connection during delete: {e}")

    # Remove from our dictionary
    del conversations[conversation_id]
    print(f"Deleted conversation {conversation_id}")

    return {"status": "deleted", "conversation_id": conversation_id}

# ---- WebSocket Routes ----
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for a conversation."""
    # Check if conversation exists *before* accepting
    if conversation_id not in conversations:
        print(f"Rejecting websocket: Conversation {conversation_id} not found.")
        # Don't accept, just return. Client should handle refused connection.
        # await websocket.close(code=1008, reason="Conversation not found")
        return

    conversation = conversations[conversation_id]

    # Accept the connection
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted for conversation {conversation_id}")
    except Exception as e:
        print(f"Failed to accept WebSocket connection for {conversation_id}: {e}")
        return

    # Determine if this connection should be the owner
    # Owner is the first connection, or if the current owner is None
    is_owner = not conversation.owner_connection or len(conversation.connections) == 0

    # Add connection to the conversation (this also handles starting the sender task if needed)
    add_connection(websocket, conversation_id, is_owner=is_owner)

    # Send connection status and history via the queue
    status_message = "Connected as owner" if conversation.owner_connection == websocket else "Connected as viewer"
    await send_message(conversation_id, status_message, "connection_status")
    # Optionally send is_owner flag if client needs it explicitly beyond the message content
    # await send_message(conversation_id, json.dumps({"is_owner": is_owner}), "connection_info") # Example

    print(f"Sending history ({len(conversation.messages)} messages) to new connection for {conversation_id}")
    for message in conversation.messages:
        # We queue history messages too, sender task will handle sending
        await conversation.outgoing_queue.put(message) # Use put directly to bypass history append

    # If this new connection *is* the owner, ensure the sender task is running
    # (add_connection should handle this, but double-check could be added)
    if conversation.owner_connection == websocket and (conversation.sender_task is None or conversation.sender_task.done()):
         print(f"WS Endpoint: Re-checking sender task for owner {conversation_id}")
         conversation.sender_task = asyncio.create_task(_message_sender_task(conversation_id))


    try:
        # Main message loop for this connection
        while True:
            message_text = await websocket.receive_text()
            print(f"[RECEIVED][{conversation_id}] Raw: {message_text[:100]}")

            try:
                data = json.loads(message_text)
                if isinstance(data, dict) and "type" in data:
                    # Handle specific message types
                    if data["type"] == "input_response" and "content" in data:
                        # Check if this message is from the designated owner
                        if websocket == conversation.owner_connection:
                            print(f"Received input response for {conversation_id}")
                            conversation.input_response = data["content"]
                            conversation.input_request_event.set() # Signal waiting coroutine
                        else:
                             print(f"Ignoring input_response from non-owner for {conversation_id}")
                    elif data["type"] == "user" and "content" in data:
                        # User chat message - queue it to be sent (currently only to owner)
                        print(f"Received user chat message for {conversation_id}")
                        await send_message(conversation_id, data["content"], "user")
                    else:
                        # Unknown structured message type
                        print(f"Received unknown message type for {conversation_id}: {data['type']}")
                        # Treat as chat message?
                        # await send_message(conversation_id, message_text, "user_unknown")

                else:
                    # Unstructured JSON or non-dict JSON - treat as chat
                    print(f"Received unstructured JSON for {conversation_id} - treating as chat.")
                    await send_message(conversation_id, message_text, "user")

            except json.JSONDecodeError:
                # Plain text message - treat as chat
                print(f"Received plain text for {conversation_id} - treating as chat.")
                await send_message(conversation_id, message_text, "user")
            except Exception as e:
                 print(f"Error processing received message for {conversation_id}: {e}")


    except WebSocketDisconnect:
        print(f"WebSocket disconnected from conversation {conversation_id}")
    except Exception as e:
        print(f"Error in WebSocket handler for {conversation_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure connection is removed and sender task potentially stopped
        print(f"Cleaning up connection for {conversation_id}")
        remove_connection(websocket, conversation_id)
        # WebSocket should be closed automatically by FastAPI/Starlette on disconnect or error
        # try:
        #     await websocket.close()
        # except:
        #     pass # Already closed or error

# Utility to run the server
def run_server():
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    # Reload disabled for stability with background tasks
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)

if __name__ == "__main__":
    try:
        print("Starting RAG Conversation Server...")
        # Remove circular import check here - imports happen within functions
        # import graph_nodes
        run_server()
    except ImportError as e:
        print(f"Import error detected: {e}")
        # This might indicate issues beyond circular imports now
    except Exception as e:
        print(f"Error starting server: {e}") 
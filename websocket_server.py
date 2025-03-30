import asyncio
import json
import uuid
from typing import Dict, Set, Optional, List, Any
import contextvars
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import from common module instead
from common import broadcast_message, get_user_input, active_connections_var, queue_message
from websocket_callbacks import WebsocketCallbackManager  # Add this import

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

# Store active conversations
conversations: Dict[str, Conversation] = {}

# Dictionary to store active callback managers
conversation_callbacks: Dict[str, WebsocketCallbackManager] = {}

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

# ---- Connection Management ----
def get_active_connections() -> Set[WebSocket]:
    return active_connections_var.get()

def add_connection(websocket: WebSocket, conversation_id: str, is_owner: bool = False):
    """Add WebSocket connection to a conversation."""
    # Add to global connections
    connections = active_connections_var.get()
    connections.add(websocket)
    active_connections_var.set(connections)
    
    # Add to conversation
    if conversation_id in conversations:
        conversations[conversation_id].connections.add(websocket)
        
        # Set as owner if specified or if first connection
        if is_owner or len(conversations[conversation_id].connections) == 1:
            conversations[conversation_id].owner_connection = websocket
            print(f"Set connection as owner for conversation {conversation_id}")
            
        print(f"Added connection to conversation {conversation_id}. Total connections: {len(conversations[conversation_id].connections)}")

def remove_connection(websocket: WebSocket, conversation_id: str):
    """Remove WebSocket connection from a conversation."""
    # Remove from global connections
    connections = active_connections_var.get()
    if websocket in connections:
        connections.remove(websocket)
        active_connections_var.set(connections)
    
    # Remove from conversation
    if conversation_id in conversations:
        conv = conversations[conversation_id]
        if websocket in conv.connections:
            conv.connections.remove(websocket)
            print(f"Removed connection from conversation {conversation_id}. Remaining: {len(conv.connections)}")
            
            # If we removed the owner, assign a new owner if connections remain
            if websocket == conv.owner_connection and conv.connections:
                conv.owner_connection = next(iter(conv.connections))
                print(f"Assigned new owner for conversation {conversation_id}")
            elif not conv.connections:
                # No connections left, clear owner
                conv.owner_connection = None

# ---- Message Broadcasting ----
async def broadcast_to_conversation(conversation_id: str, message: str, message_type: str = "text", rag_message: bool = False):
    """Send a message to clients in a conversation.
    
    Args:
        conversation_id: ID of the conversation
        message: The message content
        message_type: Type of message (text, system, etc.)
        rag_message: If True, send only to the owner (for RAG analysis messages)
    """
    # Simplified console logging
    if message_type == "rag_token":
        print(f"[TOKEN] {message}")
    else:
        print(f"[BROADCAST] {conversation_id}: {message_type} - {message[:50]}...")
    
    # Early return if conversation doesn't exist
    if conversation_id not in conversations:
        print(f"Conversation {conversation_id} not found")
        return
    
    conv = conversations[conversation_id]
    
    # Prepare message data
    message_data = {
        "type": message_type,
        "content": message,
        "timestamp": datetime.now().isoformat()
    }
    
    # Only store non-token messages to avoid crowding the history
    if message_type != "rag_token":
        conv.messages.append(message_data)
    
    # Format as JSON once
    json_message = json.dumps(message_data)
    
    # Target connection(s)
    target_connection = None
    if rag_message and conv.owner_connection:
        # For RAG messages, only send to the owner
        target_connection = conv.owner_connection
    else:
        # For other messages, send to all connections
        for connection in conv.connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                print(f"Error sending to connection: {e}")
    
    # Send to owner for RAG messages
    if target_connection:
        try:
            await target_connection.send_text(json_message)
        except Exception as e:
            print(f"Error sending to owner: {e}")

# ---- RAG Workflow Integration ----
async def run_rag_workflow(conversation_id: str, background_tasks: BackgroundTasks):
    """Run the RAG workflow for a conversation."""
    # Import here to avoid circular imports
    from graph_nodes import create_workflow_graph
    from langchain_core.runnables import RunnableConfig
    
    try:
        conversation = conversations[conversation_id]
        conversation.is_active = True
        
        # Ensure we have an owner connection
        if not conversation.owner_connection:
            print(f"No owner connection for conversation {conversation_id}")
            return
            
        # Send initial message directly to owner websocket
        status_message = {
            "type": "system",
            "content": "Starting RAG analysis...",
            "timestamp": datetime.now().isoformat()
        }
        await conversation.owner_connection.send_text(json.dumps(status_message))
        
        # Create and store the callback manager for this conversation
        # Pass the owner's websocket connection directly
        callback_manager = WebsocketCallbackManager(
            conversation_id=conversation_id,
            websocket=conversation.owner_connection
        )
        conversation_callbacks[conversation_id] = callback_manager
        
        # Create workflow graph
        app = create_workflow_graph()
        
        # Create input state
        inputs = GraphState(
            checklist_path=conversation.checklist_path,
            document_path=conversation.document_path,
            target_phase=conversation.target_phase
        )
        
        # Configure and run graph with proper callback handler
        config = RunnableConfig(
            recursion_limit=25,
            callbacks=[callback_manager],
            metadata={"conversation_id": conversation_id}  # Include conversation_id in metadata
        )
        
        # Run the workflow
        final_state = await app.ainvoke(inputs, config=config)
        
        # Store the result
        conversation.workflow_result = final_state
        
        # Send the final result directly to the owner
        if final_state and final_state.get("final_results"):
            results_message = {
                "type": "rag_result",
                "content": json.dumps(final_state["final_results"]),
                "timestamp": datetime.now().isoformat()
            }
            await conversation.owner_connection.send_text(json.dumps(results_message))
        elif final_state and final_state.get("error_message"):
            error_message = {
                "type": "error",
                "content": f"Error: {final_state['error_message']}",
                "timestamp": datetime.now().isoformat()
            }
            await conversation.owner_connection.send_text(json.dumps(error_message))
        else:
            warning_message = {
                "type": "warning",
                "content": "Analysis completed but no results were found",
                "timestamp": datetime.now().isoformat()
            }
            await conversation.owner_connection.send_text(json.dumps(warning_message))
    except Exception as e:
        print(f"Error running RAG workflow: {str(e)}")
        # Try to send error message if we have a valid connection
        if conversation_id in conversations and conversations[conversation_id].owner_connection:
            try:
                error_message = {
                    "type": "error",
                    "content": f"Error running RAG workflow: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                await conversations[conversation_id].owner_connection.send_text(json.dumps(error_message))
            except Exception as send_error:
                print(f"Error sending error message: {send_error}")
    finally:
        # Mark conversation as inactive
        if conversation_id in conversations:
            conversations[conversation_id].is_active = False
        
        # Deactivate and clean up callback manager
        if conversation_id in conversation_callbacks:
            conversation_callbacks[conversation_id].deactivate()
            conversation_callbacks.pop(conversation_id, None)

# Update get_user_input to use the callback manager if available
async def get_user_input(conversation_id: str, prompt: str) -> str:
    """Get input from the owner of a conversation.
    
    This function sends a prompt to the conversation owner and waits for a response.
    The RAG workflow will pause here until a response is received or the timeout occurs.
    """
    # If we have a callback manager for this conversation, use it
    if conversation_id in conversation_callbacks:
        return await conversation_callbacks[conversation_id].get_user_input(prompt)
    
    # Otherwise, fall back to the original implementation
    if conversation_id not in conversations:
        print(f"Conversation {conversation_id} not found")
        return "skip"
    
    conv = conversations[conversation_id]
    
    if not conv.owner_connection:
        print(f"No owner connection for conversation {conversation_id}")
        return "skip"
    
    # First, notify the client that we're pausing for input
    await broadcast_to_conversation(
        conversation_id, 
        "RAG analysis paused: Waiting for human input...", 
        "rag_progress", 
        rag_message=True
    )
    
    # Send prompt to the owner only
    await broadcast_to_conversation(conversation_id, prompt, "input_request", rag_message=True)
    
    # Create an event to wait for the response
    input_received = asyncio.Event()
    user_response = ["skip"]  # Use list for mutable reference
    
    # Set up a response handler
    async def handle_owner_message(websocket: WebSocket):
        try:
            # Wait for a message from the owner
            message = await websocket.receive_text()
            print(f"Received response from owner of conversation {conversation_id}: {message[:50]}...")
            
            try:
                # Try to parse as JSON
                data = json.loads(message)
                if isinstance(data, dict) and "content" in data:
                    user_response[0] = data["content"]
                    print(f"Parsed input response content: {data['content'][:50]}...")
                else:
                    user_response[0] = message
                    print(f"Using message as is: {message[:50]}...")
            except json.JSONDecodeError:
                user_response[0] = message
                print(f"Using raw message as response: {message[:50]}...")
            
            # Signal that we've received input
            input_received.set()
        except WebSocketDisconnect:
            # Owner disconnected
            print(f"Owner disconnected while waiting for input in conversation {conversation_id}")
            input_received.set()  # Signal to continue with default
        except Exception as e:
            # Other error
            print(f"Error receiving owner message: {e}")
            input_received.set()  # Signal to continue with default
    
    # Start listening for the owner's response
    task = asyncio.create_task(handle_owner_message(conv.owner_connection))
    task.set_name(f"wait_for_input_{conversation_id}")
    
    # Notify user we're waiting for input
    print(f"Waiting for human input in conversation {conversation_id}...")
    
    try:
        # Wait for the response with a timeout
        await asyncio.wait_for(input_received.wait(), timeout=120)  # 2-minute timeout
        
        # Notify that input was received
        if user_response[0] != "skip":
            await broadcast_to_conversation(
                conversation_id, 
                "Input received. Resuming RAG analysis...", 
                "rag_progress", 
                rag_message=True
            )
            print(f"Human input received: {user_response[0][:50]}...")
        else:
            await broadcast_to_conversation(
                conversation_id, 
                "Input skipped. Resuming RAG analysis...", 
                "rag_progress", 
                rag_message=True
            )
            print("Human input skipped.")
        
        # Return the response
        return user_response[0]
    except asyncio.TimeoutError:
        # Timeout occurred
        print(f"Timeout waiting for human input in conversation {conversation_id}")
        await broadcast_to_conversation(
            conversation_id, 
            "Timeout waiting for input. Resuming RAG analysis...", 
            "rag_progress", 
            rag_message=True
        )
        return "skip"
    finally:
        # Clean up the task if it's still running
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

# ---- API Routes ----
@app.get("/")
async def root():
    """Root endpoint to verify the API is running."""
    return {"status": "online", "message": "RAG Conversation Server is running"}

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    
    # Create conversation object
    conversation = Conversation(id=conversation_id, title=request.title)
    conversation.checklist_path = request.checklist_path
    conversation.document_path = request.document_path
    conversation.target_phase = request.target_phase
    
    # Store in our dictionary
    conversations[conversation_id] = conversation
    
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
    
    if conversations[conversation_id].is_active:
        raise HTTPException(status_code=400, detail="Conversation is already active")
    
    # Start the RAG workflow in the background
    background_tasks.add_task(run_rag_workflow, conversation_id, background_tasks)
    
    return {"status": "started", "conversation_id": conversation_id}

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get all messages in a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversations[conversation_id].messages

@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, message: MessageRequest):
    """Send a message to a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    await broadcast_to_conversation(conversation_id, message.content, "user")
    
    return {"status": "sent", "conversation_id": conversation_id}

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Close all connections
    for connection in list(conversations[conversation_id].connections):
        try:
            await connection.close()
        except:
            pass
    
    # Remove from our dictionary
    del conversations[conversation_id]
    
    return {"status": "deleted", "conversation_id": conversation_id}

# ---- WebSocket Routes ----
@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for a conversation."""
    # Early return if conversation doesn't exist
    if conversation_id not in conversations:
        await websocket.close(code=1008, reason="Conversation not found")
        return
    
    # Try to optimize the connection - this is optional and may not work on all WebSocket implementations
    try:
        # Different WebSocket implementations may have different attribute structures
        # Try several common patterns
        if hasattr(websocket, '_transport') and hasattr(websocket._transport, '_sock'):
            # FastAPI/Starlette WebSocket with direct transport access
            socket = websocket._transport._sock
            socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("TCP_NODELAY enabled via _transport._sock")
        elif hasattr(websocket, 'raw_socket'):
            # Direct socket access
            socket = websocket.raw_socket
            socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("TCP_NODELAY enabled via raw_socket")
        elif hasattr(websocket, 'socket'):
            # Alternative direct socket access
            socket = websocket.socket
            socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("TCP_NODELAY enabled via socket")
        else:
            # No known socket access method - skip optimization
            print("TCP_NODELAY not enabled - no known socket access method")
    except Exception as e:
        # Non-critical, so just log and continue if this fails
        print(f"Note: Could not enable TCP_NODELAY on WebSocket: {e}")
    
    # Accept the connection
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted for conversation {conversation_id}")
    except Exception as e:
        print(f"Failed to accept WebSocket connection: {e}")
        return
    
    # Determine if this is the owner (first connection = owner)
    is_owner = not conversations[conversation_id].connections
    
    # Add to our conversation
    add_connection(websocket, conversation_id, is_owner)
    
    try:
        # Notify about connection status
        status_message = "Connected as owner" if is_owner else "Connected as viewer"
        await websocket.send_text(json.dumps({
            "type": "connection_status",
            "content": status_message,
            "timestamp": datetime.now().isoformat(),
            "is_owner": is_owner
        }))
        
        # Send conversation history
        for message in conversations[conversation_id].messages:
            await websocket.send_text(json.dumps(message))
        
        # If this is the owner, update the websocket reference in any existing callback manager
        if is_owner and conversation_id in conversation_callbacks:
            conversation_callbacks[conversation_id].set_websocket(websocket)
        
        # Main message loop
        while True:
            message = await websocket.receive_text()
            
            # If this is the owner and there's an active callback manager waiting for input,
            # forward the message to the callback manager
            if is_owner and conversation_id in conversation_callbacks:
                callback_manager = conversation_callbacks[conversation_id]
                callback_manager.handle_user_message(message)
            
            # Try to queue the message for any waiting input requests
            message_processed = queue_message(websocket, message)
            
            # If the message was processed as an input response, we're done
            if message_processed:
                continue
            
            # Process the message for the chat functionality
            try:
                # Try to parse as JSON
                data = json.loads(message)
                if isinstance(data, dict) and "type" in data:
                    if data["type"] == "user" and "content" in data:
                        # User message - create a message record and add to history
                        message_data = {
                            "type": "user",
                            "content": data["content"],
                            "timestamp": datetime.now().isoformat()
                        }
                        conversations[conversation_id].messages.append(message_data)
                        
                        # Send to all connections in this conversation
                        for conn in conversations[conversation_id].connections:
                            try:
                                await conn.send_text(json.dumps(message_data))
                            except Exception as e:
                                print(f"Error sending to connection: {e}")
                    elif data["type"] == "input_response" and "content" in data:
                        # This message is a response to an input request
                        # The actual handling happens via the callback manager
                        print(f"Input response received in conversation {conversation_id}")
                else:
                    # Unstructured JSON message - treat as user message
                    message_data = {
                        "type": "user",
                        "content": message,
                        "timestamp": datetime.now().isoformat()
                    }
                    conversations[conversation_id].messages.append(message_data)
                    
                    # Send to all connections in this conversation
                    for conn in conversations[conversation_id].connections:
                        try:
                            await conn.send_text(json.dumps(message_data))
                        except Exception as e:
                            print(f"Error sending to connection: {e}")
            except json.JSONDecodeError:
                # Plain text - treat as user message
                message_data = {
                    "type": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                }
                conversations[conversation_id].messages.append(message_data)
                
                # Send to all connections in this conversation
                for conn in conversations[conversation_id].connections:
                    try:
                        await conn.send_text(json.dumps(message_data))
                    except Exception as e:
                        print(f"Error sending to connection: {e}")
    except WebSocketDisconnect:
        print(f"WebSocket disconnected from conversation {conversation_id}")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        # Always ensure connection is removed
        remove_connection(websocket, conversation_id)
        try:
            await websocket.close()
        except:
            # Already closed, ignore
            pass

# Utility to run the server
def run_server():
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

if __name__ == "__main__":
    try:
        print("Starting RAG Conversation Server...")
        print("Checking for circular imports...")
        # Import graph_nodes here to test for circular imports
        import graph_nodes
        print("No circular imports detected.")
        run_server()
    except ImportError as e:
        print(f"Import error detected: {e}")
        print("Please check your module structure to resolve circular imports.")
    except Exception as e:
        print(f"Error starting server: {e}") 
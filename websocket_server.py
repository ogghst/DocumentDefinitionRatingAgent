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
from common import broadcast_message, get_user_input, active_connections_var

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
    from langchain_core.callbacks.base import BaseCallbackHandler
    
    try:
        conversation = conversations[conversation_id]
        conversation.is_active = True
        
        await broadcast_to_conversation(
            conversation_id, 
            "Starting RAG analysis...", 
            "system",
            rag_message=True  # Only send to owner
        )
        
        # Create a proper callback handler class for broadcasting messages
        class ConversationCallbackHandler(BaseCallbackHandler):
            """Callback handler for sending messages to the conversation owner."""
            
            def __init__(self, conversation_id):
                self.conversation_id = conversation_id
                # Store a shared event loop reference
                self.loop = asyncio.get_event_loop()
                super().__init__()
            
            def _schedule_coroutine(self, coro):
                """Simple helper to schedule a coroutine in the main event loop."""
                if self.loop.is_running():
                    self.loop.create_task(coro)
                else:
                    asyncio.run_coroutine_threadsafe(coro, self.loop)
            
            def on_llm_start(self, serialized, prompts, **kwargs):
                """Run when LLM starts."""
                # Simple scheduling for async function
                self._schedule_coroutine(broadcast_to_conversation(
                    self.conversation_id, 
                    "LLM starting to generate...", 
                    "rag_progress", 
                    rag_message=True
                ))
                return super().on_llm_start(serialized, prompts, **kwargs)
            
            def on_llm_new_token(self, token, **kwargs):
                """Run on new LLM token - simplified implementation."""
                # Use the simplified helper to schedule the async send
                self._schedule_coroutine(broadcast_to_conversation(
                    self.conversation_id, token, "rag_token", rag_message=True
                ))
            
            def on_chain_end(self, outputs, **kwargs):
                """Run on chain end."""
                self._schedule_coroutine(broadcast_to_conversation(
                    self.conversation_id, 
                    "Chain completed", 
                    "rag_progress", 
                    rag_message=True
                ))
            
            def on_tool_end(self, output, **kwargs):
                """Run on tool end."""
                self._schedule_coroutine(broadcast_to_conversation(
                    self.conversation_id,
                    f"Tool completed", 
                    "rag_progress", 
                    rag_message=True
                ))
            
            def on_text(self, text, **kwargs):
                """Run on text."""
                self._schedule_coroutine(broadcast_to_conversation(
                    self.conversation_id, 
                    text, 
                    "rag_progress", 
                    rag_message=True
                ))
        
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
            callbacks=[ConversationCallbackHandler(conversation_id)]
        )
        
        # Run the workflow
        final_state = await app.ainvoke(inputs, config=config)
        
        # Store the result
        conversation.workflow_result = final_state
        
        # Send the final result
        if final_state and final_state.get("final_results"):
            results_json = json.dumps(final_state["final_results"])
            await broadcast_to_conversation(
                conversation_id, 
                results_json, 
                "rag_result",
                rag_message=True  # Only send to owner
            )
        elif final_state and final_state.get("error_message"):
            await broadcast_to_conversation(
                conversation_id, 
                f"Error: {final_state['error_message']}", 
                "error",
                rag_message=True  # Only send to owner
            )
        else:
            await broadcast_to_conversation(
                conversation_id, 
                "Analysis completed but no results were found", 
                "warning",
                rag_message=True  # Only send to owner
            )
    except Exception as e:
        await broadcast_to_conversation(
            conversation_id,
            f"Error running RAG workflow: {str(e)}",
            "error",
            rag_message=True  # Only send to owner
        )
    finally:
        # Mark conversation as inactive
        if conversation_id in conversations:
            conversations[conversation_id].is_active = False

async def get_user_input(conversation_id: str, prompt: str) -> str:
    """Get input from the owner of a conversation.
    
    This function sends a prompt to the conversation owner and waits for a response.
    The RAG workflow will pause here until a response is received or the timeout occurs.
    """
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
    
    # Try to optimize the connection
    try:
        socket = websocket._transport._sock
        socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"TCP_NODELAY enabled for connection")
    except Exception:
        # Non-critical, so just continue if this fails
        pass
    
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
        
        # Main message loop
        while True:
            message = await websocket.receive_text()
            
            # Process the message
            try:
                # Try to parse as JSON
                data = json.loads(message)
                if isinstance(data, dict) and "type" in data:
                    if data["type"] == "user" and "content" in data:
                        # User message - send to everyone (chat functionality)
                        await broadcast_to_conversation(
                            conversation_id, 
                            data["content"], 
                            "user"
                        )
                    elif data["type"] == "input_response" and "content" in data:
                        # This message is a response to an input request
                        # The actual handling happens in the get_user_input function
                        # where it's waiting for a message from this connection
                        print(f"Input response received in conversation {conversation_id}")
                        
                        # We don't need to do anything special here because
                        # the get_user_input function is directly listening for
                        # messages on this WebSocket connection
                        pass
                else:
                    # Unstructured JSON message - treat as user message
                    await broadcast_to_conversation(
                        conversation_id, 
                        message, 
                        "user"
                    )
            except json.JSONDecodeError:
                # Plain text - treat as user message
                await broadcast_to_conversation(
                    conversation_id, 
                    message, 
                    "user"
                )
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
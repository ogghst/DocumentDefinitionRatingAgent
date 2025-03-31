import asyncio
from typing import Set, List, Dict, Any, Optional
import contextvars
from aioconsole import ainput
import json
from datetime import datetime
import uuid
import weakref

# Context variable for active connections
active_connections_var = contextvars.ContextVar('active_connections', default=set())

# Dictionary to store references to active callback managers by conversation ID
# This is just a declaration - it will be populated from websocket_server.py
conversation_callbacks = {}

# Message queues for each connection
# Using weakref.WeakKeyDictionary so connections can be garbage collected
connection_queues = weakref.WeakKeyDictionary()

# Shared functions for broadcasting and user input
async def broadcast_message(message: str, target_connections: Optional[List] = None):
    """Send a message to specified connections or all active ones."""
    # Always print to console for debugging
    print(f"[BROADCAST] {message}")
    
    # Import only when needed to avoid circular imports
    from websocket_server import conversation_callbacks, conversations
    
    # Try to send to all active callback managers first
    message_sent_to_callback = False
    for conv_id, callback in conversation_callbacks.items():
        if callback._active and callback.websocket:
            try:
                # Use the websocket callback's method to send message
                await callback._send_message(message, "rag_progress")
                message_sent_to_callback = True
            except Exception as e:
                print(f"Error sending to callback manager for {conv_id}: {e}")
    
    # If we already sent via callback managers, we're done
    if message_sent_to_callback:
        return
        
    # If no callback managers received the message, try direct connections
    connections = target_connections or list(active_connections_var.get())
    if connections:
        try:
            message_data = {
                "type": "rag_progress",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            json_message = json.dumps(message_data)
            
            for connection in connections:
                try:
                    await connection.send_text(json_message)
                except Exception as e:
                    print(f"Error sending to connection: {e}")
        except Exception as e:
            print(f"Error during broadcast: {e}")
    else:
        print("[BROADCAST] No active connections, message only logged to console")

# Function to queue a message for processing by waiting coroutines
def queue_message(connection, message: str):
    """Queue a message for a specific connection to be processed by waiting coroutines."""
    if connection not in connection_queues:
        # Create a new queue if it doesn't exist
        connection_queues[connection] = {}
    
    try:
        # Parse the message to determine its type
        data = json.loads(message)
        if isinstance(data, dict):
            # For input_response messages, notify all waiting requests
            if data.get("type") == "input_response":
                content = data.get("content", "")
                # Find all waiting requests and complete them
                for request_id, future in list(connection_queues[connection].items()):
                    if not future.done():
                        future.set_result(content)
                # Clear the queue after processing all waiting requests
                connection_queues[connection] = {}
                return True
    except json.JSONDecodeError:
        # Not JSON, treat as plain text
        pass
    except Exception as e:
        print(f"Error processing message: {e}")
    
    return False

# The main get_user_input function that will be called from the workflow nodes
async def get_user_input(prompt: str, conversation_id: Optional[str] = None) -> str:
    """
    Get input from a user.
    
    This is a generic function that can be called from anywhere in the codebase:
    - If conversation_id is provided and a callback manager exists, it uses that
    - If no conversation_id or no callback manager, sends to all active connections
    
    Args:
        prompt: The prompt to show the user
        conversation_id: Optional conversation ID for websocket communication
        
    Returns:
        The user's input as a string
    """
    # Try to use the callback manager if available
    if conversation_id:
        # Import only if conversation_id is provided
        from websocket_server import conversation_callbacks
        
        if conversation_id in conversation_callbacks:
            # Use the callback manager which will directly handle the websocket communication
            return await conversation_callbacks[conversation_id].get_user_input(prompt)
        else:
            # No callback manager, but log that we were expecting one
            print(f"No callback manager found for conversation {conversation_id}")
    
    # If we reach here, either no conversation_id was provided or no callback manager exists
    # Send the input request to all active connections
    connections = list(active_connections_var.get())
    if connections:
        message_data = {
            "type": "input_request",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        json_message = json.dumps(message_data)
        
        print(f"[INPUT REQUEST] Broadcasting to {len(connections)} connections: {prompt}")
        
        # Send to all connections
        for connection in connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                print(f"Error sending input request to connection: {e}")
        
        # Wait for a response from any connection
        if connections:
            connection = connections[0]
            try:
                return await _wait_for_user_response(connection)
            except Exception as e:
                print(f"Error waiting for response: {e}")
                return "Error getting user input"
    
    # Fall back to console input only if there are no active connections
    print(f"\n[USER INPUT REQUIRED] {prompt}")
    return await ainput("Your response: ")

async def _wait_for_user_response(connection):
    """Wait for user response using the message queue system."""
    # Generate a unique request ID for this waiting request
    request_id = str(uuid.uuid4())
    
    # Create a future for this request
    if connection not in connection_queues:
        connection_queues[connection] = {}
    
    future = asyncio.get_event_loop().create_future()
    connection_queues[connection][request_id] = future
    
    try:
        # Wait for the response with a timeout
        return await asyncio.wait_for(future, timeout=300)  # 5 minutes timeout
    except asyncio.TimeoutError:
        print(f"Timeout waiting for response for request {request_id}")
        return "timeout"
    except Exception as e:
        print(f"Error waiting for response: {e}")
        raise
    finally:
        # Clean up the request from the queue
        if connection in connection_queues and request_id in connection_queues[connection]:
            connection_queues[connection].pop(request_id, None) 
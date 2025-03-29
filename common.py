import asyncio
from typing import Set, List, Dict, Any, Optional
import contextvars
from aioconsole import ainput

# Context variable for active connections
active_connections_var = contextvars.ContextVar('active_connections', default=set())

# Dictionary to store references to active callback managers by conversation ID
# This is just a declaration - it will be populated from websocket_server.py
conversation_callbacks = {}

# Shared functions for broadcasting and user input
async def broadcast_message(message: str, target_connections: Optional[List] = None):
    """Send a message to specified connections or all active ones."""
    # Always print to console for debugging
    print(f"[BROADCAST] {message}")
    
    connections = target_connections or list(active_connections_var.get())
    if connections:
        try:
            for connection in connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"Error sending to connection: {e}")
        except Exception as e:
            print(f"Error during broadcast: {e}")
    else:
        print("[BROADCAST] No active connections, message only logged to console")

# The main get_user_input function that will be called from the workflow nodes
async def get_user_input(prompt: str, conversation_id: Optional[str] = None) -> str:
    """
    Get input from a user.
    
    This is a generic function that can be called from anywhere in the codebase:
    - If conversation_id is provided and a callback manager exists, it uses that
    - If in a terminal environment, falls back to console input
    
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
    
    # Fall back to console input
    print(f"\n[USER INPUT REQUIRED] {prompt}")
    return await ainput("Your response: ")

async def _wait_for_user_response(connection):
    """Continuously check for user response with async sleep"""
    while True:
        try:
            if msg := await connection.receive_text():
                return msg
            await asyncio.sleep(0.1)  # Prevent busy waiting
        except (asyncio.CancelledError, ConnectionResetError):
            raise
        except Exception:
            continue 
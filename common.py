import asyncio
from typing import Set, List, Dict, Any, Optional
import contextvars
from aioconsole import ainput

# Context variable for active connections
active_connections_var = contextvars.ContextVar('active_connections', default=set())

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

async def get_user_input(prompt: str, connections: Optional[List] = None) -> str:
    """Get input from the first available WebSocket client."""
    target_connections = connections or list(active_connections_var.get())
    
    if not target_connections:
        print(prompt)
        return input("> ")  # Fall back to console input if no connections
    
    # Send the prompt to all clients with a special marker for UI handling
    input_request_message = f"INPUT_REQUEST:{prompt}"
    await broadcast_message(input_request_message, target_connections)
    
    # Verify message was sent by logging confirmation
    print(f"[DEBUG] Input request sent to {len(target_connections)} connections")
    
    # Wait for a response from any client
    if target_connections:
        first_connection = target_connections[0]
        try:
            # Send a heartbeat to ensure connection is active
            await first_connection.send_text("PING")
            
            # Log waiting state
            print("[DEBUG] Waiting for user input...")
            
            # Actually wait for response
            response = await first_connection.receive_text()
            print(f"[DEBUG] Received response: {response}")
            return response
        except Exception as e:
            # Log specific exception
            print(f"[ERROR] Exception while waiting for input: {type(e).__name__}: {str(e)}")
            return "skip"

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
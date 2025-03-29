import asyncio
from typing import Set, List, Dict, Any, Optional
import contextvars

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
    
    # Send the prompt to all clients
    await broadcast_message(prompt, target_connections)
    
    # Wait for a response from any client
    if target_connections:
        first_connection = target_connections[0]
        try:
            return await first_connection.receive_text()
        except Exception:
            # Handle connection errors
            return "skip"
    
    return "skip"  # Default if no connections 
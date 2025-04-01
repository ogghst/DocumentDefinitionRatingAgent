import asyncio
from typing import Set, List, Dict, Any, Optional
import contextvars
from aioconsole import ainput
import json
from datetime import datetime
import uuid
import weakref

# Context variable for active connections - No longer used, remove?
# active_connections_var = contextvars.ContextVar('active_connections', default=set())

# Remove references to callback managers and queues - managed in websocket_server
# conversation_callbacks = {}
# connection_queues = weakref.WeakKeyDictionary()

# Remove old send_message function
# async def send_message(conversation_id: str, message: str, message_type: str = "rag_progress"):
#     """Send a message to the specific connection associated with the conversation_id."""
#     # ... (old implementation removed)

# Remove old queue_message function
# def queue_message(connection, message: str):
#     """Queue a message for a specific connection to be processed by waiting coroutines."""
#     # ... (old implementation removed)

# Remove old get_user_input and _wait_for_user_response functions
# async def get_user_input(prompt: str, conversation_id: Optional[str] = None) -> str:
#     """
#     Get input from a user.
#     
#     DEPRECATED: Logic moved to websocket_server.py and WebsocketCallbackManager
#     """
#     # ... (old implementation removed)

# async def _wait_for_user_response(connection):
#     """Wait for user response using the message queue system."""
#     # ... (old implementation removed)

# common.py is now significantly simpler. 
# It might primarily hold shared data models or truly universal utilities 
# if needed in the future.

# Example of a utility that might remain or be added:
# def format_timestamp(dt: datetime) -> str:
#     return dt.strftime("%Y-%m-%d %H:%M:%S") 
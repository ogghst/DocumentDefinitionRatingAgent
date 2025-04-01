import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Coroutine
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult
# Remove WebSocket import as it's no longer managed here
# from fastapi import WebSocket

# Import necessary functions without creating circular imports
# These will be imported when the file is used
# from common import broadcast_message
# from websocket_server import broadcast_to_conversation


class WebsocketCallbackManager(AsyncCallbackHandler):
    """
    A callback manager that handles websocket communication for LLM streaming tokens and user feedback.
    
    This class provides a unified interface for:
    - Streaming tokens to clients via websockets (using a callback)
    - Collecting user feedback during LLM runs (using a callback)
    - Broadcasting status updates to clients (using a callback)
    """
    
    def __init__(self,
                 conversation_id: str,
                 send_callback: Callable[[str, str], Coroutine[Any, Any, None]],
                 request_input_callback: Callable[[str], Coroutine[Any, Any, str]]):
        """
        Initialize the callback manager.
        
        Args:
            conversation_id: The ID of the conversation this manager is handling
            send_callback: An async function to call for sending messages.
                         Expected signature: send_callback(message_content: str, message_type: str)
            request_input_callback: An async function to call for requesting user input.
                                  Expected signature: request_input_callback(prompt: str) -> str
        """
        self.conversation_id = conversation_id
        # Remove direct websocket reference
        # self.websocket = websocket
        self.send_callback = send_callback
        self.request_input_callback = request_input_callback
        
        # self.loop = asyncio.get_event_loop() # loop is not explicitly needed
        # Input handling state is removed - managed by websocket_server now
        # self.waiting_for_input = False
        # self.user_input_event = asyncio.Event()
        # self.user_response = [""]
        
        self._token_buffer = []
        self._max_buffer_size = 1  # Number of tokens to buffer before sending
        self._active = True  # Flag to track if this callback manager is still active
        
        # Add LangChain callback manager compatibility attributes
        self.parent_run_id = None
        self.tags = []
        self.inheritable_tags = []
        self.metadata = {}
        self.inheritable_metadata = {}
        self.name = f"WebsocketCallback-{conversation_id}"
        
        # Add handlers attribute required by LangChain's AsyncCallbackManager
        self.handlers = []
        
        super().__init__()
    
    # Remove set_websocket method
    # def set_websocket(self, websocket: WebSocket):
    #     """Set or update the websocket connection for this callback manager."""
    #     self.websocket = websocket
    
    def deactivate(self):
        """Mark this callback manager as inactive."""
        self._active = False
        # Clear any pending events (input logic moved)
        # if not self.user_input_event.is_set():
        #     self.user_input_event.set()
    
    def copy(self):
        """Return a copy of this callback manager.
        
        Required by LangChain's ensure_config function when used in RunnableConfig.
        """
        # Create a new instance with the same conversation_id and callbacks
        new_manager = WebsocketCallbackManager(
            self.conversation_id,
            self.send_callback, 
            self.request_input_callback
        )
        
        # Copy over the necessary attributes for LangChain compatibility
        new_manager.parent_run_id = self.parent_run_id
        new_manager.tags = self.tags.copy() if self.tags else []
        new_manager.inheritable_tags = self.inheritable_tags.copy() if self.inheritable_tags else []
        new_manager.metadata = self.metadata.copy() if self.metadata else {}
        new_manager.inheritable_metadata = self.inheritable_metadata.copy() if self.inheritable_metadata else {}
        new_manager.handlers = self.handlers.copy() if self.handlers else []
        new_manager._active = self._active # Copy active status
        
        return new_manager
    
    # --- LangChain Compatibility Methods (unchanged) ---
    def get_parent_run_id(self):
        return self.parent_run_id
    def get_tags(self):
        return self.tags.copy() if self.tags else []
    def get_inheritable_tags(self):
        return self.inheritable_tags.copy() if self.inheritable_tags else []
    def get_metadata(self):
        return self.metadata.copy() if self.metadata else {}
    def get_inheritable_metadata(self):
        return self.inheritable_metadata.copy() if self.inheritable_metadata else {}
    # --- End LangChain Compatibility Methods ---
    
    async def _send_message(self, content: str, message_type: str = "rag_progress"):
        """Send a message using the provided send_callback."""
        if not self._active:
            return
            
        try:
            # Call the callback function provided during initialization
            await self.send_callback(content, message_type)
        except Exception as e:
            print(f"Error calling send_callback in manager for {self.conversation_id}: {e}")
            # Consider deactivating on persistent send errors
            # self._active = False 
    
    async def _send_buffered_tokens(self):
        """Send buffered tokens if any exist."""
        if not self._token_buffer or not self._active:
            return
            
        # Join buffered tokens
        joined_tokens = "".join(self._token_buffer)
        self._token_buffer = []
        
        # Send using the callback
        await self._send_message(joined_tokens, "rag_token")
    
    # ---- LLM Callbacks (use _send_message) ----
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        # Protect against None parameters
        if not isinstance(serialized, dict):
            serialized = {}
        
        llm_name = serialized.get("name", "LLM")
        await self._send_message(
            f"Asking to AI...", 
            "rag_progress"
        )
        
        # Clear token buffer at the start of each LLM call
        self._token_buffer = []
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token with buffering for efficiency."""
        if not self._active:
            return
            
        # Buffer tokens for more efficient websocket communication
        self._token_buffer.append(token)
        
        # Send when buffer is full or token is a natural break point
        if len(self._token_buffer) >= self._max_buffer_size or any(c in token for c in ['.', '!', '?', '\n']):
            await self._send_buffered_tokens()
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends - send any remaining buffered tokens."""
        await self._send_buffered_tokens()
        
        # Add optional completion message
        if response is not None:
            await self._send_message(
                "AI replied to the question",
                "rag_progress"
            )
    
    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when LLM errors."""
        await self._send_buffered_tokens()  # Send any buffered tokens
        
        # Make sure error is properly formatted
        error_msg = "Unknown error"
        if error is not None:
            try:
                error_msg = str(error)
            except:
                error_msg = type(error).__name__
                
        await self._send_message(
            f"Error during AI analysis: {error_msg}",
            "error"
        )
    
    # ---- Chain Callbacks (use _send_message, logic mostly unchanged) ----
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts."""
        chain_type = "Chain"
        if serialized is not None and isinstance(serialized, dict):
            chain_type = serialized.get("name", "Chain")
        # Pass (no message sent)
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        # Pass (no message sent)
    
    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chain errors."""
        error_msg = "Unknown error"
        if error is not None:
            try:
                error_msg = str(error)
            except:
                error_msg = type(error).__name__
                
        await self._send_message(
            f"Error in processing chain: {error_msg}",
            "error"
        )
    
    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run on tool end."""
        await self._send_message(
            f"Tool completed", 
            "rag_progress"
        )
    
    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        if text is None:
            return
        elif not isinstance(text, str):
            try:
                text = str(text)
            except:
                return
                
        await self._send_message(
            text,
            "rag_progress"
        )
    
    # ---- User Input Methods (use request_input_callback) ----
    
    async def get_user_input(self, prompt: str) -> str:
        """
        Get input from the user with the specified prompt by calling the request_input_callback.
        
        This is a blocking call that will wait for user input via the callback.
        """
        if not self._active:
            print(f"Callback manager for {self.conversation_id} is inactive. Cannot request input.")
            return "skip: manager inactive"
            
        try:
            # Use the callback provided during initialization to request input
            print(f"Callback manager {self.conversation_id} requesting input with prompt: {prompt}")
            response = await self.request_input_callback(prompt)
            print(f"Callback manager {self.conversation_id} received input: {response}")
            return response
        except Exception as e:
            print(f"Error calling request_input_callback in manager {self.conversation_id}: {e}")
            await self._send_message(f"Error requesting user input: {e}", "error")
            return f"error: {e}"
    
    # Remove handle_user_message - this is handled by the websocket endpoint now
    # def handle_user_message(self, message: str):
    #     """
    #     Handle a message from the user.
    #     
    #     If we're waiting for input, this will resolve the waiting.
    #     """
    #     # ... (old logic removed)
    
    # ---- Chat Model Callbacks (use _send_message, logic mostly unchanged) ----
    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Dict[str, Any]]], **kwargs: Any
    ) -> None:
        """Run when chat model starts generating."""
        # Clear token buffer at the start of each model call
        self._token_buffer = []
        # Pass (no message sent)
    
    async def on_chat_model_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chat model ends generation."""
        await self._send_buffered_tokens()
        # Pass (no message sent)
    
    async def on_chat_model_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chat model errors."""
        await self._send_buffered_tokens()  # Send any buffered tokens
        
        error_msg = "Unknown error"
        if error is not None:
            try:
                error_msg = str(error)
            except:
                error_msg = type(error).__name__
                
        await self._send_message(
            f"Error during chat model generation: {error_msg}",
            "error"
        ) 
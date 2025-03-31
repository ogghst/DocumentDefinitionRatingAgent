import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from fastapi import WebSocket

# Import necessary functions without creating circular imports
# These will be imported when the file is used
# from common import broadcast_message
# from websocket_server import broadcast_to_conversation


class WebsocketCallbackManager(AsyncCallbackHandler):
    """
    A callback manager that handles websocket communication for LLM streaming tokens and user feedback.
    
    This class provides a unified interface for:
    - Streaming tokens to clients via websockets
    - Collecting user feedback during LLM runs
    - Broadcasting status updates to clients
    """
    
    def __init__(self, conversation_id: str, websocket: Optional[WebSocket] = None):
        """
        Initialize the callback manager.
        
        Args:
            conversation_id: The ID of the conversation this manager is handling
            websocket: Direct reference to the websocket connection (owner)
        """
        self.conversation_id = conversation_id
        self.websocket = websocket  # Store direct reference to owner websocket
        self.loop = asyncio.get_event_loop()
        self.waiting_for_input = False
        self.user_input_event = asyncio.Event()
        self.user_response = [""]  # Use list for mutable reference
        self._token_buffer = []
        self._max_buffer_size = 10  # Number of tokens to buffer before sending
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
    
    def set_websocket(self, websocket: WebSocket):
        """Set or update the websocket connection for this callback manager."""
        self.websocket = websocket
    
    def deactivate(self):
        """Mark this callback manager as inactive."""
        self._active = False
        # Clear any pending events
        if not self.user_input_event.is_set():
            self.user_input_event.set()
    
    def copy(self):
        """Return a copy of this callback manager.
        
        Required by LangChain's ensure_config function when used in RunnableConfig.
        """
        # Create a new instance with the same conversation_id
        new_manager = WebsocketCallbackManager(self.conversation_id, self.websocket)
        
        # Copy over the necessary attributes for LangChain compatibility
        new_manager.parent_run_id = self.parent_run_id
        new_manager.tags = self.tags.copy() if self.tags else []
        new_manager.inheritable_tags = self.inheritable_tags.copy() if self.inheritable_tags else []
        new_manager.metadata = self.metadata.copy() if self.metadata else {}
        new_manager.inheritable_metadata = self.inheritable_metadata.copy() if self.inheritable_metadata else {}
        new_manager.handlers = self.handlers.copy() if self.handlers else []
        
        return new_manager
    
    def get_parent_run_id(self):
        """Get the parent run ID for this callback manager.
        
        Used by LangChain's callback system.
        """
        return self.parent_run_id
    
    def get_tags(self):
        """Get tags for this callback manager.
        
        Used by LangChain's callback system.
        """
        return self.tags.copy() if self.tags else []
    
    def get_inheritable_tags(self):
        """Get inheritable tags for this callback manager.
        
        Used by LangChain's callback system.
        """
        return self.inheritable_tags.copy() if self.inheritable_tags else []
    
    def get_metadata(self):
        """Get metadata for this callback manager.
        
        Used by LangChain's callback system.
        """
        return self.metadata.copy() if self.metadata else {}
    
    def get_inheritable_metadata(self):
        """Get inheritable metadata for this callback manager.
        
        Used by LangChain's callback system.
        """
        return self.inheritable_metadata.copy() if self.inheritable_metadata else {}
    
    async def _send_message(self, content: str, message_type: str = "rag_progress"):
        """Send a message directly to the websocket if it exists."""
        if not self._active or not self.websocket:
            return
            
        try:
            message_data = {
                "type": message_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            await self.websocket.send_text(json.dumps(message_data))
        except Exception as e:
            print(f"Error sending message to websocket: {e}")
            self._active = False  # Mark as inactive if sending fails
    
    async def _send_buffered_tokens(self):
        """Send buffered tokens if any exist."""
        if not self._token_buffer or not self._active:
            return
            
        # Join buffered tokens
        joined_tokens = "".join(self._token_buffer)
        self._token_buffer = []
        
        # Send directly to websocket
        await self._send_message(joined_tokens, "rag_token")
    
    # ---- LLM Callbacks ----
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        # Protect against None parameters
        if not isinstance(serialized, dict):
            serialized = {}
        
        llm_name = serialized.get("name", "LLM")
        await self._send_message(
            f"{llm_name} starting to generate...", 
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
                "Generation completed",
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
            f"Error during LLM generation: {error_msg}",
            "error"
        )
    
    # ---- Chain Callbacks ----
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts."""
        # Check if serialized is None and provide a default value
        chain_type = "Chain"
        if serialized is not None and isinstance(serialized, dict):
            chain_type = serialized.get("name", "Chain")
        
        # Not sending message to avoid noise
        pass
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        # Protect against None parameters
        if not isinstance(outputs, dict):
            outputs = {}
            
        # Not sending message to avoid noise
        pass
    
    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chain errors."""
        # Make sure error is properly formatted
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
        # Make sure output is a string
        if output is None:
            output = "No output"
        elif not isinstance(output, str):
            output = str(output)
            
        await self._send_message(
            f"Tool completed", 
            "rag_progress"
        )
    
    async def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        # Make sure text is a string
        if text is None:
            return  # Skip sending None text
        elif not isinstance(text, str):
            try:
                text = str(text)
            except:
                return  # Skip if we can't convert to string
                
        await self._send_message(
            text,
            "rag_progress"
        )
    
    # ---- User Input Methods ----
    
    async def get_user_input(self, prompt: str) -> str:
        """
        Get input from the user with the specified prompt.
        
        This is a blocking call that will wait for user input.
        """
        if not self._active or not self.websocket:
            print(f"Cannot get user input - websocket not available or manager inactive")
            return "skip"
            
        # Reset state
        self.waiting_for_input = True
        self.user_input_event.clear()
        self.user_response[0] = ""
        
        # Notify client we're waiting for input
        await self._send_message(
            "RAG analysis paused: Waiting for human input...",
            "rag_progress"
        )
        
        # Send the prompt
        await self._send_message(
            prompt,
            "input_request"
        )
        
        try:
            # Wait for response with timeout
            await asyncio.wait_for(self.user_input_event.wait(), timeout=300)  # 5 minute timeout
            return self.user_response[0]
        except asyncio.TimeoutError:
            await self._send_message(
                "Timeout waiting for user input. Proceeding with default.",
                "warning"
            )
            return "timeout"
        finally:
            self.waiting_for_input = False
    
    def handle_user_message(self, message: str):
        """
        Handle a message from the user.
        
        If we're waiting for input, this will resolve the waiting.
        """
        if self.waiting_for_input:
            try:
                # Try to parse as JSON
                data = json.loads(message)
                if isinstance(data, dict) and "content" in data:
                    self.user_response[0] = data["content"]
                else:
                    self.user_response[0] = message
            except json.JSONDecodeError:
                self.user_response[0] = message
            
            # Signal that we've received input
            self.user_input_event.set()
    
    async def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Dict[str, Any]]], **kwargs: Any
    ) -> None:
        """Run when chat model starts generating."""
        # Protect against None parameters
        if not isinstance(serialized, dict):
            serialized = {}
        
        model_name = serialized.get("name", "Chat Model")
        await self._send_message(
            f"{model_name} starting to generate...",
            "rag_progress"
        )
        
        # Clear token buffer at the start of each model call
        self._token_buffer = []
    
    async def on_chat_model_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chat model ends generation."""
        await self._send_buffered_tokens()
        
        # Add optional completion message
        if response is not None:
            await self._send_message(
                "Generation completed",
                "rag_progress"
            )
    
    async def on_chat_model_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chat model errors."""
        await self._send_buffered_tokens()  # Send any buffered tokens
        
        # Make sure error is properly formatted
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
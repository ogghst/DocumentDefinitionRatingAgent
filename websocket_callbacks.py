import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from fastapi import WebSocket

# Import necessary functions without creating circular imports
# These will be imported when the file is used
# from common import broadcast_message
# from websocket_server import broadcast_to_conversation


class WebsocketCallbackManager(BaseCallbackHandler):
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
    
    def _schedule_coroutine(self, coro):
        """Simple helper to schedule a coroutine in the main event loop."""
        if self.loop.is_running():
            self.loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, self.loop)
    
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
    
    def _send_buffered_tokens(self):
        """Send buffered tokens if any exist."""
        if not self._token_buffer or not self._active:
            return
            
        # Join buffered tokens
        joined_tokens = "".join(self._token_buffer)
        self._token_buffer = []
        
        # Send directly to websocket
        self._schedule_coroutine(self._send_message(joined_tokens, "rag_token"))
    
    # ---- LLM Callbacks ----
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts."""
        # Protect against None parameters
        if not isinstance(serialized, dict):
            serialized = {}
        
        llm_name = serialized.get("name", "LLM")
        self._schedule_coroutine(self._send_message(
            f"{llm_name} starting to generate...", 
            "rag_progress"
        ))
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token with buffering for efficiency."""
        # Buffer tokens for more efficient websocket communication
        self._token_buffer.append(token)
        
        # Send when buffer is full or token is a natural break point
        if len(self._token_buffer) >= self._max_buffer_size or any(c in token for c in ['.', '!', '?', '\n']):
            self._send_buffered_tokens()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends - send any remaining buffered tokens."""
        self._send_buffered_tokens()
        
        # Add optional completion message
        if response is not None:
            self._schedule_coroutine(self._send_message(
                "Generation completed",
                "rag_progress"
            ))
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when LLM errors."""
        self._send_buffered_tokens()  # Send any buffered tokens
        
        # Make sure error is properly formatted
        error_msg = "Unknown error"
        if error is not None:
            try:
                error_msg = str(error)
            except:
                error_msg = type(error).__name__
                
        self._schedule_coroutine(self._send_message(
            f"Error during LLM generation: {error_msg}",
            "error"
        ))
    
    # ---- Chain Callbacks ----
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts."""
        # Check if serialized is None and provide a default value
        chain_type = "Chain"
        if serialized is not None and isinstance(serialized, dict):
            chain_type = serialized.get("name", "Chain")
        
        #self._schedule_coroutine(self._send_message(
        #    f"Starting {chain_type}...",
        #    "rag_progress"
        #))
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends."""
        # Protect against None parameters
        if not isinstance(outputs, dict):
            outputs = {}
            
        #self._schedule_coroutine(self._send_message(
        #    "Chain completed",
        #    "rag_progress"
        #))
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Run when chain errors."""
        # Make sure error is properly formatted
        error_msg = "Unknown error"
        if error is not None:
            try:
                error_msg = str(error)
            except:
                error_msg = type(error).__name__
                
        self._schedule_coroutine(self._send_message(
            f"Error in processing chain: {error_msg}",
            "error"
        ))
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run on tool end."""
        # Make sure output is a string
        if output is None:
            output = "No output"
        elif not isinstance(output, str):
            output = str(output)
            
        self._schedule_coroutine(self._send_message(
            f"Tool completed", 
            "rag_progress"
        ))
    
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on text."""
        # Make sure text is a string
        if text is None:
            return  # Skip sending None text
        elif not isinstance(text, str):
            try:
                text = str(text)
            except:
                return  # Skip if we can't convert to string
                
        self._schedule_coroutine(self._send_message(
            text,
            "rag_progress"
        ))
    
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
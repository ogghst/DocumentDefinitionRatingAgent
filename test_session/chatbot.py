from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import random
import time
import uuid

app = FastAPI()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)

    async def stream_response(self, response: str, client_id: str):
        """Stream a response word by word with random pauses"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            words = response.split()
            for i, word in enumerate(words):
                # Add space before words (except the first one)
                if i > 0:
                    await websocket.send_text(" ")
                
                # Stream each character of the word
                for char in word:
                    await websocket.send_text(char)
                    # Random pause between characters
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                
                # Slightly longer pause between words
                await asyncio.sleep(random.uniform(0.05, 0.2))

manager = ConnectionManager()

# Simple response generator function
def get_response(message: str) -> str:
    # In a real app, this would call your chatbot logic
    responses = {
        "hello": "Hello there! How can I help you today?",
        "how are you": "I'm just a computer program, but I'm functioning well! How about you?",
        "bye": "Goodbye! Have a great day!"
    }
    
    message = message.lower()
    for key in responses:
        if key in message:
            return responses[key]
    
    return "I'm not sure how to respond to that. Can you try asking something else?"

# HTML page with WebSocket client
@app.get("/")
async def get():
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat</title>
            <style>
                body { 
                    max-width: 600px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    font-family: Arial, sans-serif;
                }
                #messages { 
                    border: 1px solid #ddd; 
                    height: 300px; 
                    overflow-y: auto; 
                    padding: 10px; 
                    margin-bottom: 10px;
                }
                #messageInput { 
                    width: 85%; 
                    padding: 8px; 
                }
                button { 
                    padding: 8px 12px; 
                }
                .user { color: blue; }
                .bot { color: green; }
            </style>
        </head>
        <body>
            <h1>Streaming Chatbot</h1>
            <div id="messages"></div>
            <input type="text" id="messageInput" placeholder="Type a message...">
            <button onclick="sendMessage()">Send</button>
            
            <script>
                const clientId = Date.now().toString();
                let ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
                let msgDiv = document.getElementById('messages');
                let input = document.getElementById('messageInput');
                let currentBotResponse = "";
                let botResponseDiv = null;
                
                ws.onopen = function(event) {
                    msgDiv.innerHTML += '<div>Connected to server</div>';
                };
                
                ws.onmessage = function(event) {
                    // First message of a response
                    if (event.data.startsWith('BOT_START:')) {
                        currentBotResponse = "";
                        botResponseDiv = document.createElement('div');
                        botResponseDiv.className = 'bot';
                        botResponseDiv.innerHTML = '<strong>Bot:</strong> ';
                        msgDiv.appendChild(botResponseDiv);
                    }
                    // End of bot message
                    else if (event.data === 'BOT_END') {
                        // Response complete
                        botResponseDiv = null;
                    }
                    // Normal message chunk
                    else {
                        currentBotResponse += event.data;
                        if (botResponseDiv) {
                            botResponseDiv.innerHTML = '<strong>Bot:</strong> ' + currentBotResponse;
                        }
                    }
                    
                    // Auto scroll to bottom
                    msgDiv.scrollTop = msgDiv.scrollHeight;
                };
                
                ws.onclose = function(event) {
                    msgDiv.innerHTML += '<div>Disconnected from server</div>';
                };
                
                function sendMessage() {
                    const message = input.value;
                    if (message) {
                        // Display user message
                        msgDiv.innerHTML += `<div class="user"><strong>You:</strong> ${message}</div>`;
                        
                        // Send to server
                        ws.send(message);
                        
                        // Clear input
                        input.value = '';
                        
                        // Auto scroll
                        msgDiv.scrollTop = msgDiv.scrollHeight;
                    }
                }
                
                // Send message on Enter key
                input.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            message = await websocket.receive_text()
            
            # Let the user know we're starting a response
            await websocket.send_text(f"BOT_START:")
            
            # Get bot response
            response = get_response(message)
            
            # Stream the response
            await manager.stream_response(response, client_id)
            
            # Signal the end of the response
            await websocket.send_text("BOT_END")
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
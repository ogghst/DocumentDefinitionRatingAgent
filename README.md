# RAG Conversation Server

A WebSocket-based conversation server for Retrieval-Augmented Generation (RAG) analysis with human-in-the-loop functionality. This system analyzes documents against compliance checklists and seamlessly requests human input when needed.

## 🔍 Overview

This system enables real-time document analysis against structured checklists, with a focus on project compliance validation. Key features include:

- Real-time token streaming from LLM responses
- WebSocket-based client-server communication
- Human-in-the-loop workflow for uncertain analyses
- Conversation management with support for multiple concurrent sessions
- Document retrieval and embedding for semantic search

## 🏗️ Architecture

The application follows a modular architecture with these main components:

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client     │◄────►│  WebSocket   │◄────►│ RAG Workflow │
│  (Browser)   │      │    Server    │      │    Engine    │
└──────────────┘      └──────────────┘      └──────────────┘
                             ▲                     ▲
                             │                     │
                             ▼                     ▼
                      ┌──────────────┐     ┌──────────────┐
                      │ Conversation │     │   Document   │
                      │  Management  │     │  Processing  │
                      └──────────────┘     └──────────────┘

### Components

1. **WebSocket Server**: FastAPI-based server that manages WebSocket connections and conversation state
2. **RAG Workflow Engine**: LangGraph-powered workflow for document analysis
3. **Client Interface**: HTML/JavaScript client for interacting with the server
4. **Conversation Management**: Handles multiple concurrent conversations and user roles
5. **Document Processing**: Loads, chunks, and indexes documents for RAG retrieval

## 🧩 Key Components

### WebSocket Server (`websocket_server.py`)

Central server component that:
- Manages WebSocket connections for real-time communication
- Handles conversation state and message routing
- Coordinates the RAG workflow execution
- Facilitates human-in-the-loop interaction

### Graph Nodes (`graph_nodes.py`)

Defines the LangGraph workflow:
- Loads and filters checklists for targeted analysis
- Processes documents and creates vector embeddings
- Executes the RAG analysis chain
- Determines when human feedback is needed
- Formats analysis results

### RAG Engine (`rag_engine.py`)

Core RAG implementation:
- Creates and configures the RAG chain
- Defines prompts for analysis and question generation
- Handles token streaming and parsing
- Manages the reliability assessment

### Client Interface (`client.html`)

Browser-based interface that:
- Connects to the WebSocket server
- Displays real-time token streaming
- Presents analysis results
- Facilitates human input when requested
- Manages multiple conversations

## 📚 Libraries Used

- **FastAPI**: Web server framework for API and WebSocket endpoints
- **LangChain**: Framework for building LLM applications
- **LangGraph**: Orchestration for complex LLM workflows
- **Pydantic**: Data validation and settings management
- **Ollama**: Local LLM integration (configurable)
- **AsyncIO**: Asynchronous I/O for efficient concurrency
- **Pandas**: Data handling for checklist processing
- **HTML/JavaScript**: Client-side interface

## 🚀 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-conversation-server.git
   cd rag-conversation-server
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. Start the server:
   ```bash
   python websocket_server.py
   ```

5. Open the client in a browser:
   ```
   http://localhost:8765/client.html
   ```

## 💬 Usage

1. **Create a new conversation** with the desired document and checklist
2. **Start the RAG analysis** to begin processing
3. **Watch real-time token streaming** as the analysis progresses
4. **Provide input when requested** for low-confidence assessments
5. **Review final results** showing compliance status for each check item

## ⚙️ Configuration

Key configuration options:

- **LLM Settings**: Configure the model in `llm_setup.py`
- **Document Processing**: Adjust chunking in `document_processor.py`
- **RAG Parameters**: Modify prompts and thresholds in `rag_engine.py`
- **Human-in-the-loop Settings**: Configure timeouts and attempts in `graph_nodes.py`

## 🔧 Development

For local development:

```bash
# Run the server with auto-reload
uvicorn websocket_server:app --reload --port 8765

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 📄 License

MIT License
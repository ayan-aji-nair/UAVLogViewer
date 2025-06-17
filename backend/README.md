# UAV Log Viewer Chatbot Backend

An agentic chatbot backend for analyzing UAV log data using LangChain and FastAPI.

## Features

- **Agentic Analysis**: Uses LangChain agents for flexible, reasoning-based UAV log analysis
- **Session Persistence**: Maintains chat history and context across sessions
- **Streaming Responses**: Real-time token-by-token response streaming
- **Context Awareness**: Integrates UAV log data context for informed analysis
- **Redis Integration**: Optional Redis backend for session storage (falls back to in-memory)

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Redis for session persistence
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Run the Application

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the built-in runner
python main.py
```

## API Endpoints

### Chat Endpoints

- `POST /api/chat/message` - Send a message and get a response
- `POST /api/chat/stream` - Stream response token by token
- `GET /api/chat/history/{session_id}` - Get chat history
- `POST /api/chat/context/{session_id}` - Update session context
- `POST /api/chat/session` - Create a new session

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Usage Examples

### Send a Message

```bash
curl -X POST "http://localhost:8000/api/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze this flight log for any anomalies",
    "context_data": {
      "log_file": "flight_001.log",
      "flight_duration": "15 minutes",
      "max_altitude": "120m"
    }
  }'
```

### Stream Response

```bash
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What patterns do you see in the altitude data?",
    "session_id": "existing-session-id"
  }'
```

## Architecture

### Components

1. **UAVChatbotService**: Main service orchestrating LangChain agents
2. **SessionManager**: Handles chat session persistence
3. **UAVAnalysisTool**: LangChain tool for log analysis
4. **FastAPI Routes**: RESTful API endpoints

### Agentic Behavior

The chatbot uses LangChain agents to provide flexible, reasoning-based analysis:

- **Dynamic Pattern Recognition**: Identifies anomalies and patterns without rigid rules
- **Context-Aware Analysis**: Considers multiple factors and correlations
- **Adaptive Responses**: Provides insights based on the specific log data context

### Session Management

- **Redis Backend**: Primary storage with TTL-based expiration
- **In-Memory Fallback**: Automatic fallback if Redis unavailable
- **Context Persistence**: Maintains UAV log context across messages

## Development

### Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── api/
│   │   └── chat.py        # Chat API endpoints
│   ├── models/
│   │   └── chat.py        # Pydantic models
│   ├── services/
│   │   ├── chatbot.py     # Main chatbot service
│   │   └── session_manager.py  # Session management
│   └── config.py          # Configuration settings
└── env.example            # Environment variables template
```

### Adding New Analysis Tools

To add new UAV analysis capabilities:

1. Create a new tool class inheriting from `BaseTool`
2. Implement the `_run` method with your analysis logic
3. Add the tool to the `tools` list in `UAVChatbotService`

### Customizing the Agent

Modify the system prompt in `UAVChatbotService` to change the agent's behavior and analysis approach.

## Integration with Frontend

The existing Vue.js frontend has a chatbot checkbox in the sidebar. To integrate:

1. Create a chat interface component
2. Connect to the FastAPI endpoints
3. Handle session management and context updates
4. Display streaming responses

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**: Set `OPENAI_API_KEY` in your `.env` file
2. **Redis Connection Failed**: The app will automatically fall back to in-memory storage
3. **CORS Errors**: Update `CORS_ORIGINS` in your `.env` file to include your frontend URL

### Debug Mode

Enable debug mode by setting `DEBUG=true` in your `.env` file for detailed logging. 
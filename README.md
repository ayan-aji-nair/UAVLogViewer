# UAV Log Viewer with AI Chatbot

A comprehensive UAV (Unmanned Aerial Vehicle) log analysis tool with an AI-powered chatbot that can analyze flight data, answer questions about performance metrics, and provide insights using ArduPilot documentation and vector search.

## Features

- **Log File Support**: Upload and analyze various UAV log formats (MAVLink, DataFlash, DJI, etc.)
- **AI Chatbot**: Ask questions about your flight data using natural language
- **Vector Search**: Intelligent search through ArduPilot documentation
- **3D Visualization**: Interactive 3D flight path visualization using Cesium
- **Real-time Analysis**: Comprehensive analysis of altitude, GPS, battery, and other flight metrics
- **Anomaly Detection**: Automatic detection of flight anomalies and issues

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (for the AI chatbot)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd UAVLogViewer
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# Required: OpenAI API Key for the AI chatbot
OPENAI_API_KEY=your_openai_api_key_here

# Optional: OpenAI Model (default: gpt-4)
OPENAI_MODEL=gpt-4

# Optional: Cesium Token for 3D maps (get free token from https://cesium.com/ion/signup/)
VUE_APP_CESIUM_TOKEN=your_cesium_token_here

# Optional: Debug mode (default: false)
DEBUG=false
```

### 3. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 4. Access the Application

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Architecture

The application consists of three main services:

### Frontend (Vue.js)
- **Port**: 8080
- **Purpose**: Web interface for log upload and visualization
- **Features**: 3D flight path visualization, log file upload, interactive charts

### Backend (FastAPI + Python)
- **Port**: 8000
- **Purpose**: API server with AI chatbot and log processing
- **Features**: 
  - AI-powered chatbot using OpenAI GPT
  - Vector database for ArduPilot documentation search
  - Log file parsing and analysis
  - Session management with Redis

### Redis
- **Port**: 6379
- **Purpose**: Session storage and caching
- **Features**: Chat session persistence, temporary data storage

## Usage Guide

### 1. Upload Log Files

1. Open http://localhost:8080 in your browser
2. Click "Upload" to select your UAV log file
3. Supported formats: `.tlog`, `.bin`, `.log`, `.ulg`, etc.
4. Wait for the file to be processed

### 2. Use the AI Chatbot

1. After uploading a log file, click on the chat icon
2. Ask questions about your flight data, such as:
   - "What is the maximum altitude reached?"
   - "Show me the GPS coordinates"
   - "Are there any anomalies in the flight?"
   - "What was the average battery voltage?"
   - "Did the UAV experience any sudden altitude changes?"

### 3. View 3D Visualization

1. Navigate to the 3D view tab
2. Explore your flight path in 3D space
3. Use mouse controls to rotate, zoom, and pan
4. Toggle different data layers (GPS, altitude, etc.)

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key for the AI chatbot | `sk-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4` | `gpt-3.5-turbo` |
| `VUE_APP_CESIUM_TOKEN` | Cesium token for 3D maps | None | `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` |
| `DEBUG` | Enable debug mode | `false` | `true` |

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Redis

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

## API Endpoints

### Chat Endpoints

- `POST /api/chat` - Send a message to the AI chatbot
- `GET /api/chat/sessions/{session_id}` - Get chat history
- `POST /api/chat/sessions/{session_id}/context` - Update session context

### File Upload

- `POST /api/upload` - Upload a log file for analysis

### Vector Database

- `POST /api/vector/init` - Initialize vector database with ArduPilot documentation
- `GET /api/vector/status` - Check vector database status

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is valid and has sufficient credits
   - Check the key is correctly set in the `.env` file

2. **Vector Database Not Ready**
   - The system automatically initializes the vector database on first use
   - You can manually trigger initialization: `POST /api/vector/init`

3. **Port Conflicts**
   - Ensure ports 8000, 8080, and 6379 are available
   - Modify ports in `docker-compose.yml` if needed

4. **Memory Issues**
   - Large log files may require more memory
   - Increase Docker memory limits if needed

### Logs and Debugging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis

# Access backend container
docker-compose exec backend bash

# Check vector database status
curl http://localhost:8000/api/vector/status
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation at http://localhost:8000/docs

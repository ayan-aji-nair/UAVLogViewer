# Docker Setup for UAV Log Viewer

This project now includes a complete Docker setup with Redis, FastAPI backend, and Vue.js frontend.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key (for chatbot functionality)
- Cesium Ion token (for 3D mapping)

### Environment Setup
1. Copy the environment example file:
   ```bash
   cp backend/env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   VUE_APP_CESIUM_TOKEN=your_cesium_ion_token_here
   ```

### Development Environment
```bash
# Start development environment with hot reload
make dev

# Or manually:
docker-compose --profile dev up -d
```

### Legacy Frontend-Only Mode
If you want to run just the frontend like your original setup:
```bash
# Replicates your original Docker run command
make legacy-run

# Or manually:
docker build -t uavlogviewer .
docker run -e VUE_APP_CESIUM_TOKEN=$VUE_APP_CESIUM_TOKEN -it -p 8080:8080 -v ${PWD}:/usr/src/app uavlogviewer
```

### Production Environment
```bash
# Build and start production environment
make prod

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 📁 Docker Files Structure

```
├── docker-compose.yml              # Main compose file
├── docker-compose.prod.yml         # Production overrides
├── docker-compose.override.yml     # Development overrides
├── Dockerfile                      # Frontend development
├── Dockerfile.prod                 # Frontend production
├── backend/Dockerfile              # Backend container
├── nginx.conf                      # Nginx configuration
├── Makefile                        # Convenient commands
└── .dockerignore                   # Exclude files from build
```

## 🛠️ Available Commands

### Development Commands
```bash
make help          # Show all available commands
make dev           # Start development environment
make legacy-run    # Run with original Docker command (frontend only)
make build         # Build all containers
make up            # Start all services
make down          # Stop all services
make logs          # Show logs for all services
make clean         # Remove containers, networks, and volumes
```

### Production Commands
```bash
make prod          # Start production environment
make prod-build    # Build production containers
```

### Debug Commands
```bash
make debug         # Start with Redis Commander
make redis-cli     # Access Redis CLI
make backend-shell # Access backend container shell
make frontend-shell # Access frontend container shell
```

### Individual Services
```bash
make backend       # Start only backend + Redis
make frontend      # Start only frontend
make redis         # Start only Redis
```

## 🌐 Service URLs

### Development
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Redis Commander**: http://localhost:8081 (debug mode)

### Production
- **Frontend**: http://localhost (via Nginx)
- **Backend API**: http://localhost/api (via Nginx)
- **API Docs**: http://localhost/docs (via Nginx)

## 🔧 Configuration

### Environment Variables

#### Required
- `OPENAI_API_KEY`: Your OpenAI API key for chatbot functionality
- `VUE_APP_CESIUM_TOKEN`: Your Cesium Ion token for 3D mapping

#### Optional
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4)
- `DEBUG`: Enable debug mode (default: false)
- `REDIS_HOST`: Redis host (default: redis)
- `REDIS_PORT`: Redis port (default: 6379)

### Redis Configuration
- **Development**: In-memory fallback if Redis unavailable
- **Production**: Persistent Redis with AOF (Append Only File)
- **Memory Limit**: 256MB in production with LRU eviction

## 🏗️ Architecture

### Development Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Backend   │    │    Redis    │
│   (Vue.js)  │◄──►│  (FastAPI)  │◄──►│   (Cache)   │
│   :8080     │    │   :8000     │    │   :6379     │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Production Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Nginx    │    │   Backend   │    │    Redis    │
│  (Reverse   │◄──►│  (FastAPI)  │◄──►│   (Cache)   │
│   Proxy)    │    │   :8000     │    │   :6379     │
│    :80      │    └─────────────┘    └─────────────┘
└─────────────┘
       │
       ▼
┌─────────────┐
│   Frontend  │
│  (Static)   │
│   (Built)   │
└─────────────┘
```

## 🔍 Monitoring and Debugging

### Health Checks
- **Backend**: `GET /health` endpoint
- **Redis**: Redis CLI ping
- **Frontend**: Nginx serves static files

### Logs
```bash
# All services
make logs

# Individual services
make backend-logs
make frontend-logs
```

### Redis Management
```bash
# Access Redis CLI
make redis-cli

# Monitor Redis (in CLI)
MONITOR

# Check Redis info
INFO
```

## 🚀 Deployment

### Local Production Build
```bash
# Build production images
make prod-build

# Start production environment
make prod
```

### Cloud Deployment
1. Set environment variables in your cloud platform
2. Use `docker-compose.prod.yml` for production
3. Configure SSL certificates in `nginx.conf`
4. Set up proper domain names in CORS settings

### Environment-Specific Configurations
- **Development**: Hot reload, debug mode, volume mounts
- **Production**: Optimized builds, Nginx reverse proxy, rate limiting
- **Debug**: Additional Redis Commander for database inspection

## 🔒 Security Features

### Production Security
- **Rate Limiting**: API endpoints (10 req/s), Chat (5 req/s)
- **Security Headers**: XSS protection, content type options
- **Non-root Users**: All containers run as non-root
- **Resource Limits**: Memory limits on containers

### Network Security
- **Internal Network**: Services communicate via Docker network
- **Port Exposure**: Only necessary ports exposed
- **CORS Configuration**: Proper cross-origin settings

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using the ports
   lsof -i :8080
   lsof -i :8000
   lsof -i :6379
   ```

2. **Redis Connection Issues**
   ```bash
   # Check Redis status
   make redis-cli
   PING
   ```

3. **Backend Not Starting**
   ```bash
   # Check backend logs
   make backend-logs
   
   # Check environment variables
   docker-compose exec backend env | grep OPENAI
   ```

4. **Frontend Build Issues**
   ```bash
   # Rebuild frontend
   docker-compose build frontend
   ```

5. **Cesium Token Issues**
   ```bash
   # Check if token is set
   echo $VUE_APP_CESIUM_TOKEN
   
   # Rebuild with token
   VUE_APP_CESIUM_TOKEN=your_token make build
   ```

### Clean Slate
```bash
# Stop everything and clean up
make clean

# Rebuild and start
make build
make dev
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Vue.js Docker Guide](https://vuejs.org/guide/quick-start.html#with-build-tools)
- [Redis Docker Guide](https://redis.io/docs/stack/get-started/installation/docker/)
- [Cesium Ion Documentation](https://cesium.com/docs/tutorials/getting-started/) 
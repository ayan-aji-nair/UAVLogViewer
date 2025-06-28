@echo off
REM UAV Log Viewer Quick Start Script for Windows
REM This script helps you set up and run the UAV Log Viewer with Docker

echo 🚁 UAV Log Viewer Quick Start
echo ==============================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first:
    echo    https://docs.docker.com/desktop/install/windows/
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first:
    echo    https://docs.docker.com/compose/install/
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are installed

REM Check if .env file exists
if not exist .env (
    echo 📝 Creating .env file from template...
    if exist env.example (
        copy env.example .env >nul
        echo ✅ Created .env file from env.example
        echo.
        echo ⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key:
        echo    1. Get your API key from: https://platform.openai.com/api-keys
        echo    2. Edit .env file and replace 'sk-your_openai_api_key_here' with your actual key
        echo    3. Run this script again
        echo.
        echo    notepad .env
        pause
        exit /b 0
    ) else (
        echo ❌ env.example file not found. Please create a .env file manually.
        pause
        exit /b 1
    )
)

REM Check if OPENAI_API_KEY is set
findstr "sk-your_openai_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo ❌ Please update your OpenAI API key in .env file
    echo    Get your API key from: https://platform.openai.com/api-keys
    pause
    exit /b 1
)

echo ✅ Environment variables are configured

REM Pull latest images and start services
echo 🐳 Starting UAV Log Viewer with Docker Compose...
docker-compose up -d

echo.
echo 🎉 UAV Log Viewer is starting up!
echo.
echo 📱 Access the application:
echo    Frontend: http://localhost:8080
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo 📊 Monitor logs:
echo    docker-compose logs -f
echo.
echo 🛑 Stop the application:
echo    docker-compose down
echo.
echo 🚀 Happy flying!
pause 
#!/bin/bash

# UAV Log Viewer Quick Start Script
# This script helps you set up and run the UAV Log Viewer with Docker

set -e

echo "🚁 UAV Log Viewer Quick Start"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Created .env file from env.example"
        echo ""
        echo "⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key:"
        echo "   1. Get your API key from: https://platform.openai.com/api-keys"
        echo "   2. Edit .env file and replace 'sk-your_openai_api_key_here' with your actual key"
        echo "   3. Run this script again"
        echo ""
        echo "   nano .env  # or use your preferred editor"
        exit 0
    else
        echo "❌ env.example file not found. Please create a .env file manually."
        exit 1
    fi
fi

# Check if OPENAI_API_KEY is set
if grep -q "sk-your_openai_api_key_here" .env; then
    echo "❌ Please update your OpenAI API key in .env file"
    echo "   Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

echo "✅ Environment variables are configured"

# Pull latest images and start services
echo "🐳 Starting UAV Log Viewer with Docker Compose..."
docker-compose up -d

echo ""
echo "🎉 UAV Log Viewer is starting up!"
echo ""
echo "📱 Access the application:"
echo "   Frontend: http://localhost:8080"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📊 Monitor logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop the application:"
echo "   docker-compose down"
echo ""
echo "🚀 Happy flying!" 
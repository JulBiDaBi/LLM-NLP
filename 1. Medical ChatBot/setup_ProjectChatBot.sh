#!/bin/bash

# This script sets up the environment for the Project ChatBot
# Define the project directory
PROJECT_DIR="$(pwd)"

# Create a virtual environment
# if [ ! -d "$PROJECT_DIR/venv" ]; then
    # echo "Creating virtual environment..."
    # python3 -m venv "$PROJECT_DIR/venv"
# else
    # echo "Virtual environment already exists."
# fi

# Activate the virtual environment
# source "$PROJECT_DIR/venv/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Installing required packages..."
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping package installation."
fi

echo "Project ChatBot environment setup complete."

# Root file for the Project ChatBot
touch "$PROJECT_DIR/README.md"
# touch "$PROJECT_DIR/.requirements.txt"
touch "$PROJECT_DIR/.gitignore"

# Add folder Data
mkdir -p "$PROJECT_DIR/Data"
mkdir -p "$PROJECT_DIR/Data/raw"
mkdir -p "$PROJECT_DIR/Data/processed"

# API
mkdir -p "$PROJECT_DIR/api/routes"
touch "$PROJECT_DIR/api/main.py"

# Chatbot
mkdir -p "$PROJECT_DIR/chatbot/langchain"
mkdir -p "$PROJECT_DIR/chatbot/prompts"
mkdir -p "$PROJECT_DIR/chatbot/memory"

# Pinecone
mkdir -p "$PROJECT_DIR/pinecone"
touch "$PROJECT_DIR/pinecone/index.py"
touch "$PROJECT_DIR/pinecone/query.py"

# Utils
mkdir -p "$PROJECT_DIR/utils"
touch "$PROJECT_DIR/utils/preprocessing.py"
touch "$PROJECT_DIR/utils/validation.py"

# Tests
mkdir -p "$PROJECT_DIR/tests"
touch "$PROJECT_DIR/tests/test_api.py"
touch "$PROJECT_DIR/tests/test_chatbot.py"

# Frontend
mkdir -p "$PROJECT_DIR/frontend"
touch "$PROJECT_DIR/frontend/streamlit_app.py"

# Docker
touch "$PROJECT_DIR/Dockerfile"
touch "$PROJECT_DIR/docker/Dockerfile"
touch "$PROJECT_DIR/docker-compose.yml"

echo "Project ChatBot structure created successfully."
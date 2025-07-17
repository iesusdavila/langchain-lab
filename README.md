# LangChain Lab

A collection of LangChain experiments and projects exploring different AI capabilities including chains, agents, RAG systems, and memory management.

## Project Structure

### Learning Modules

- **class_1/** - Basic LLM chains and structured output parsing
- **class_2/** - Client-server applications with FastAPI
- **class_3/** - Retrieval systems and simple RAG implementation
- **class_4/** - PDF processing with Groq and LlamaCpp
- **class_5/** - Conversation memory patterns
- **class_6/** - Agent implementations

### Projects

- **projects/games/** - Game agent implementation
- **projects/ros_consultant/** - ROS documentation chatbot with API backend

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Add your API keys
```

## Usage

Each class folder contains standalone examples. Run individual scripts:

```bash
python3 class_1/basic_chain.py
```

For the ROS consultant project:

```bash
# Start backend
python3 projects/ros_consultant/backend/api_server.py

# Run frontend
streamlit run projects/ros_consultant/frontend/client.py
```

## Models (GGUF for LlamaCPP)

Local models are stored in `models/` directory:
- Mistral-7B-Instruct
- Meta-Llama-3.1-8B-Instruct
- Llama-3.2-1B-Instruct
- mxbai-embed-large-v1

# Derma RAG Chatbot

A dermatology assistant chatbot using RAG + LLM + image processing.

## Requirements
- macOS
- Python 3.10
- Node.js 22+
- Ollama

## Setup Steps

### 1. Create Python virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama
```bash
brew install ollama
ollama pull llama3.2:1b
```

### 3. Fix paths for your machine
```bash
python fix_paths.py
```

### 4. Run the app

Terminal 1:
```bash
ollama serve
```

Terminal 2:
```bash
source venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 3:
```bash
cd frontend/frontend
npm install
npm run dev
```

### 5. Open browser
Go to http://localhost:5173

## Notes
- vectorstore/ and processed/ folders must be present (get from project owner)
- data/textbook_images/ folder must be present (get from project owner)
- data/textbooks/ folder must be present (get from project owner)

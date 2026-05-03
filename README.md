## Setup Instructions

### Requirements
- Python 3.10
- Node.js 22
- Ollama

### Backend
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Ollama
```bash
ollama pull llama3.2:1b
ollama serve
```

### Run Backend
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Run Frontend
```bash
cd frontend/frontend
npm install
npm run dev
```

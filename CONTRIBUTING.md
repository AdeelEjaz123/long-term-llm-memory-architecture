# Contributing to Semantic RAG Conversational Memory Engine

Thank you for your interest in contributing! We welcome contributions, whether it's fixing bugs, improving documentation, adding new vector database adapters, or optimizing the memory retrieval pipeline.

---

## 🛠️ Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/long-term-llm-memory-architecture.git
   cd long-term-llm-memory-architecture
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Start PostgreSQL with pgvector:**
   ```bash
   docker compose up -d
   ```

4. **Verify Ollama models are running locally:**
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text:latest
   ```

---

## 🧪 Running Tests

Always ensure the end-to-end integration test passes before submitting a pull request:
```bash
python run_test.py
```

---

## 📐 Code Style & Conventions

- Format code cleanly according to **PEP 8**.
- Use descriptive variable and function names.
- Type annotate function arguments and return types.
- Write docstrings for new services and database helper functions.

---

## 🚀 Submitting a Pull Request

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes with conventional commit messages:
   ```bash
   git commit -m "feat: Add hybrid reciprocal rank fusion for memory search"
   ```
3. Push to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
4. Open a Pull Request on GitHub against `main`. Fill in the PR template.


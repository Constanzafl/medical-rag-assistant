# 🏥 Medical RAG Assistant — Consenso HTA SAC 2025

A Retrieval-Augmented Generation (RAG) system that answers clinical questions based on the Argentine Consensus on Arterial Hypertension (SAC 2025). Built as a portfolio project demonstrating the end-to-end RAG pipeline.

## Architecture

```
PDF Document → Loader (PyMuPDF) → Chunking → Embeddings (HuggingFace) → ChromaDB
                                                                            ↓
User Question → Embedding → Similarity Search → Context + Question → LLM (Groq/Llama 3) → Answer
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| LLM | openai/gpt-oss-120b| Fast inference, free tier |
| Embeddings | sentence-transformers (multilingual) | Local, free, handles Spanish |
| Vector Store | ChromaDB | Simple, persistent, no server needed |
| Orchestration | LangChain | Standard RAG pipeline |
| PDF Loading | PyMuPDF | Good with tables and formatted text |

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/Constanzafl/medical-rag-assistant.git
cd medical-rag-assistant

# 2. Install dependencies (Poetry crea el entorno virtual automáticamente)
poetry install

# 3. Activate the virtual environment
poetry shell

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Groq API key

# 5. Add your PDF to data/
# Place the SAC 2025 consensus PDF in the data/ folder

# 6. Run ingestion (processes PDF → vector store)
python src/ingest.py data/consenso_hta_sac_2025.pdf

# 7. Start the assistant
python src/app.py
```

## Project Structure

```
medical-rag-assistant/
├── data/                   # Source PDFs
├── src/
│   ├── loader.py           # PDF loading and chunking
│   ├── vectorstore.py      # Embeddings + ChromaDB operations
│   ├── chain.py            # RAG chain (retrieval + generation)
│   ├── ingest.py           # One-time ingestion script
│   └── app.py              # CLI interface
├── chroma_db/              # Persisted vector store (auto-generated)
├── .env.example            # Environment variables template
├── requirements.txt
└── README.md
```

## Example Questions

```
- ¿Cómo se clasifica la hipertensión arterial?
- ¿Cuáles son los valores objetivo de presión arterial?
- ¿Qué fármacos se recomiendan como primera línea?
- ¿Cuándo se considera hipertensión resistente?
```

## Design Decisions

- **Medical document as source**: Using a real clinical guideline allows validating response accuracy from domain expertise — a deliberate choice to demonstrate that RAG quality depends on retrieval quality, not just the LLM.
- **Multilingual embeddings**: The SAC consensus is in Spanish; `paraphrase-multilingual-MiniLM-L12-v2` handles this well without translating the source.
- **Temperature 0**: Factual medical content requires deterministic responses.
- **Source attribution**: Every response includes page references for traceability.

## Next Steps

- [ ] Add Streamlit UI
- [ ] Implement conversational memory (multi-turn)
- [ ] Add agent with tools (PubMed search, document comparison)
- [ ] Evaluation framework for response quality

---

Built by [Constanza Florio](https://www.linkedin.com/in/mariaconstanzaflorio) — Data Scientist with a medical background.

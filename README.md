# Universal RAG AI

A production-grade, fully local RAG (Retrieval-Augmented Generation) API built with FastAPI, LangChain, and Ollama. Features a modern web UI, multi-modal ingestion (PDF, Images, Instagram), a Universal Cloud Router (OpenAI, DeepSeek, etc.), and a built-in Prompt Optimizer to save API tokens.

## 🚀 Features
- **Multi-Modal Ingestion:** Upload PDFs, Images (JPG/PNG), or scrape Instagram URLs.
- **OCR Support:** Uses PyMuPDF and Tesseract for scanned documents and images.
- **Hybrid AI Routing:** Switch seamlessly between local Ollama models and any OpenAI-compatible Cloud API.
- **Token Optimizer:** Uses a local LLM to compress prompts before sending them to paid APIs.
- **Vector Search:** Uses LanceDB for fast, local vector storage.
- **Dynamic Web UI:** Clean, dark-mode frontend for uploading documents and chatting.

## 📋 Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (for image extraction)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Marcelhod/rag-api.git
   cd rag-api
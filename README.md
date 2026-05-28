# 📄 RAG Document Q&A

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow) ![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple) ![Groq](https://img.shields.io/badge/Groq-Llama3-black) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

> Upload any PDF and ask questions in plain English. Get precise answers **with exact page citations** — powered by Retrieval-Augmented Generation (RAG).

🚀 **Live Demo:** [rag-document-app-b38xffdguamazzrketgxh6.streamlit.app](https://rag-document-app-b38xffdguamazzrketgxh6.streamlit.app/)

---

## 🧩 Problem Statement

Reading long PDFs (research papers, legal docs, textbooks) and manually hunting for specific information wastes hours. Generic chatbots hallucinate answers when working with your private documents. This app solves both: it **embeds your PDF into a local vector store** and uses a LLM only to synthesise answers from retrieved passages — so every answer is grounded in your actual document, with a page number to verify.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Streamlit UI)                      │
│            Upload PDF  ──→  Ask a Question                   │
└────────────────┬────────────────────────┬───────────────────┘
                 │                        │
    ┌────────────▼──────────┐   ┌─────────▼──────────────────┐
    │   rag_pipeline.py     │   │        rag_qa.py            │
    │  (Indexing Pipeline)  │   │    (Query Pipeline)         │
    └────────────┬──────────┘   └─────────┬──────────────────┘
                 │                        │
    ┌────────────▼──────────┐   ┌─────────▼──────────────────┐
    │  PyPDFLoader           │   │  FAISS Vector Search       │
    │  (Extract text +       │   │  (Top-k relevant chunks)   │
    │   page numbers)        │   └─────────┬──────────────────┘
    └────────────┬──────────┘             │
                 │               ┌────────▼───────────────────┐
    ┌────────────▼──────────┐    │   Groq API (Llama 3)       │
    │  HuggingFace Embeddings│    │   (Answer generation with  │
    │  all-MiniLM-L6-v2     │    │    retrieved context)      │
    │  (Runs fully local)   │    └────────────────────────────┘
    └────────────┬──────────┘
                 │
    ┌────────────▼──────────┐
    │  FAISS Index           │
    │  (Persisted locally)   │
    └───────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📤 PDF Upload | Upload any PDF — textbooks, papers, contracts |
| 🔍 Semantic Search | Finds the most relevant passages, not just keywords |
| 💬 Natural Language Q&A | Ask questions in plain English |
| 📖 Page Citations | Every answer references the exact page number |
| 🏠 Local Embeddings | HuggingFace model runs on your machine — no data sent |
| ⚡ Fast Inference | Groq's ultra-fast Llama 3 for near-instant responses |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG Orchestration | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS |
| LLM | Groq API — Llama 3 8B |
| PDF Parsing | LangChain PyPDFLoader |

---

## 📁 Project Structure

```
rag-document-qa/
├── app.py               # Streamlit frontend
├── rag_pipeline.py      # PDF ingestion + FAISS index creation
├── rag_qa.py            # Query handling + LLM answer generation
├── requirements.txt
├── .gitignore
├── faiss_index/         # Persisted vector index (auto-generated)
│   ├── index.faiss
│   └── index.pkl
└── sample_docs/         # Example PDFs to test with
```

---

## 🚀 Setup & Run Locally

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (takes 30 seconds)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/bharadwabansi/rag-document-qa.git
cd rag-document-qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=your_key_here
# On Windows: set GROQ_API_KEY=your_key_here

# 4. Launch the app
streamlit run app.py
```

### Usage

1. Open `http://localhost:8501`
2. Upload a PDF using the sidebar uploader
3. Wait for indexing to complete (~10 seconds for a 50-page doc)
4. Type any question in the chat box
5. Get your answer with the page number it came from

---

## 📸 App Preview

```
┌──────────────────────────────────────────────────────┐
│  📄 RAG Document Q&A                                 │
│  ─────────────────────────────────────────────────── │
│  [Upload PDF]  Transformers.pdf ✅ Indexed          │
│                                                       │
│  You:  What is the proposed methodology?              │
│                                                       │
│  Assistant:  The paper proposes a transformer-based  │
│  approach that... [Source: Page 4]                   │
│                                                       │
│  You:  how decoder is different from encoder?                     │
│  Assistant:  Results showed a 12% improvement in...  │
│  [Source: Page 3]                                    │
└──────────────────────────────────────────────────────┘
```
Upload PDF
<img width="1899" height="847" alt="image" src="https://github.com/user-attachments/assets/4b845625-e549-43c1-ad42-a85bfee33953" />
Indexing is done
<img width="1900" height="833" alt="image" src="https://github.com/user-attachments/assets/206f6f65-320e-4c7c-b473-310bb5b16b9c" />
Type any question
<img width="1890" height="852" alt="image" src="https://github.com/user-attachments/assets/19048555-298f-460b-80a1-f53ffd6c4490" />

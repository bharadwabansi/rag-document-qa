import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # runs locally, no API key

load_dotenv()

# ─── STEP 1: Load PDF ────────────────────────────────────────────────────────

def load_pdf(pdf_path: str):
    """Load a PDF file and return list of pages."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   ✅ Loaded {len(pages)} pages")
    return pages


# ─── STEP 2: Split into chunks ───────────────────────────────────────────────

def split_into_chunks(pages):
    """Split pages into smaller overlapping chunks."""
    print("\n✂️  Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~500 characters per chunk
        chunk_overlap=100,    # 100 char overlap so context isn't lost at edges
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(pages)
    print(f"   ✅ Created {len(chunks)} chunks from {len(pages)} pages")
    print(f"\n   📌 Sample chunk preview:")
    print("   " + "-" * 50)
    print("   " + chunks[0].page_content[:300].replace("\n", "\n   "))
    print("   " + "-" * 50)
    return chunks


# ─── STEP 3: Create embeddings + save to FAISS ───────────────────────────────

def create_vector_store(chunks, save_path: str = "faiss_index"):
    """
    Convert chunks to embeddings using HuggingFace (FREE, runs on your laptop)
    No API key needed. Downloads a small model (~90MB) first time only.
    """
    print("\n🔢 Creating embeddings using HuggingFace (runs locally, 100% free)...")
    print("   First time: downloads model ~90MB — wait a minute")
    print("   After that: instant from cache\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",   # small, fast, accurate — perfect for RAG
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"   ✅ Vector store saved to '{save_path}/'")
    return vector_store


# ─── STEP 4: Test retrieval ──────────────────────────────────────────────────

def test_retrieval(vector_store, query: str):
    """Test that retrieval works by searching for a query."""
    print(f"\n🔍 Testing retrieval with query: '{query}'")

    results = vector_store.similarity_search(query, k=3)

    print(f"   ✅ Found {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] Page {doc.metadata.get('page', '?')} —")
        print("   " + doc.page_content[:200].replace("\n", " "))
        print()


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Put any PDF inside sample_docs/ and rename it sample.pdf
    PDF_PATH = "sample_docs/sample.pdf"

    if not os.path.exists(PDF_PATH):
        print(f"\n❌ PDF not found at: {PDF_PATH}")
        print("   → Create a folder called sample_docs/ in your project")
        print("   → Drop any PDF inside it and rename it sample.pdf\n")
        exit(1)

    pages        = load_pdf(PDF_PATH)
    chunks       = split_into_chunks(pages)
    vector_store = create_vector_store(chunks)
    test_retrieval(vector_store, "What is this document about?")

    print("\n🎉 Indexing complete! Your vector store is ready.")
    print("   Next: we build the question-answering pipeline using Groq (free LLM)\n")
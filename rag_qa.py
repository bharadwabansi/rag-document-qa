import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# ─── STEP 1: Load the saved vector store ─────────────────────────────────────

def load_vector_store(save_path: str = "faiss_index"):
    """Load the FAISS vector store we built in rag_pipeline.py"""
    print("\n📂 Loading vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # safe — we created this file ourselves
    )

    print("   ✅ Vector store loaded successfully")
    return vector_store


# ─── STEP 2: Set up Groq LLM ─────────────────────────────────────────────────

def load_llm():
    """Load the free Groq LLM (Llama 3)"""
    print("\n🤖 Connecting to Groq LLM (Llama 3, free)...")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",        # free, fast, good quality
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,               # lower = more factual, less creative
        max_tokens=1024
    )

    print("   ✅ Groq LLM ready")
    return llm


# ─── STEP 3: Build the prompt template ───────────────────────────────────────

def build_prompt():
    """
    This prompt tells the LLM exactly how to behave:
    - Only use the context we retrieved
    - Always mention which page the answer came from
    - Say 'I don't know' if the answer isn't in the document
    """
    template = """
You are a helpful assistant that answers questions based ONLY on the provided document context.

Rules:
- Answer using only the context below. Do not use outside knowledge.
- Always mention the page number(s) where you found the answer.
- If the answer is not in the context, say: "I couldn't find this in the document."
- Keep answers clear and concise.

Context from document:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# ─── STEP 4: Build the full RAG chain ────────────────────────────────────────

def build_qa_chain(vector_store, llm):
    """
    RetrievalQA chain:
    1. Takes a question
    2. Retrieves top 4 relevant chunks from FAISS
    3. Passes chunks + question to Groq LLM
    4. Returns the answer
    """
    print("\n⛓️  Building RAG chain...")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",            # "stuff" = put all chunks into one prompt
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 4}     # retrieve top 4 most relevant chunks
        ),
        return_source_documents=True,  # so we can show which pages were used
        chain_type_kwargs={
            "prompt": build_prompt()
        }
    )

    print("   ✅ RAG chain ready")
    return qa_chain


# ─── STEP 5: Ask a question ───────────────────────────────────────────────────

def ask_question(qa_chain, question: str):
    """Ask a question and display the answer with sources."""
    print(f"\n{'='*55}")
    print(f"❓ Question: {question}")
    print(f"{'='*55}")

    result = qa_chain.invoke({"query": question})

    # Print the answer
    print(f"\n💬 Answer:\n")
    print(f"   {result['result']}\n")

    # Print source pages used
    print("📄 Sources used:")
    seen_pages = set()
    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "?")
        if page not in seen_pages:
            seen_pages.add(page)
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"   • Page {page}: {preview}...")

    print()


# ─── MAIN: Interactive Q&A loop ───────────────────────────────────────────────

if __name__ == "__main__":

    # Load everything
    vector_store = load_vector_store()
    llm          = load_llm()
    qa_chain     = build_qa_chain(vector_store, llm)

    print("\n" + "="*55)
    print("🎉 RAG Q&A ready! Ask anything about your document.")
    print("   Type 'quit' to exit")
    print("="*55)

    # Ask a few automatic questions first to show it works
    sample_questions = [
        "What is this document about?",
        "What are the main topics covered?"
    ]

    for q in sample_questions:
        ask_question(qa_chain, q)

    # Then go interactive — user types their own questions
    print("\n💬 Now ask your own questions:\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ["quit", "exit", "q"]:
            print("\n👋 Bye! Push your code to GitHub before closing.\n")
            break
        if user_q:
            ask_question(qa_chain, user_q)
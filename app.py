import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
<style>
.source-box {
    background: #f1f3f4;
    border-left: 3px solid #4CAF50;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.83em;
    color: #444;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None   # (pages, chunks)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1024
    )

def index_pdf(file_bytes: bytes):
    """Convert PDF bytes → FAISS vector store. Returns (vector_store, pages, chunks)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    pages = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    return vector_store, len(pages), len(chunks)

def make_qa_chain(vector_store):
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer ONLY from the context below.
Always mention the page number(s) where you found the answer.
If the answer is not in the context, say: "I couldn't find this in the document."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        # Only re-index if it's a different file
        if st.session_state.doc_name != uploaded.name:
            with st.spinner("🔍 Indexing PDF... please wait"):
                vs, n_pages, n_chunks = index_pdf(uploaded.read())
                st.session_state.qa_chain  = make_qa_chain(vs)
                st.session_state.doc_name  = uploaded.name
                st.session_state.doc_stats = (n_pages, n_chunks)
                st.session_state.messages  = []
            st.success("✅ Done! Ask your questions →")
            st.rerun()   # ← forces main area to refresh immediately

    if st.session_state.doc_stats:
        st.divider()
        st.markdown(f"**📄 {st.session_state.doc_name}**")
        c1, c2 = st.columns(2)
        c1.metric("Pages",  st.session_state.doc_stats[0])
        c2.metric("Chunks", st.session_state.doc_stats[1])

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("LangChain · FAISS · Groq · Streamlit")

# ─── MAIN AREA ────────────────────────────────────────────────────────────────

st.title("📄 RAG Document Q&A")
st.caption("Upload a PDF in the sidebar, then ask anything about it.")
st.divider()

# Case 1: No PDF uploaded yet
if st.session_state.qa_chain is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.markdown("""
**What you can ask once uploaded:**
- *"What is this document about?"*
- *"Summarise the key points"*
- *"What does it say about X?"*
- *"List the main findings from page 3"*
    """)

# Case 2: PDF is loaded — show chat
else:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">Page {s["page"]}: {s["preview"]}</div>',
                            unsafe_allow_html=True
                        )

    # Chat input — this is always visible when a PDF is loaded
    if question := st.chat_input("Ask something about your document..."):

        # Add + show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer from RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result  = st.session_state.qa_chain.invoke({"query": question})
                answer  = result["result"]
                src_docs = result["source_documents"]

            st.markdown(answer)

            # Deduplicate sources by page
            sources, seen = [], set()
            for doc in src_docs:
                page = doc.metadata.get("page", "?")
                if page not in seen:
                    seen.add(page)
                    sources.append({
                        "page": page,
                        "preview": doc.page_content[:150].replace("\n", " ") + "..."
                    })

            if sources:
                with st.expander("📄 Sources"):
                    for s in sources:
                        st.markdown(
                            f'<div class="source-box">Page {s["page"]}: {s["preview"]}</div>',
                            unsafe_allow_html=True
                        )

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
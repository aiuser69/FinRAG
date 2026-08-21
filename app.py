import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinRAG | Document Intelligence",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system: Enterprise Dashboard
# Sharp borders, muted slate colors, standard top navigation, 
# and highly structured content blocks.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
      --bg-color: #0F111A;
      --surface-color: #161A23;
      --border-color: #2D3342;
      --text-primary: #E2E8F0;
      --text-secondary: #94A3B8;
      --accent-color: #2563EB;
      --font-family: 'Inter', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
      background-color: var(--bg-color) !important;
      color: var(--text-primary) !important;
      font-family: var(--font-family) !important;
    }

    [data-testid="stHeader"] { display: none !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ---------- Top Navigation Bar ---------- */
    .top-nav {
      display: flex;
      align-items: center;
      background-color: var(--surface-color);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 32px;
      margin: -3rem -3rem 2rem -3rem; /* Offset Streamlit default padding */
      position: sticky;
      top: 0;
      z-index: 999;
    }
    .nav-brand {
      font-weight: 700;
      font-size: 1.1rem;
      letter-spacing: 0.05em;
      color: var(--text-primary);
      margin-right: 32px;
      text-transform: uppercase;
    }
    .nav-links {
      display: flex;
      gap: 24px;
    }
    .nav-links a {
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      transition: color 0.2s ease;
    }
    .nav-links a:hover {
      color: var(--text-primary);
    }

    /* ---------- Typography & Layout ---------- */
    .page-header {
      font-size: 2rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 0.5rem;
      letter-spacing: -0.02em;
    }
    .page-subheader {
      font-size: 1rem;
      color: var(--text-secondary);
      margin-bottom: 2.5rem;
    }
    
    .section-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--text-primary);
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 8px;
      margin: 2rem 0 1rem 0;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .section-text {
      color: var(--text-secondary);
      font-size: 0.95rem;
      line-height: 1.6;
    }

    /* ---------- Ordered List for Instructions ---------- */
    ol.instruction-list {
      color: var(--text-secondary);
      font-size: 0.95rem;
      line-height: 1.7;
      padding-left: 20px;
    }
    ol.instruction-list li strong {
      color: var(--text-primary);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
      background-color: var(--surface-color) !important;
      border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    
    .sidebar-header {
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-bottom: 1rem;
    }

    /* ---------- Chat Elements ---------- */
    [data-testid="stChatMessage"] { 
      background: transparent !important; 
      border-bottom: 1px solid rgba(45, 51, 66, 0.5);
      padding: 1.5rem 0;
    }
    .msg-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      margin-bottom: 8px;
    }
    .msg-label.system { color: var(--accent-color); }

    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
      background: var(--surface-color) !important; 
      border: 1px solid var(--border-color) !important;
      border-radius: 4px !important; 
      color: var(--text-primary) !important; 
    }
    [data-testid="stChatInput"]:focus-within { 
      border-color: var(--accent-color) !important; 
    }

    /* ---------- Citations / Expanders ---------- */
    [data-testid="stExpander"] { 
      border: 1px solid var(--border-color) !important; 
      border-radius: 4px !important; 
      background: var(--surface-color) !important; 
      margin-top: 16px !important;
    }
    [data-testid="stExpander"] summary {
      font-weight: 500 !important; 
      font-size: 0.85rem !important;
      color: var(--text-secondary) !important;
    }
    .citation-badge {
      display: inline-block;
      background: var(--bg-color);
      border: 1px solid var(--border-color);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: monospace;
      color: var(--text-primary);
      margin: 0 8px 8px 0;
    }
    .citation-text {
      font-size: 0.85rem;
      color: var(--text-secondary);
      border-left: 2px solid var(--border-color);
      padding-left: 12px;
      margin-top: 8px;
      margin-bottom: 16px;
    }

    /* ---------- File Uploader ---------- */
    [data-testid="stFileUploader"] { 
      background: var(--bg-color) !important; 
      border: 1px dashed var(--border-color) !important; 
      border-radius: 4px !important; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Setup Groq Client securely
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# 2. Cache embedding model
@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ---------------------------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-nav">
      <div class="nav-brand">FinRAG Terminal</div>
      <div class="nav-links">
        <a href="#about">About</a>
        <a href="#instructions">Instructions</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar - Document Upload
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">Data Ingestion</div>', unsafe_allow_html=True)

if "custom_db" not in st.session_state:
    st.session_state.custom_db = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded_file = st.sidebar.file_uploader("Upload target PDF", type="pdf")

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Indexing document vectors..."):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            page_documents = []

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    page_documents.append(Document(
                        page_content=text,
                        metadata={"page_num": page_num + 1, "doc_type": "Source Document"}
                    ))

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            chunks = text_splitter.split_documents(page_documents)
            st.session_state.custom_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.file_name = uploaded_file.name
        st.sidebar.success("Index complete. Document ready.")
else:
    st.session_state.custom_db = None
    st.session_state.file_name = None

st.sidebar.markdown(
    """
    <div style="margin-top: 2rem; font-size: 0.75rem; color: var(--text-secondary); line-height: 1.6;">
      <strong>SECURITY PROTOCOL</strong><br>
      Data is processed exclusively in temporary session memory. No disk persistence. Data is purged upon session termination.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main Area - Info Sections & Chat
# ---------------------------------------------------------------------------
st.markdown('<div class="page-header">Document Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subheader">Secure Retrieval-Augmented Generation (RAG) for analytical processing.</div>', unsafe_allow_html=True)

# Anchor for About
st.markdown('<div id="about" class="section-title">About</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-text">
      FinRAG is an enterprise-grade document analysis terminal designed to extract and synthesize information from dense financial filings, reports, and documentation. Utilizing localized vector embeddings and advanced language modeling, it ensures all outputs are strictly grounded in the provided source material to prevent data hallucination.
    </div>
    """,
    unsafe_allow_html=True
)

# Anchor for Instructions
st.markdown('<div id="instructions" class="section-title">Instructions</div>', unsafe_allow_html=True)
st.markdown(
    """
    <ol class="instruction-list">
      <li><strong>Ingest Document:</strong> Utilize the sidebar module to upload a target PDF file.</li>
      <li><strong>Vector Processing:</strong> Allow the system to autonomously chunk and embed the document content into the active session memory.</li>
      <li><strong>Execute Query:</strong> Input specific, targeted questions into the terminal interface below.</li>
      <li><strong>Audit Trail:</strong> Review the system's generated response and expand the "Audit Verification" panel to confirm page-level source citations.</li>
    </ol>
    """,
    unsafe_allow_html=True
)

st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)

# Stop if no file is attached
if st.session_state.custom_db is None:
    st.warning("SYSTEM HALT: Target document required for analysis. Please upload a file via the sidebar to proceed.")
    st.stop()

# Active Document Indicator
st.markdown(
    f"""
    <div style="background: var(--surface-color); border: 1px solid var(--border-color); padding: 12px 16px; border-radius: 4px; margin-bottom: 2rem; font-size: 0.85rem;">
      <span style="color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Active Target File:</span> 
      <strong style="color: var(--text-primary); margin-left: 8px;">{st.session_state.file_name}</strong>
    </div>
    """, 
    unsafe_allow_html=True
)

active_vectorstore = st.session_state.custom_db


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    with st.expander("VIEW AUDIT VERIFICATION"):
        for src in sources:
            page_num = src.get('page_num', 'N/A')
            st.markdown(f"<div class='citation-badge'>PAGE {page_num}</div>", unsafe_allow_html=True)
            excerpt = src.get('excerpt', '')
            st.markdown(f"<div class='citation-text'>\"{excerpt}...\"</div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        label_class = "system" if msg["role"] == "assistant" else "user"
        label = "FinRAG System" if msg["role"] == "assistant" else "Analyst"
        st.markdown(f"<div class='msg-label {label_class}'>{label}</div>", unsafe_allow_html=True)
        
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_prompt := st.chat_input("Enter query parameter..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown("<div class='msg-label user'>Analyst</div>", unsafe_allow_html=True)
        st.markdown(user_prompt)

    with st.spinner("Executing retrieval sequence..."):
        results = active_vectorstore.similarity_search(query=user_prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in results])

        system_prompt = f"""You are a strict, professional financial research assistant.
Answer ONLY using the provided context. If the context does not contain the answer, explicitly state:
"Data not found in the provided document." Do not hallucinate or use outside knowledge. Use a formal, objective tone.

Context:
{context}"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-20b",
        )
        answer = chat_completion.choices[0].message.content

        sources = [
            {
                "page_num": doc.metadata.get("page_num"),
                "excerpt": doc.page_content[:150].replace("\n", " ")
            }
            for doc in results
        ]

    with st.chat_message("assistant"):
        st.markdown("<div class='msg-label system'>FinRAG System</div>", unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

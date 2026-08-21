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
    page_title="FinRAG Workspace",
    page_icon="✨",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system: Modern Minimalist SaaS
# Soft dark backgrounds, clean card layouts, readable sans-serif typography,
# and elegant blue accents. No fake tech jargon.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
      --bg:         #0B0F19; /* Deep modern navy/black */
      --card-bg:    #111827; /* Slightly lighter card background */
      --card-alt:   #1F2937; /* Hover or alternate card */
      --border:     #374151;
      --accent:     #3B82F6; /* Clean modern blue */
      --accent-dim: rgba(59, 130, 246, 0.1);
      --text-main:  #F9FAFB;
      --text-muted: #9CA3AF;
      --font:       'Inter', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
      background-color: var(--bg) !important;
      color: var(--text-main) !important;
      font-family: var(--font) !important;
    }

    [data-testid="stHeader"] { background-color: transparent !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ---------- Headers & Titles ---------- */
    .main-title {
      font-weight: 700;
      font-size: 2.5rem;
      letter-spacing: -0.02em;
      color: var(--text-main);
      margin-bottom: 0.2rem;
    }
    .sub-title {
      font-size: 1.1rem;
      color: var(--text-muted);
      margin-bottom: 2rem;
      font-weight: 400;
    }
    
    /* ---------- Instructions Card ---------- */
    .instructions-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .instructions-card h3 {
      margin-top: 0;
      color: var(--text-main);
      font-size: 1.25rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .instructions-list {
      color: var(--text-muted);
      line-height: 1.6;
      margin-bottom: 0;
      padding-left: 20px;
    }
    .instructions-list li { margin-bottom: 8px; }
    .instructions-list strong { color: var(--text-main); }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
      background-color: var(--card-bg) !important;
      border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }
    
    .sidebar-title {
      font-weight: 600;
      font-size: 1.2rem;
      color: var(--text-main);
      margin-bottom: 1rem;
    }

    .privacy-note {
      background: var(--accent-dim);
      border-left: 4px solid var(--accent);
      padding: 12px 16px;
      border-radius: 4px 8px 8px 4px;
      margin-top: 24px;
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .privacy-note strong { color: var(--accent); }

    /* ---------- Chat Elements ---------- */
    [data-testid="stChatMessage"] { 
      background: transparent !important; 
      padding: 1rem 0;
    }
    .msg-avatar {
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 4px 8px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      display: inline-block;
      margin-bottom: 8px;
    }
    .msg-avatar.ai { color: var(--accent); background: var(--accent-dim); border-color: var(--accent); }

    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
      background: var(--card-bg) !important; 
      border: 1px solid var(--border) !important;
      border-radius: 12px !important; 
      color: var(--text-main) !important; 
    }
    [data-testid="stChatInput"]:focus-within { 
      border-color: var(--accent) !important; 
      box-shadow: 0 0 0 2px var(--accent-dim) !important; 
    }

    /* ---------- Citations / Expanders ---------- */
    [data-testid="stExpander"] { 
      border: 1px solid var(--border) !important; 
      border-radius: 8px !important; 
      background: var(--card-bg) !important; 
      margin-top: 12px !important;
    }
    [data-testid="stExpander"] summary {
      font-weight: 500 !important; 
      font-size: 0.9rem !important;
      color: var(--text-muted) !important;
    }
    
    .citation-badge {
      display: inline-block;
      background: var(--card-alt);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 16px;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-main);
      margin: 4px 8px 8px 0;
    }
    .citation-text {
      font-size: 0.9rem;
      color: var(--text-muted);
      font-style: italic;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }
    .citation-text:last-child { border-bottom: none; }

    /* ---------- File Uploader ---------- */
    [data-testid="stFileUploader"] { 
      background: var(--bg) !important; 
      border: 1px dashed var(--border) !important; 
      border-radius: 8px !important; 
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
# Sidebar - Document Upload
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">📁 Workspace</div>', unsafe_allow_html=True)

if "custom_db" not in st.session_state:
    st.session_state.custom_db = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Processing document..."):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            page_documents = []

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    page_documents.append(Document(
                        page_content=text,
                        metadata={"page_num": page_num + 1, "doc_type": "Document"}
                    ))

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            chunks = text_splitter.split_documents(page_documents)
            st.session_state.custom_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.file_name = uploaded_file.name
        st.sidebar.success("✓ Document ready for analysis")
else:
    st.session_state.custom_db = None
    st.session_state.file_name = None

# Plain English Privacy Note
st.sidebar.markdown(
    """
    <div class="privacy-note">
      <strong>Privacy Secure</strong><br>
      Your document is processed in memory for this session only. It is not saved to any database and disappears as soon as you refresh or close this tab.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main Area - Header & Instructions
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">FinRAG Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent document analysis with verifiable citations.</div>', unsafe_allow_html=True)

# Beautiful Instructions Card
st.markdown(
    """
    <div class="instructions-card">
      <h3>📖 How to use FinRAG</h3>
      <ul class="instructions-list">
        <li><strong>Step 1: Upload a PDF.</strong> Drag and drop any document (annual report, research paper, etc.) into the sidebar on the left.</li>
        <li><strong>Step 2: Let it process.</strong> Wait a few seconds for the AI to read and index the pages.</li>
        <li><strong>Step 3: Ask questions.</strong> Use the chat bar below to ask specific questions about the document.</li>
        <li><strong>Step 4: Verify the facts.</strong> Click the <i>"View Verified Sources"</i> dropdown under the AI's answer to see the exact page numbers and text excerpts it used to generate the response.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop if no file is attached
if st.session_state.custom_db is None:
    st.info("👈 Please upload a PDF in the sidebar to get started.")
    st.stop()

# Active Document Indicator
st.markdown(
    f"""
    <div style="background: var(--card-bg); border: 1px solid var(--border); padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; display: inline-block;">
      <span style="color: var(--text-muted); font-size: 0.9rem;">Currently analyzing:</span> 
      <strong style="color: var(--accent); margin-left: 6px;">{st.session_state.file_name}</strong>
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
    with st.expander("📚 View Verified Sources"):
        badges_html = "<div>"
        for src in sources:
            page_num = src.get('page_num', 'N/A')
            badges_html += f"<span class='citation-badge'>Page {page_num}</span>"
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)
        
        for src in sources:
            excerpt = src.get('excerpt', '')
            st.markdown(f"<div class='citation-text'>\"{excerpt}...\"</div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        avatar_class = "ai" if msg["role"] == "assistant" else "user"
        label = "FinRAG" if msg["role"] == "assistant" else "You"
        st.markdown(f"<div class='msg-avatar {avatar_class}'>{label}</div>", unsafe_allow_html=True)
        
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_prompt := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown("<div class='msg-avatar user'>You</div>", unsafe_allow_html=True)
        st.markdown(user_prompt)

    with st.spinner("Searching document and generating answer..."):
        results = active_vectorstore.similarity_search(query=user_prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in results])

        system_prompt = f"""You are a helpful research assistant.
Answer ONLY using the provided context. If the context does not contain the answer, explicitly state:
"I don't know based on the provided document." Do not hallucinate or use outside knowledge.

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
        st.markdown("<div class='msg-avatar ai'>FinRAG</div>", unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

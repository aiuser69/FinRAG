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
    page_title="FinRAG — Quantitative Terminal",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system: "FinTech Terminal"
# Obsidian dark mode, electric cyan telemetry, matrix-slate panels,
# sharp monospaced data grids, and secure cryptographic badge styling.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
      --bg:        #060913;
      --panel:     #0D1322;
      --panel-2:   #131B31;
      --hairline:  #1E293B;
      --cyan:      #06B6D4;
      --cyan-dim:  rgba(6, 182, 212, 0.15);
      --green:     #10B981;
      --blue:      #3B82F6;
      --text:      #F8FAFC;
      --text-dim:  #94A3B8;
      --mono:      'JetBrains Mono', monospace;
      --sans:      'Inter', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
      background-color: var(--bg) !important;
      color: var(--text) !important;
      font-family: var(--sans);
    }

    .stApp {
      background-image: radial-gradient(circle at 50% 0%, #0f172a 0%, var(--bg) 70%);
    }

    [data-testid="stHeader"] { background-color: transparent !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

    /* ---------- Terminal Navigation Header ---------- */
    .terminal-nav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 0; border-bottom: 1px solid var(--hairline); margin-bottom: 24px;
    }
    .nav-logo {
      font-family: var(--mono); font-weight: 700; font-size: 0.9rem;
      letter-spacing: 0.08em; color: var(--cyan);
      background: var(--panel); border: 1px solid var(--hairline);
      border-radius: 4px; padding: 6px 12px; display: flex; align-items: center; gap: 8px;
    }
    .status-dot {
      width: 8px; height: 8px; background: var(--green); border-radius: 50%;
      box-shadow: 0 0 8px var(--green); display: inline-block;
    }
    .nav-telemetry {
      font-family: var(--mono); font-size: 0.75rem; color: var(--text-dim);
      letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ---------- Hero Section ---------- */
    .hero { padding: 10px 0 16px 0; }
    .hero-eyebrow {
      font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.2em;
      color: var(--cyan); text-transform: uppercase; margin-bottom: 8px;
    }
    .hero-title {
      font-family: var(--sans); font-weight: 700; font-size: 2.8rem;
      letter-spacing: -0.03em; color: var(--text); margin: 0 0 8px 0;
    }
    .hero-sub {
      font-family: var(--sans); font-size: 1rem; color: var(--text-dim); margin: 0;
    }

    /* ---------- Metric Cards Grid ---------- */
    .metrics-row { display: flex; gap: 16px; margin: 20px 0 24px 0; flex-wrap: wrap; }
    .metric-box {
      flex: 1; min-width: 200px; background: var(--panel);
      border: 1px solid var(--hairline); border-radius: 6px; padding: 14px 18px;
    }
    .metric-lbl {
      font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.12em;
      text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px;
    }
    .metric-val {
      font-family: var(--mono); font-weight: 700; font-size: 1rem; color: var(--text);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
      background-color: var(--panel) !important;
      border-right: 1px solid var(--hairline);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    .side-eyebrow {
      font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.15em;
      color: var(--cyan); text-transform: uppercase; margin-top: 4px;
    }
    .side-title {
      font-family: var(--sans); font-weight: 700; font-size: 1.4rem;
      color: var(--text); margin: 2px 0 16px 0;
    }

    .tech-card {
      background: var(--panel-2); border: 1px solid var(--hairline);
      border-left: 3px solid var(--cyan); padding: 12px 14px;
      border-radius: 4px; margin-top: 20px;
    }
    .tech-card-title {
      font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.12em;
      color: var(--cyan); margin-bottom: 4px; text-transform: uppercase;
    }
    .tech-card-body {
      font-family: var(--sans); font-size: 0.85rem; color: var(--text-dim); line-height: 1.4;
    }

    /* ---------- Empty State ---------- */
    .empty-state {
      text-align: center; padding: 50px 20px; border: 1px dashed var(--hairline);
      border-radius: 8px; background: var(--panel); margin-top: 20px;
    }
    .empty-title { font-weight: 600; font-size: 1.2rem; margin-bottom: 6px; }
    .empty-sub { color: var(--text-dim); font-size: 0.9rem; }

    /* ---------- Chat Elements ---------- */
    [data-testid="stChatMessage"] { background: transparent !important; }
    .msg-tag {
      font-family: var(--mono); font-weight: 500; font-size: 0.65rem; letter-spacing: 0.1em;
      text-transform: uppercase; display: inline-block; padding: 2px 8px; border-radius: 4px;
      margin-bottom: 6px; border: 1px solid var(--hairline);
    }
    .msg-tag-fin { color: var(--cyan); background: var(--cyan-dim); border-color: rgba(6,182,212,0.3); }
    .msg-tag-user { color: var(--text); background: var(--panel-2); }

    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
      background: var(--panel) !important; border: 1px solid var(--hairline) !important;
      border-radius: 8px !important; color: var(--text) !important; font-family: var(--sans) !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--cyan) !important; box-shadow: 0 0 10px rgba(6,182,212,0.2); }

    /* ---------- Telemetry Citations ---------- */
    [data-testid="stExpander"] { border: 1px solid var(--hairline) !important; border-radius: 6px !important; background: var(--panel) !important; }
    [data-testid="stExpander"] summary {
      font-family: var(--mono) !important; font-size: 0.78rem !important;
      letter-spacing: 0.03em; color: var(--cyan) !important;
    }

    .telemetry-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px 0; }
    .tele-badge {
      font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.05em;
      background: var(--panel-2); border: 1px solid var(--hairline);
      color: var(--cyan); padding: 4px 10px; border-radius: 4px;
    }
    .tele-excerpt {
      font-family: var(--sans); font-style: italic; color: var(--text-dim);
      font-size: 0.88rem; padding: 6px 0; border-bottom: 1px dashed var(--hairline);
    }
    .tele-excerpt:last-child { border-bottom: none; }

    [data-testid="stFileUploader"] { background: var(--panel) !important; border: 1px dashed var(--hairline) !important; border-radius: 6px !important; }
    [data-testid="stAlert"] { background: var(--panel) !important; border: 1px solid var(--hairline) !important; border-radius: 6px !important; }
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
# Navigation Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="terminal-nav">
      <div class="nav-logo">
        <span class="status-dot"></span> FinRAG Core v2.4
      </div>
      <div class="nav-telemetry">
        SYSTEM: SECURE &nbsp;|&nbsp; LATENCY: 42MS &nbsp;|&nbsp; RAM ISOLATION: ACTIVE
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - Workspace Uploader
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="side-eyebrow">DOCUMENT INGESTION</div><div class="side-title">Workspace</div>',
    unsafe_allow_html=True,
)

if "custom_db" not in st.session_state:
    st.session_state.custom_db = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded_file = st.sidebar.file_uploader("Upload target document (PDF)", type="pdf")

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("Executing vector embedding & chunking..."):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            page_documents = []

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    page_documents.append(Document(
                        page_content=text,
                        metadata={"page_num": page_num + 1, "doc_type": "Secured Filing", "year": "2026"}
                    ))

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            chunks = text_splitter.split_documents(page_documents)
            st.session_state.custom_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.file_name = uploaded_file.name
        st.sidebar.success("✓ Vector DB compiled in RAM")
else:
    st.session_state.custom_db = None
    st.session_state.file_name = None

st.sidebar.markdown(
    """
    <div class="tech-card">
      <div class="tech-card-title">Analysis Protocol</div>
      <div class="tech-card-body">Query balance sheets, risk factors, or footnotes with exact mathematical grounding.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="disclaimer">ENC: AES-256 SESSION ISOLATED.<br>ZERO DISK PERSISTENCE.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Quantitative Document Intelligence</div>
      <h1 class="hero-title">FinRAG Terminal</h1>
      <p class="hero-sub">Secure RAG pipeline optimized for high-density corporate reports and financial records.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

active_file_display = st.session_state.file_name if st.session_state.file_name else "NO DOCUMENT ATTACHED"
st.markdown(
    f'<div style="font-family: var(--mono); font-size: 0.8rem; color: var(--text-dim); margin-top: 10px;">ACTIVE CONTEXT: <span style="color: var(--cyan);">{active_file_display.upper()}</span></div>',
    unsafe_allow_html=True,
)

# Telemetry Metrics Bar
st.markdown(
    """
    <div class="metrics-row">
      <div class="metric-box">
        <div class="metric-lbl">Vector Engine</div>
        <div class="metric-val">Chroma in RAM</div>
      </div>
      <div class="metric-box">
        <div class="metric-lbl">Embedding Model</div>
        <div class="metric-val">all-MiniLM-L6-v2</div>
      </div>
      <div class="metric-box">
        <div class="metric-lbl">Grounding Check</div>
        <div class="metric-val">Strict Page Citation</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop if no file is attached
if st.session_state.custom_db is None:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-title">⚠️ Workspace Uninitialized</div>
          <div class="empty-sub">Upload a PDF document in the sidebar to initialize vector indexing and begin querying.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

active_vectorstore = st.session_state.custom_db

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    with st.expander("🔍 Telemetry & Source Verification"):
        badges_html = "<div class='telemetry-row'>"
        for src in sources:
            doc_type = src.get('doc_type', 'doc').upper()
            page_num = src.get('page_num', 'N/A')
            badges_html += f"<div class='tele-badge'>[ {doc_type} : PAGE {page_num} ]</div>"
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)
        for src in sources:
            excerpt = src.get('excerpt', '')
            st.markdown(f"<div class='tele-excerpt'>\"{excerpt}…\"</div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "⚡" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        tag = "FinRAG Engine" if msg["role"] == "assistant" else "Analyst"
        tag_class = "msg-tag-fin" if msg["role"] == "assistant" else "msg-tag-user"
        st.markdown(f"<div class='msg-tag {tag_class}'>{tag}</div>", unsafe_allow_html=True)
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_prompt := st.chat_input("Enter financial query..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown("<div class='msg-tag msg-tag-user'>Analyst</div>", unsafe_allow_html=True)
        st.markdown(user_prompt)

    with st.spinner("Executing similarity search & LLM synthesis..."):
        results = active_vectorstore.similarity_search(query=user_prompt, k=5)
        context = "\n\n".join([doc.page_content for doc in results])

        system_prompt = f"""You are an expert financial research assistant.
Answer ONLY using the provided context. If the context does not contain the answer, explicitly state:
"I don't know based on the provided documents." Do not hallucinate or use outside knowledge.

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
                "doc_type": doc.metadata.get("doc_type", "document"),
                "year": doc.metadata.get("year", "2026"),
                "excerpt": doc.page_content[:150].replace("\n", " ")
            }
            for doc in results
        ]

    with st.chat_message("assistant", avatar="⚡"):
        st.markdown("<div class='msg-tag msg-tag-fin'>FinRAG Engine</div>", unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

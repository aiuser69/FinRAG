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
    page_title="FinRAG — NSE Filings Intelligence",
    page_icon="🪙",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system: "The Ledger"
# Ink-navy terminal + aged gold + graph-paper grid + rubber-stamped citations.
# Built around the one thing this product actually does: turn a 200-page
# filing into a cited answer, styled like the paper trail it comes from.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
      --bg:       #0B1220;
      --panel:    #121B2E;
      --panel-2:  #17233A;
      --hairline: #26314A;
      --gold:     #C9932F;
      --gold-hi:  #E7B655;
      --green:    #4C9A6A;
      --red:      #C1554B;
      --blue:     #5C87C9;
      --ink:      #EAE7DD;
      --ink-dim:  #9AA3B8;
      --ink-mono: #B9C2D6;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
      background-color: var(--bg) !important;
      color: var(--ink) !important;
      font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
      background-image:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 32px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 32px);
    }

    [data-testid="stHeader"] { background-color: transparent !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
      background-color: var(--panel) !important;
      border-right: 1px solid var(--hairline);
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }

    .side-eyebrow {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      letter-spacing: 0.16em;
      color: var(--gold);
      text-transform: uppercase;
      margin-top: 4px;
    }
    .side-title {
      font-family: 'Newsreader', serif;
      font-weight: 600;
      font-size: 1.5rem;
      color: var(--ink);
      margin: 2px 0 16px 0;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      background-color: var(--panel-2) !important;
      border-color: var(--hairline) !important;
      border-radius: 4px !important;
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.88rem !important;
    }

    .note-box {
      border-left: 3px solid var(--gold);
      background: var(--panel-2);
      padding: 12px 14px;
      border-radius: 4px;
      margin-top: 20px;
    }
    .note-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.64rem;
      letter-spacing: 0.14em;
      color: var(--gold-hi);
      margin-bottom: 4px;
    }
    .note-text {
      font-family: 'Newsreader', serif;
      font-style: italic;
      font-size: 0.92rem;
      color: var(--ink-dim);
      line-height: 1.4;
    }

    .disclaimer {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.66rem;
      color: var(--ink-dim);
      letter-spacing: 0.02em;
      border-top: 1px solid var(--hairline);
      margin-top: 28px;
      padding-top: 14px;
      line-height: 1.5;
    }

    /* ---------- Hero ---------- */
    .hero { padding: 12px 0 20px 0; border-bottom: 1px solid var(--hairline); }
    .eyebrow {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      color: var(--gold);
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .hero-title {
      font-family: 'Newsreader', serif;
      font-weight: 600;
      font-size: 3.2rem;
      line-height: 1;
      margin: 0 0 10px 0;
      color: var(--ink);
    }
    .hero-sub {
      font-family: 'Newsreader', serif;
      font-style: italic;
      font-size: 1.1rem;
      color: var(--ink-dim);
      margin: 0;
      max-width: 560px;
    }

    /* ---------- Ticker tape ---------- */
    .ticker-wrap {
      overflow: hidden;
      white-space: nowrap;
      background: var(--panel);
      border-bottom: 1px solid var(--hairline);
      padding: 9px 0;
      margin: 0 0 26px 0;
    }
    .ticker-track {
      display: inline-block;
      animation: tickerScroll 42s linear infinite;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      color: var(--ink-mono);
    }
    .tkr-code { color: var(--gold-hi); margin-right: 2px; }
    @keyframes tickerScroll {
      from { transform: translateX(0); }
      to   { transform: translateX(-50%); }
    }
    @media (prefers-reduced-motion: reduce) {
      .ticker-track { animation: none; }
    }

    /* ---------- Scope line ---------- */
    .scope-line {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.8rem;
      color: var(--ink-dim);
      letter-spacing: 0.03em;
      margin: 4px 0 18px 0;
    }
    .scope-value { color: var(--gold-hi); font-weight: 600; }

    /* ---------- Ledger metric cards ---------- */
    .ledger-row { display: flex; gap: 16px; margin: 4px 0 32px 0; flex-wrap: wrap; }
    .ledger-card {
      flex: 1; min-width: 190px;
      background: var(--panel);
      border: 1px solid var(--hairline);
      border-top: 3px solid var(--accent, var(--gold));
      border-radius: 6px;
      padding: 16px 18px;
      transition: transform 0.15s ease;
    }
    .ledger-card:hover { transform: translateY(-2px); }
    .ledger-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.66rem;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      color: var(--ink-dim);
      margin-bottom: 6px;
    }
    .ledger-value {
      font-family: 'Newsreader', serif;
      font-weight: 600;
      font-size: 1.2rem;
      color: var(--ink);
    }

    /* ---------- Empty state ---------- */
    .empty-state {
      text-align: center;
      padding: 60px 20px;
      border: 1px dashed var(--hairline);
      border-radius: 10px;
      background: var(--panel);
      margin-top: 20px;
    }
    .empty-icon { font-size: 2rem; margin-bottom: 10px; }
    .empty-title {
      font-family: 'Newsreader', serif;
      font-weight: 600;
      font-size: 1.3rem;
      color: var(--ink);
      margin-bottom: 6px;
    }
    .empty-sub {
      font-family: 'IBM Plex Sans', sans-serif;
      color: var(--ink-dim);
      font-size: 0.92rem;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
      background: transparent !important;
      animation: fadeInUp 0.3s ease both;
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
      [data-testid="stChatMessage"] { animation: none; }
    }

    .msg-tag {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.64rem;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      display: inline-block;
      padding: 2px 9px;
      border-radius: 3px;
      margin-bottom: 8px;
    }
    .msg-tag-fin { color: var(--bg); background: var(--gold-hi); }
    .msg-tag-user {
      color: var(--ink-mono);
      background: var(--panel-2);
      border: 1px solid var(--hairline);
    }

    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
      background: var(--panel) !important;
      border: 1px solid var(--hairline) !important;
      border-radius: 8px !important;
      color: var(--ink) !important;
      font-family: 'IBM Plex Sans', sans-serif !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--gold) !important; }

    /* ---------- Expander / stamped sources ---------- */
    [data-testid="stExpander"] {
      border: 1px solid var(--hairline) !important;
      border-radius: 6px !important;
      background: var(--panel) !important;
    }
    [data-testid="stExpander"] summary {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.8rem !important;
      letter-spacing: 0.03em;
      color: var(--gold-hi) !important;
    }

    .stamp-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 14px 0; }
    .stamp {
      border: 1.5px dashed var(--gold);
      border-radius: 3px;
      padding: 6px 10px;
      background: rgba(201, 147, 47, 0.06);
    }
    .stamp-head {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.66rem;
      letter-spacing: 0.03em;
      color: var(--gold-hi);
      white-space: nowrap;
    }
    .excerpt {
      font-family: 'Newsreader', serif;
      font-style: italic;
      color: var(--ink-dim);
      font-size: 0.92rem;
      padding: 5px 0 12px 0;
      border-bottom: 1px dashed var(--hairline);
    }
    .excerpt:last-child { border-bottom: none; }

    [data-testid="stFileUploader"] {
      background: var(--panel-2) !important;
      border: 1px dashed var(--hairline) !important;
      border-radius: 6px !important;
    }

    [data-testid="stAlert"] {
      background: var(--panel-2) !important;
      border: 1px solid var(--hairline) !important;
      border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Setup Groq Client securely
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# 2. Cache resources
@st.cache_resource
def load_base_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

base_vectorstore = load_base_vectorstore()
embeddings = load_embeddings()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="side-eyebrow">TERMINAL</div><div class="side-title">Coverage</div>',
    unsafe_allow_html=True,
)

companies = [
    "Adani", "Bajaj", "Bharti Airtel", "HDFC Bank",
    "Hindustan Unilever", "ICICI Bank", "Infosys",
    "Larsen & Toubro", "Mahindra", "Maruti Suzuki",
    "Reliance", "SBI", "Sun Pharma", "TCS", "Titan",
    "--- Upload Custom PDF ---"
]
company = st.sidebar.selectbox("Select Target Filing", companies)

# 3. Handle Custom PDF Uploads
if "custom_db" not in st.session_state:
    st.session_state.custom_db = None

if company == "--- Upload Custom PDF ---":
    uploaded_file = st.sidebar.file_uploader("Upload private research PDF", type="pdf")

    if uploaded_file is not None and st.session_state.custom_db is None:
        with st.spinner("Processing document vectors..."):
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            page_documents = []

            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text")
                if text.strip():
                    page_documents.append(Document(
                        page_content=text,
                        metadata={"page_num": page_num + 1, "doc_type": "Custom Upload", "year": "N/A"}
                    ))

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            chunks = text_splitter.split_documents(page_documents)
            st.session_state.custom_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

    if st.session_state.custom_db is not None:
        st.sidebar.success("🔒 Secure in-memory database active")

    custom_vectorstore = st.session_state.custom_db
else:
    custom_vectorstore = None
    st.session_state.custom_db = None

st.sidebar.markdown(
    """
    <div class="note-box">
      <div class="note-label">NOTE</div>
      <div class="note-text">Ask about margins, regulatory headwinds, or capital allocation strategy.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="disclaimer">FOR RESEARCH &amp; EDUCATIONAL USE ONLY.<br>NOT INVESTMENT ADVICE.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NSE Annual Reports · Earnings Calls · Cited Answers</div>
      <h1 class="hero-title">FinRAG</h1>
      <p class="hero-sub">Ask a filing a question. Get an answer with the page number.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ticker tape of covered companies
tickers = {
    "Adani": "ADANIENT", "Bajaj": "BAJFINANCE", "Bharti Airtel": "BHARTIARTL",
    "HDFC Bank": "HDFCBANK", "Hindustan Unilever": "HINDUNILVR", "ICICI Bank": "ICICIBANK",
    "Infosys": "INFY", "Larsen & Toubro": "LT", "Mahindra": "M&M",
    "Maruti Suzuki": "MARUTI", "Reliance": "RELIANCE", "SBI": "SBIN",
    "Sun Pharma": "SUNPHARMA", "TCS": "TCS", "Titan": "TITAN",
}
ticker_items = "&nbsp;&nbsp;•&nbsp;&nbsp;".join(
    f"{name} <span class='tkr-code'>{code}</span>" for name, code in tickers.items()
)
st.markdown(
    f"""
    <div class="ticker-wrap">
      <div class="ticker-track">{ticker_items}&nbsp;&nbsp;•&nbsp;&nbsp;{ticker_items}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="scope-line">ACTIVE SCOPE → <span class="scope-value">{company.upper()}</span></div>',
    unsafe_allow_html=True,
)

# Ledger metric cards
num_companies = len(companies) - 1
st.markdown(
    f"""
    <div class="ledger-row">
      <div class="ledger-card" style="--accent: var(--gold);">
        <div class="ledger-label">Coverage</div>
        <div class="ledger-value">{num_companies} Nifty 50 Cos.</div>
      </div>
      <div class="ledger-card" style="--accent: var(--green);">
        <div class="ledger-label">Source Depth</div>
        <div class="ledger-value">Reports + Concalls</div>
      </div>
      <div class="ledger-card" style="--accent: var(--blue);">
        <div class="ledger-label">Grounding</div>
        <div class="ledger-value">Every Answer Cited</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop if custom upload is missing
if company == "--- Upload Custom PDF ---" and custom_vectorstore is None:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-icon">📎</div>
          <div class="empty-title">Upload a filing to begin</div>
          <div class="empty-sub">Drop a PDF in the sidebar — annual report, prospectus, or concall transcript — and ask it anything.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

active_vectorstore = custom_vectorstore if company == "--- Upload Custom PDF ---" else base_vectorstore

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def render_sources(sources):
    if not sources:
        return
    with st.expander("📎 Verified sources"):
        stamps_html = "<div class='stamp-row'>"
        for i, src in enumerate(sources):
            rotate = "-1.1deg" if i % 2 == 0 else "1.1deg"
            stamps_html += f"""
            <div class="stamp" style="transform:rotate({rotate});">
              <div class="stamp-head">✓ VERIFIED · {src['doc_type'].upper()} · {src['year']} · PG {src['page_num']}</div>
            </div>
            """
        stamps_html += "</div>"
        st.markdown(stamps_html, unsafe_allow_html=True)
        for src in sources:
            st.markdown(f"<div class='excerpt'>&ldquo;{src['excerpt']}…&rdquo;</div>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🪙" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        tag = "FinRAG · Grounded Answer" if msg["role"] == "assistant" else "You"
        tag_class = "msg-tag-fin" if msg["role"] == "assistant" else "msg-tag-user"
        st.markdown(f"<div class='msg-tag {tag_class}'>{tag}</div>", unsafe_allow_html=True)
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown("<div class='msg-tag msg-tag-user'>You</div>", unsafe_allow_html=True)
        st.markdown(user_prompt)

    with st.spinner("Synthesizing financial records..."):
        if company == "--- Upload Custom PDF ---":
            results = active_vectorstore.similarity_search(query=user_prompt, k=5)
        else:
            results = active_vectorstore.similarity_search(query=user_prompt, k=5, filter={"company": company})

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
                "doc_type": doc.metadata.get("doc_type", "report"),
                "year": doc.metadata.get("year", "2026"),
                "excerpt": doc.page_content[:150].replace("\n", " ")
            }
            for doc in results
        ]

    with st.chat_message("assistant", avatar="🪙"):
        st.markdown("<div class='msg-tag msg-tag-fin'>FinRAG · Grounded Answer</div>", unsafe_allow_html=True)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
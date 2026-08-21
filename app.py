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
# Design system: "Mela Ledger"
# Forest green poster background, marigold-gold display type with a hard
# drop shadow, a rotated pink sticker, and a candy-striped textile border
# on the CTA — festival-poster energy applied to a filings terminal.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,700;0,6..96,800;0,6..96,900&family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
      --bg:        #123D2A;
      --bg-deep:   #0D2E1F;
      --panel:     #0F3527;
      --panel-2:   #143F2C;
      --hairline:  rgba(255,255,255,0.14);
      --gold:      #F4C430;
      --gold-hi:   #FFDD6B;
      --pink:      #FF2E88;
      --pink-deep: #D6006A;
      --cream:     #FBF3DD;
      --ink:       #0B2A1C;
      --stripe-red:#C1272D;
      --font-display: 'Bodoni Moda', serif;
      --font-mono:    'Space Mono', monospace;
      --font-body:    'IBM Plex Sans', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
      background-color: var(--bg) !important;
      color: var(--cream) !important;
      font-family: var(--font-body);
    }

    .stApp {
      background-image: radial-gradient(ellipse at center, rgba(0,0,0,0) 40%, rgba(0,0,0,0.35) 100%);
    }

    [data-testid="stHeader"] { background-color: transparent !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--hairline); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--gold); }

    /* ---------- Top nav ---------- */
    .nav-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 0 4px 0;
    }
    .nav-logo {
      font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem;
      letter-spacing: 0.04em; color: var(--gold-hi);
      background: var(--bg-deep); border: 2px solid var(--gold);
      border-radius: 6px; padding: 6px 12px; transform: rotate(-2deg);
      display: inline-block;
    }
    .nav-links {
      font-family: var(--font-mono); font-size: 0.8rem; letter-spacing: 0.08em;
      color: var(--cream); text-transform: uppercase; opacity: 0.85;
    }
    .nav-cta {
      font-family: var(--font-mono); font-weight: 700; font-size: 0.82rem;
      letter-spacing: 0.06em; text-transform: uppercase;
      color: var(--ink) !important; background: var(--gold);
      padding: 10px 22px 8px 22px; border-radius: 4px; text-decoration: none;
      border: 4px solid var(--gold);
      border-image: repeating-linear-gradient(45deg, var(--stripe-red) 0 4px, var(--cream) 4px 8px) 4;
      box-shadow: 4px 4px 0 var(--ink);
    }

    /* ---------- Hero ---------- */
    .hero { padding: 34px 0 6px 0; text-align: left; }
    .eyebrow {
      font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.2em;
      color: var(--gold-hi); text-transform: uppercase; margin-bottom: 14px;
    }
    .hero-title-row { display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin-bottom: 6px; }
    .hero-title {
      font-family: var(--font-display); font-weight: 900; font-style: italic;
      font-size: 6.2rem; line-height: 0.95; color: var(--gold);
      filter: drop-shadow(6px 8px 0px rgba(0,0,0,0.35));
      margin: 0;
    }
    .sticker {
      font-family: var(--font-mono); font-weight: 700; font-size: 1.05rem;
      letter-spacing: 0.02em; color: var(--cream); background: var(--pink);
      border: 3px solid var(--cream); border-radius: 46% 54% 51% 49% / 55% 45% 55% 45%;
      padding: 10px 18px; transform: rotate(-9deg); display: inline-block;
      margin: 0 -14px; box-shadow: 4px 5px 0 rgba(0,0,0,0.35);
      position: relative; top: -18px;
    }
    .hero-meta {
      font-family: var(--font-mono); font-size: 0.85rem; letter-spacing: 0.04em;
      color: var(--cream); display: flex; justify-content: space-between;
      flex-wrap: wrap; gap: 8px; border-top: 1px solid var(--hairline);
      border-bottom: 1px solid var(--hairline); padding: 14px 0; margin-top: 20px;
      text-transform: uppercase; opacity: 0.9;
    }
    .hero-meta .credit { color: var(--gold-hi); }

    /* ---------- Sunburst ---------- */
    .burst-wrap { display: flex; justify-content: center; margin: 18px 0 6px 0; opacity: 0.9; }

    /* ---------- Ticker tape ---------- */
    .ticker-wrap {
      overflow: hidden; white-space: nowrap;
      background: var(--gold); border-top: 3px solid var(--ink); border-bottom: 3px solid var(--ink);
      padding: 9px 0; margin: 26px 0 30px 0; transform: rotate(-0.4deg);
    }
    .ticker-track {
      display: inline-block; animation: tickerScroll 42s linear infinite;
      font-family: var(--font-mono); font-weight: 700; font-size: 0.8rem;
      letter-spacing: 0.05em; color: var(--ink); text-transform: uppercase;
    }
    .tkr-code { color: var(--pink-deep); margin-right: 2px; }
    @keyframes tickerScroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    @media (prefers-reduced-motion: reduce) { .ticker-track { animation: none; } }

    /* ---------- Scope line ---------- */
    .scope-line {
      font-family: var(--font-mono); font-size: 0.82rem; color: var(--cream);
      letter-spacing: 0.03em; margin: 4px 0 18px 0; opacity: 0.85;
    }
    .scope-value { color: var(--pink); font-weight: 700; }

    /* ---------- Sticker metric cards ---------- */
    .ledger-row { display: flex; gap: 20px; margin: 10px 0 34px 0; flex-wrap: wrap; }
    .ledger-card {
      flex: 1; min-width: 190px; background: var(--accent, var(--cream));
      color: var(--ink); border: 3px solid var(--ink); border-radius: 8px;
      padding: 16px 18px; box-shadow: 5px 5px 0 rgba(0,0,0,0.4);
      transition: transform 0.15s ease;
    }
    .ledger-card:nth-child(1) { transform: rotate(-1.6deg); }
    .ledger-card:nth-child(2) { transform: rotate(1deg); }
    .ledger-card:nth-child(3) { transform: rotate(-0.8deg); }
    .ledger-card:hover { transform: translateY(-3px) rotate(0deg); }
    .ledger-label {
      font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.13em;
      text-transform: uppercase; opacity: 0.75; margin-bottom: 6px;
    }
    .ledger-value { font-family: var(--font-display); font-weight: 800; font-size: 1.35rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { background-color: var(--bg-deep) !important; border-right: 3px solid var(--gold); }
    [data-testid="stSidebar"] * { color: var(--cream) !important; }
    .side-eyebrow {
      font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.18em;
      color: var(--gold-hi); text-transform: uppercase; margin-top: 4px;
    }
    .side-title { font-family: var(--font-display); font-weight: 800; font-style: italic; font-size: 1.7rem; color: var(--gold); margin: 2px 0 16px 0; }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      background-color: var(--panel) !important; border: 2px solid var(--gold) !important;
      border-radius: 6px !important; font-family: var(--font-mono) !important; font-size: 0.88rem !important;
    }

    .note-box {
      border: 2px dashed var(--pink); background: var(--panel);
      padding: 12px 14px; border-radius: 6px; margin-top: 20px;
    }
    .note-label { font-family: var(--font-mono); font-size: 0.64rem; letter-spacing: 0.14em; color: var(--pink); margin-bottom: 4px; text-transform: uppercase; }
    .note-text { font-family: var(--font-body); font-size: 0.9rem; color: var(--cream); opacity: 0.85; line-height: 1.4; }

    .disclaimer {
      font-family: var(--font-mono); font-size: 0.64rem; color: var(--cream); opacity: 0.6;
      letter-spacing: 0.02em; border-top: 1px solid var(--hairline); margin-top: 28px; padding-top: 14px; line-height: 1.5;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
      text-align: center; padding: 60px 20px; border: 3px dashed var(--gold);
      border-radius: 12px; background: var(--panel); margin-top: 20px;
    }
    .empty-icon { font-size: 2rem; margin-bottom: 10px; }
    .empty-title { font-family: var(--font-display); font-weight: 800; font-style: italic; font-size: 1.5rem; color: var(--gold); margin-bottom: 6px; }
    .empty-sub { font-family: var(--font-body); color: var(--cream); opacity: 0.8; font-size: 0.92rem; }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] { background: transparent !important; animation: fadeInUp 0.3s ease both; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) { [data-testid="stChatMessage"] { animation: none; } }

    .msg-tag {
      font-family: var(--font-mono); font-weight: 700; font-size: 0.64rem; letter-spacing: 0.1em;
      text-transform: uppercase; display: inline-block; padding: 3px 11px; border-radius: 20px;
      margin-bottom: 8px; border: 2px solid var(--ink);
    }
    .msg-tag-fin { color: var(--ink); background: var(--gold); }
    .msg-tag-user { color: var(--cream); background: var(--pink-deep); border-color: var(--cream); }

    [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] {
      background: var(--panel) !important; border: 3px solid var(--gold) !important;
      border-radius: 10px !important; color: var(--cream) !important; font-family: var(--font-body) !important;
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--pink) !important; }

    /* ---------- Expander / sticker citations ---------- */
    [data-testid="stExpander"] { border: 2px solid var(--gold) !important; border-radius: 8px !important; background: var(--panel) !important; }
    [data-testid="stExpander"] summary {
      font-family: var(--font-mono) !important; font-weight: 700 !important; font-size: 0.8rem !important;
      letter-spacing: 0.03em; color: var(--gold-hi) !important;
    }

    .stamp-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 14px 0; }
    .stamp {
      border: 2px solid var(--cream); border-radius: 40% 60% 55% 45% / 55% 45% 55% 45%;
      padding: 7px 13px; background: var(--pink); box-shadow: 3px 4px 0 rgba(0,0,0,0.35);
    }
    .stamp:nth-child(odd) { transform: rotate(-3deg); }
    .stamp:nth-child(even) { transform: rotate(2.5deg); }
    .stamp-head { font-family: var(--font-mono); font-weight: 700; font-size: 0.66rem; letter-spacing: 0.03em; color: var(--cream); white-space: nowrap; }
    .excerpt {
      font-family: var(--font-body); font-style: italic; color: var(--cream); opacity: 0.8;
      font-size: 0.92rem; padding: 5px 0 12px 0; border-bottom: 1px dashed var(--hairline);
    }
    .excerpt:last-child { border-bottom: none; }

    [data-testid="stFileUploader"] { background: var(--panel) !important; border: 2px dashed var(--gold) !important; border-radius: 8px !important; }
    [data-testid="stAlert"] { background: var(--panel) !important; border: 2px solid var(--gold) !important; border-radius: 8px !important; }
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
# Top nav
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="nav-row">
      <div class="nav-logo">🪙 FINRAG</div>
      <div class="nav-links">GROUNDED · CITED · NIFTY 50</div>
      <a class="nav-cta" href="#chat-anchor">OPEN TERMINAL ↓</a>
    </div>
    """,
    unsafe_allow_html=True,
)

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
      <div class="eyebrow">NSE Annual Reports · Earnings Calls</div>
      <div class="hero-title-row">
        <span class="hero-title">FIN</span>
        <span class="sticker">CITED</span>
        <span class="hero-title">RAG</span>
      </div>
      <div class="hero-meta">
        <span>NIFTY 50 FILINGS &nbsp;·&nbsp; GROUNDED ANSWERS ONLY</span>
        <span class="credit">BUILT BY MOHOK</span>
      </div>
    </div>
    <div class="burst-wrap">
      <svg width="150" height="70" viewBox="0 0 150 70" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g stroke="#F4C430" stroke-width="3" stroke-linecap="round">
          <line x1="75" y1="70" x2="75" y2="20" />
          <line x1="75" y1="70" x2="45" y2="30" />
          <line x1="75" y1="70" x2="105" y2="30" />
          <line x1="75" y1="70" x2="20" y2="55" />
          <line x1="75" y1="70" x2="130" y2="55" />
        </g>
      </svg>
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

# Sticker metric cards
num_companies = len(companies) - 1
st.markdown(
    f"""
    <div class="ledger-row">
      <div class="ledger-card" style="--accent: var(--gold);">
        <div class="ledger-label">Coverage</div>
        <div class="ledger-value">{num_companies} Nifty 50 Cos.</div>
      </div>
      <div class="ledger-card" style="--accent: var(--cream);">
        <div class="ledger-label">Source Depth</div>
        <div class="ledger-value">Reports + Concalls</div>
      </div>
      <div class="ledger-card" style="--accent: var(--pink); color: var(--cream);">
        <div class="ledger-label">Grounding</div>
        <div class="ledger-value">Every Answer Cited</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Anchor target for the "OPEN TERMINAL" nav button
st.markdown('<div id="chat-anchor"></div>', unsafe_allow_html=True)

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
        for src in sources:
            stamps_html += f"""
            <div class="stamp">
              <div class="stamp-head">✓ {src['doc_type'].upper()} · {src['year']} · PG {src['page_num']}</div>
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
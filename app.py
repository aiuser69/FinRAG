import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

st.set_page_config(page_title="FinRAG MVP", page_icon="📈", layout="wide")

# 1. Setup Groq Client
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# 2. Cache the static database and embedding model
@st.cache_resource
def load_base_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

base_vectorstore = load_base_vectorstore()
embeddings = load_embeddings()

# --- Sidebar ---
st.sidebar.title("🏢 Company Scope")

companies = [
    "Adani", "Bajaj", "Bharti Airtel", "HDFC Bank", 
    "Hindustan Unilever", "ICICI Bank", "Infosys", 
    "Larsen & Toubro", "Mahindra", "Maruti Suzuki", 
    "Reliance", "SBI", "Sun Pharma", "TCS", "Titan", 
    "--- Upload Custom PDF ---"
]
company = st.sidebar.selectbox("Select Nifty50 Company or Upload", companies)
st.sidebar.markdown("---")

# 3. Handle Custom PDF Uploads Safely in Memory
if "custom_db" not in st.session_state:
    st.session_state.custom_db = None

if company == "--- Upload Custom PDF ---":
    uploaded_file = st.sidebar.file_uploader("Upload your own PDF", type="pdf")
    
    # Only process if we haven't already processed it
    if uploaded_file is not None and st.session_state.custom_db is None:
        with st.spinner("Processing your PDF..."):
            # Use getvalue() so Streamlit doesn't lose the file on rerun
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
            
            # Save to session state so it survives chat reruns
            st.session_state.custom_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
    if st.session_state.custom_db is not None:
        st.sidebar.success("PDF Ready for Questions!")
        
    custom_vectorstore = st.session_state.custom_db
else:
    custom_vectorstore = None
    st.session_state.custom_db = None # Clear memory if they switch companies

st.sidebar.info("💡 **Tip:** Ask questions about management commentary, margins, key risks, or strategic initiatives.")

# --- Main UI ---
st.title("📈 FinRAG — Financial Research Assistant")
st.caption(f"Ask grounded questions across filings for **{company}**.")

# Stop the app if they selected custom upload but haven't uploaded a file yet
if company == "--- Upload Custom PDF ---" and custom_vectorstore is None:
    st.info("👈 Please upload a PDF in the sidebar to start asking questions.")
    st.stop()

# Switch between the static DB or the temporary upload DB
active_vectorstore = custom_vectorstore if company == "--- Upload Custom PDF ---" else base_vectorstore

# Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Document Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['doc_type'].upper()}** — *Page {src['page_num']}*")
                    st.caption(f"> \"{src['excerpt']}...\"")

# Chat Input & Response Generation
if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Searching document chunks..."):
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
                "doc_type": doc.metadata.get("doc_type"),
                "excerpt": doc.page_content[:150].replace("\n", " ")
            }
            for doc in results
        ]

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📚 View Document Sources"):
            for src in sources:
                st.markdown(f"- **{src['doc_type'].upper()}** — *Page {src['page_num']}*")
                st.caption(f"> \"{src['excerpt']}...\"")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
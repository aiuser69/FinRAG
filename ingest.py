import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def process_and_ingest_pdf(pdf_path, company_name, doc_type, year, persist_directory="data/chroma_db"):
    # 1. Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    page_documents = []

    print(f"Extracting text from {company_name} {doc_type} ({year})...")
    
    # 2. Iterate page-by-page to lock in exact metadata
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Skip empty pages
        if not text.strip():
            continue
            
        # Create a LangChain Document with explicit metadata
        metadata = {
            "company": company_name,
            "doc_type": doc_type,
            "year": str(year),
            "page_num": page_num + 1  # 1-indexed for human readability
        }
        
        page_documents.append(Document(page_content=text, metadata=metadata))

    # 3. Configure the Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(page_documents)
    print(f"Generated {len(chunks)} chunks from {len(page_documents)} pages.")

    # 4. Initialize Local Embeddings (free, offline model)
    print("Loading embedding model (this may take a moment on first run)...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Store in Local Persistent ChromaDB
    print("Saving chunks into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully saved chunks to ChromaDB at {persist_directory}!")
    return vectorstore

if __name__ == "__main__":
    # A list of all 15 files mapped to their company names
    new_filings = [
        {"path": "data/raw/Adani-Report-FY-2025-2026.pdf", "name": "Adani"},
        {"path": "data/raw/Airtel-Report-FY-2025-2026.pdf", "name": "Bharti Airtel"},
        {"path": "data/raw/Bajaj-FY-2025-2026.pdf", "name": "Bajaj"},
        {"path": "data/raw/HDFC-Report-FY-2025-2026.pdf", "name": "HDFC Bank"},
        {"path": "data/raw/Hindustan-Report-FY-2025-2026.pdf", "name": "Hindustan Unilever"},
        {"path": "data/raw/ICICI-Report-FY-2025-2026.pdf", "name": "ICICI Bank"},
        {"path": "data/raw/Infosys-Report-FY-2025-2026.pdf", "name": "Infosys"},
        {"path": "data/raw/Larsen-Report-FY-2025-2026.pdf", "name": "Larsen & Toubro"},
        {"path": "data/raw/Mahindra Annual Report 2025-26.pdf", "name": "Mahindra"},
        {"path": "data/raw/Maruti-Report-FY-2025-22026.pdf", "name": "Maruti Suzuki"},
        {"path": "data/raw/Reliance-Report-FY-2025-2026.pdf", "name": "Reliance"},
        {"path": "data/raw/SBI-Report-FY-2025-2026.pdf", "name": "SBI"},
        {"path": "data/raw/Sun-Report-FY-2025-26.pdf", "name": "Sun Pharma"},
        {"path": "data/raw/TCS-Report-FY-2025-2026.pdf", "name": "TCS"},
        {"path": "data/raw/Titan-Report-FY-2025-2026.pdf", "name": "Titan"}
    ]
    
    # Loop through and ingest each one automatically
    for filing in new_filings:
        print(f"\n--- Starting {filing['name']} ---")
        try:
            process_and_ingest_pdf(
                pdf_path=filing["path"],
                company_name=filing["name"],
                doc_type="annual_report",
                year="2026"
            )
        except Exception as e:
            print(f"Skipped {filing['name']} because of an error: {e}")
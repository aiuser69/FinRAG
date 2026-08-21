import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# 1. Setup Groq API 
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"  # Replace with your actual Groq API key
client = Groq()

def test_query(question, company_name):
    # 2. Load our saved database
    print("Loading database...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

    # 3. Search for the top 5 chunks matching the question
    print(f"Searching database for answers about {company_name}...")
    results = vectorstore.similarity_search(
        query=question,
        k=5,
        filter={"company": company_name} # Only look at TCS documents
    )

    # 4. Assemble the prompt package
    context = "\n\n".join([doc.page_content for doc in results])
    
    system_prompt = f"""You are a financial research assistant. Answer ONLY using the provided context. 
    If the context doesn't contain the answer, say "I don't know based on the provided documents."
    Do not guess or use outside knowledge.
    
    Context:
    {context}"""

    # 5. Ask Groq (Llama 3.1 70B)
    print("Asking Llama 3.1...")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        model="openai/gpt-oss-20b",
    )
    
    # 6. Print the Results
    print("\n" + "="*50)
    print("ANSWER:")
    print(chat_completion.choices[0].message.content)
    print("="*50)
    print("\nSOURCES USED:")
    for doc in results:
        page = doc.metadata.get('page_num')
        doc_type = doc.metadata.get('doc_type')
        year = doc.metadata.get('year')
        print(f"- Page {page} of {doc_type} ({year})")

if __name__ == "__main__":
    # Feel free to change this question to test it!
    test_query("What were the key highlights regarding revenue growth?", "TCS")
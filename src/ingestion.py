import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_docs():
    # 1. Load your Capgemini Strategy PDFs from the 'data' folder
    print("--- Phase 1: Loading Documents ---")
    loader = DirectoryLoader("./data", glob="./*.pdf", loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    
    # 2. Split documents into smaller chunks (so the AI can read them easily)
    print("--- Phase 2: Splitting Text ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    
    # 3. Create Embeddings using Gemini's model
    print("--- Phase 3: Generating Embeddings & Saving to ChromaDB ---")
    # Use the exact string from your terminal output
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Create and persist the Vector Database
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chromadb"
    )
    
    print(f"Success! Ingested {len(documents)} chunks into ./chromadb")

if __name__ == "__main__":
    ingest_docs()
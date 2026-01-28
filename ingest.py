import os
import time
from dotenv import load_dotenv

# Standard LangChain Imports (No custom wrappers needed)
from langchain_community.document_loaders import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_docs():
    print("🚀 Starting Ingestion Process...")

    # --- 1. Connect to Confluence ---
    # We configure the Space Key and Limit HERE to avoid the ValueError
    loader = ConfluenceLoader(
        url=os.getenv("CONFLUENCE_URL"),
        username=os.getenv("CONFLUENCE_USERNAME"),
        api_key=os.getenv("CONFLUENCE_API_TOKEN"),
        cloud=True,
        space_key="~626cd32efff19d00692997cb", # <--- Your specific Space Key
        limit=50
    )
    
    print("   - Fetching pages from Confluence...")
    documents = loader.load()
    print(f"   - ✅ Found {len(documents)} pages.")

    # --- NEW: Print the names of every page found ---
    print(f"   - ✅ Found {len(documents)} pages:")
    for i, doc in enumerate(documents, 1):
        print(f"       {i}. {doc.metadata.get('title', 'Untitled Page')}")

    if not documents:
        print("   - ⚠️ No documents found. Check permissions!")
        return

    # --- 2. Split Text ---
    print("   - Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"   - ✅ Created {len(splits)} chunks.")

    # --- 3. Initialize Embeddings ---
    # We use the model that works for you. 
    # Since we set Pinecone to 3072, this will now fit perfectly.
    print("   - Initializing Embedding Model (gemini-embedding-001)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # --- 4. Save to Pinecone ---
    print(f"   - Saving to Pinecone index: {os.getenv('PINECONE_INDEX_NAME')}...")
    
    # We add a small retry/batch logic automatically handled by LangChain
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("🎉 Success! Your bot has learned the documents.")

if __name__ == "__main__":
    ingest_docs()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single script to test performance of Chroma and Pinecone RAG setups on arxiv papers.
Requires the following packages:
  pip install langchain sentence-transformers transformers torch chromadb pypdf pinecone-client

NOTE: This script references ChatCerebras (for LLM calls) and Pinecone (a remote vector DB).
If you do not have valid credentials for either, you will need to adapt/remove those parts.

Usage:
1. Ensure you have downloaded papers to the arxiv_papers directory using download_arxiv_papers.py
2. Update any API keys (ChatCerebras, Pinecone) below with your own.
3. Run: python3 this_script.py
"""

import time
import os
import glob

# --------- Imports for Embeddings, Vector Stores, and LLM ---------
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# --------- Chroma (local vector store) imports ---------
import chromadb
from langchain_community.vectorstores import Chroma

# --------- Pinecone (remote vector store) imports ---------
import pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.vectorstores import Pinecone

# --------- Cerebras LLM (via API) ---------
# If you do not have a ChatCerebras account/key, remove or replace this with your own LLM
from langchain_cerebras import ChatCerebras

# ------------------ Global Configuration ------------------
EMBEDDING_MODELS = {
    "roberta-large": "sentence-transformers/all-roberta-large-v1",  # 1024 dimensions
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",            # 384 dimensions
}

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Path to directory containing arxiv papers
ARXIV_PAPERS_DIR = os.getenv("ARXIV_PAPERS_DIR", "arxiv_papers")

# Get API keys from environment variables
CHAT_CEREBRAS_API_KEY = os.getenv("CHAT_CEREBRAS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test-index")

# Longer test queries, as requested
TEST_QUERIES = [
    "What is the significance of Toroidal plasma equilibrium in the context of nuclear fusion, and how is it typically maintained in modern fusion reactors?",
    "Explain the principle behind Langmuir probes used in plasma diagnostics, including how they measure electron temperature and density.",
    "Please provide a brief historical overview of the development of Stellarators, highlighting the major challenges and technological breakthroughs.",
]


# ------------------ Utility Functions ------------------
def get_embedding_dimension(model_name: str) -> int:
    """
    Get the embedding dimension for a given SentenceTransformer model
    by creating a temporary embedding.
    """
    model = SentenceTransformer(model_name)
    test_embedding = model.encode("test", convert_to_tensor=False)
    return len(test_embedding)


def load_arxiv_papers():
    """
    Load all PDF documents from the arxiv_papers directory, returning one Document per PDF page.
    """
    if not os.path.exists(ARXIV_PAPERS_DIR):
        raise FileNotFoundError(f"Directory {ARXIV_PAPERS_DIR} not found. Please run download_arxiv_papers.py first.")

    documents = []
    pdf_files = glob.glob(os.path.join(ARXIV_PAPERS_DIR, "*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {ARXIV_PAPERS_DIR}")
        
    print(f"Loading {len(pdf_files)} papers from {ARXIV_PAPERS_DIR}...")

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            # This returns a list of Document objects, one for each page
            pdf_pages = loader.load()
            
            # Optionally add more metadata (e.g. page_number)
            for i, page_doc in enumerate(pdf_pages, start=1):
                page_doc.metadata["source"] = pdf_file
                page_doc.metadata["page_number"] = i
            
            documents.extend(pdf_pages)
            print(f"Loaded {len(pdf_pages)} pages from {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
            continue

    return documents


# ------------------ Chroma RAG Setup & Test ------------------
def setup_chroma_rag(documents, embedding_model: str, persist_directory: str = "./chroma_db"):
    """
    Sets up a RetrievalQA chain using Chroma as the vector store.
    Embedding is handled by a specified SentenceTransformer model.
    LLM is ChatCerebras (can be replaced with another model if needed).
    """

    # Validate the embedding model
    if embedding_model not in EMBEDDING_MODELS.values():
        raise ValueError(f"Embedding model must be one of: {list(EMBEDDING_MODELS.values())}")

    # Get actual embedding dimension for the model
    dimension = get_embedding_dimension(embedding_model)
    print(f"[Chroma] Using embedding model: {embedding_model}")
    print(f"[Chroma] Detected embedding dimension: {dimension}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you have a GPU
    )

    # Initialize LLM
    llm = ChatCerebras(
        api_key=CHAT_CEREBRAS_API_KEY,
        model="llama3.3-70b"
    )

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=dimension, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create Chroma vector store (with persistence)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    return qa_chain


def test_chroma_rag(qa_chain, queries):
    """
    Runs a series of queries against the provided RetrievalQA chain (Chroma),
    printing out time metrics and responses.
    """
    print("\n========== Testing Chroma RAG Latency ==========")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        start_time = time.time()
        output = qa_chain.run(query)
        total_time = time.time() - start_time
        print(f"[Chroma] Query {i} Latency: {total_time:.2f} seconds")
        print(f"[Chroma] LLM Output: {output}")


# ------------------ Pinecone RAG Setup & Test ------------------
def delete_pinecone_index(pc: PineconeClient, index_name: str):
    """Safely delete a Pinecone index if it exists."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"[Pinecone] Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        time.sleep(5)  # Give some time for the deletion to complete


def create_pinecone_index(pc: PineconeClient, index_name: str, dimension: int):
    """Create a new Pinecone index with specified dimensions."""
    print(f"[Pinecone] Creating new Pinecone index '{index_name}' with dimension {dimension}...")
    pc.create_index(
        name=index_name,
        metric="cosine",
        dimension=dimension,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)  # Give some time for index to be ready


def setup_pinecone_rag(documents, embedding_model: str):
    """
    Sets up a RetrievalQA chain using Pinecone as the vector store.
    Embedding is handled by a specified SentenceTransformer model.
    LLM is ChatCerebras (can be replaced with another model if needed).
    """

    # Validate the embedding model
    if embedding_model not in EMBEDDING_MODELS.values():
        raise ValueError(f"Embedding model must be one of: {list(EMBEDDING_MODELS.values())}")

    # Get actual embedding dimension
    dimension = get_embedding_dimension(embedding_model)
    print(f"[Pinecone] Using embedding model: {embedding_model}")
    print(f"[Pinecone] Detected embedding dimension: {dimension}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you have a GPU
    )

    # Initialize LLM
    llm = ChatCerebras(
        api_key=CHAT_CEREBRAS_API_KEY,
        model="llama3.3-70b"
    )

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=dimension, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialize Pinecone client
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    # Recreate the index if dimension mismatch or index doesn't exist
    if PINECONE_INDEX_NAME in existing_indexes:
        # Describe stats to see if dimension matches
        index_ref = pc.Index(PINECONE_INDEX_NAME)
        try:
            stats = index_ref.describe_index_stats()
            existing_dim = stats.dimension
            if existing_dim != dimension:
                print(f"[Pinecone] Dimension mismatch: existing={existing_dim}, required={dimension}. Recreating index.")
                delete_pinecone_index(pc, PINECONE_INDEX_NAME)
                create_pinecone_index(pc, PINECONE_INDEX_NAME, dimension)
        except Exception as e:
            print(f"[Pinecone] Could not describe index stats, recreating index: {e}")
            delete_pinecone_index(pc, PINECONE_INDEX_NAME)
            create_pinecone_index(pc, PINECONE_INDEX_NAME, dimension)
    else:
        # Create index
        create_pinecone_index(pc, PINECONE_INDEX_NAME, dimension)

    # Now get the index reference
    index_ref = pc.Index(PINECONE_INDEX_NAME)

    # Create vector store
    vectorstore = Pinecone(index=index_ref, embedding=embeddings, text_key="text")

    # Upsert documents
    print("[Pinecone] Upserting documents...")
    vectorstore.add_documents(docs)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    return qa_chain


def test_pinecone_rag(qa_chain, queries):
    """
    Runs a series of queries against the provided RetrievalQA chain (Pinecone),
    printing out time metrics and responses.
    """
    print("\n========== Testing Pinecone RAG Latency ==========")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        start_time = time.time()
        output = qa_chain.run(query)
        total_time = time.time() - start_time
        print(f"[Pinecone] Query {i} Latency: {total_time:.2f} seconds")
        print(f"[Pinecone] LLM Output: {output}")


# ------------------ Main ------------------
def main():
    # 1) Load documents from arxiv_papers directory
    try:
        documents = load_arxiv_papers()
        print(f"Successfully loaded {len(documents)} total pages from arxiv papers.")
    except Exception as e:
        print(f"Error loading arxiv papers: {str(e)}")
        return

    # 2) Test Chroma RAG
    try:
        print("\n--- Setting up Chroma RAG (MiniLM) ---")
        chroma_chain = setup_chroma_rag(documents, EMBEDDING_MODELS["minilm"])
        test_chroma_rag(chroma_chain, TEST_QUERIES)
    except Exception as e:
        print(f"[Chroma] Error occurred: {e}")

    # 3) Test Pinecone RAG
    try:
        print("\n--- Setting up Pinecone RAG (MiniLM) ---")
        pinecone_chain = setup_pinecone_rag(documents, EMBEDDING_MODELS["minilm"])
        test_pinecone_rag(pinecone_chain, TEST_QUERIES)
    except Exception as e:
        print(f"[Pinecone] Error occurred: {e}")


if __name__ == "__main__":
    main()

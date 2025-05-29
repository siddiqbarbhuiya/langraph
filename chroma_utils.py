# chroma_utils.py
import os
from typing import List, Dict, Any

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma # Import for LangChain's Chroma wrapper

class ChromaDBManager:
    def __init__(self, collection_name: str = "resumes", persist_directory: str = "./chroma_db"):
        """
        Initializes the ChromaDBManager.

        Args:
            collection_name (str): The name of the collection in ChromaDB.
            persist_directory (str): The directory to persist the ChromaDB data.
        """
        self.persist_directory = persist_directory
        # Initialize ChromaDB client to persist data to disk
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Get or create the collection using ChromaDB client directly
        # This ensures the collection exists before LangChain's Chroma wrapper tries to use it.
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Initialize LangChain's Chroma wrapper for easier integration with LangChain tools/agents
        self.langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection.name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory # Important for persistence
        )
        print(f"ChromaDB initialized. Collection: '{collection_name}', Path: '{persist_directory}'")

    def add_resumes(self, resumes: List[Dict[str, Any]]):
        """
        Adds resumes to the ChromaDB collection using LangChain's Chroma wrapper.

        Args:
            resumes (List[Dict[str, Any]]): A list of resume dictionaries,
                                             each with 'id', 'content', and 'metadata'.
        """
        # Convert raw resume data to LangChain Document objects
        lc_documents = [
            Document(page_content=resume["content"], metadata=resume["metadata"])
            for resume in resumes
        ]

        print(f"Adding {len(lc_documents)} documents to ChromaDB...")
        # LangChain's Chroma.add_documents handles embedding generation and storage
        self.langchain_chroma.add_documents(lc_documents)
        print("Resumes added to ChromaDB.")

    def search_resumes(self, query: str, k: int = 3) -> List[Document]:
        """
        Searches for resumes similar to the query using LangChain's Chroma wrapper.

        Args:
            query (str): The search query.
            k (int): The number of top similar resumes to retrieve.

        Returns:
            List[Document]: A list of LangChain Document objects representing the
                            most similar resumes.
        """
        print(f"Searching ChromaDB for query: '{query}' with k={k}")
        results = self.langchain_chroma.similarity_search(query, k=k)
        print(f"Found {len(results)} results.")
        return results

# Example usage (for initial data loading/testing)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found. Please set it in your environment or .env file.")
        exit()

    from resume_data import RESUMES

    db_manager = ChromaDBManager()

    # Clear existing collection for a fresh start (optional, for development)
    try:
        # Before deleting, ensure the client is properly initialized
        temp_client = chromadb.PersistentClient(path="./chroma_db")
        temp_client.delete_collection(name="resumes")
        print("Existing 'resumes' collection deleted.")
    except Exception as e:
        print(f"Could not delete collection (might not exist or error): {e}")
    
    # Re-initialize db_manager after potential deletion to ensure a clean state
    db_manager = ChromaDBManager()
    db_manager.add_resumes(RESUMES)

    # Test search
    print("\n--- Testing Search ---")
    query = "experienced Python developer"
    found_resumes = db_manager.search_resumes(query)
    for i, resume in enumerate(found_resumes):
        print(f"Result {i+1}:")
        print(f"  Name: {resume.metadata.get('name', 'N/A')}")
        print(f"  Role: {resume.metadata.get('role', 'N/A')}")
        print(f"  Content Snippet: {resume.page_content[:100]}...")
        print("-" * 20)

    query = "data scientist with machine learning"
    found_resumes = db_manager.search_resumes(query)
    print(f"\n--- Testing Search for '{query}' ---")
    for i, resume in enumerate(found_resumes):
        print(f"Result {i+1}:")
        print(f"  Name: {resume.metadata.get('name', 'N/A')}")
        print(f"  Role: {resume.metadata.get('role', 'N/A')}")
        print(f"  Content Snippet: {resume.page_content[:100]}...")
        print("-" * 20)
# import chromadb
# # from langchain.embeddings import GoogleGenerativeAIEmbeddings
# # from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# def get_chroma_db():
#     client = chromadb.PersistentClient(path="vector_store")
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return Chroma(client=client, collection_name="docs", embedding_function=embedding)

# def store_chunks_in_chroma(docs):
#     vectordb = get_chroma_db()
#     vectordb.add_texts(docs)

# def query_chroma(query):
#     vectordb = get_chroma_db()
#     return vectordb.similarity_search(query, k=3)

import chromadb
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # NEW
import os

# Initialize Chroma vector store locally
def get_chroma_db():
    persist_directory = "./chroma_db"
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})  # Small, fast model
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb

# Store text chunks into Chroma
def store_chunks_in_chroma(chunks):
    vectordb = get_chroma_db()
    vectordb.add_texts(chunks)
    vectordb.persist()

def query_chroma(query):
      vectordb = get_chroma_db()
      return vectordb.similarity_search(query, k=3)

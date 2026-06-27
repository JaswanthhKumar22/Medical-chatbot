from typing import List
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    load=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents= load.load()
    return documents
    
def filter_to_minimal_docs(docs: List [Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List [Document] = []
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document (
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

#split the documents into smaller chunnks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents (minimal_docs)
    return texts_chunk

def download_embeddings():
    """"
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings= HuggingFaceEmbeddings (
        model_name=model_name,
    )
    return embeddings

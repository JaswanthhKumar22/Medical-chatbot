from typing import List
import os
from typing import List
from langchain.schema import Document
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
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROK_API_KEY=os.getenv("GROK_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GROK_API_KEY"]=GROK_API_KEY

extracted_data=load_pdf_files("data")
minimal_docs=filter_to_minimal_docs(extracted_data)
texts_chunk=text_split(minimal_docs)

embedding=download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbox"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
index=pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)
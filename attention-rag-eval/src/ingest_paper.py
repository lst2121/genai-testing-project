import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import config

load_dotenv()

print("📄 Loading PDF...")
loader = PyPDFLoader(config.PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages from the paper")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

print("🔪 Splitting text into chunks...")
docs = text_splitter.split_documents(documents)
print(f"Generated {len(docs)} text chunks")

print("🔎 Initializing OpenAI Embeddings...")
embeddings = OpenAIEmbeddings()

if os.path.exists(config.PERSIST_DIRECTORY) and os.listdir(config.PERSIST_DIRECTORY):
    print("📁 Chroma DB already exists. Loading existing index...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
        collection_name=config.EMBEDDING_COLLECTION_NAME
    )
else:
    print("💾 Storing chunks in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIRECTORY,
        collection_name=config.EMBEDDING_COLLECTION_NAME
    )
    print("Chroma DB created and persisted at:", config.PERSIST_DIRECTORY)
# src/rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
import config

load_dotenv()

# Initialize retriever
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=config.PERSIST_DIRECTORY,
    collection_name=config.EMBEDDING_COLLECTION_NAME
)
retriever = vectorstore.as_retriever()

# Initialize LLM with fallback
def get_llm():
    """Get LLM with fallback to OpenAI if DeepSeek not available"""
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            return ChatDeepSeek(model=config.DEEPSEEK_CHAT_MODEL)
        except Exception as e:
            print(f"DeepSeek initialization failed: {e}")
            print("Falling back to OpenAI...")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Neither DEEPSEEK_API_KEY nor OPENAI_API_KEY found in environment")
    
    print("Using OpenAI as LLM")
    return ChatOpenAI(model=config.OPENAI_CHAT_MODEL)

llm = get_llm()

# Prompt for final answer
rag_prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {question}
    """
)

# Build LCEL-style pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda inputs: {
        "context": "\n\n".join(doc.page_content for doc in inputs["context"]),
        "question": inputs["question"]
    })
    | rag_prompt
    | llm
)

# Entry point for RAG inference
def run_rag_pipeline(question: str) -> str:
    return rag_chain.invoke(question).content

# CLI for quick test
if __name__ == "__main__":
    query = input("Ask a question from 'Attention is All You Need': ")
    print("Answer:", run_rag_pipeline(query))

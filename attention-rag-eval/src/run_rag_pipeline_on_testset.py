# src/run_rag_pipeline_on_testset.py

import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
import config

load_dotenv()

# Load generated testset
df = pd.read_csv("data/ragas_outputs/generated_testset.csv")

# Initialize retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=config.PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name=config.EMBEDDING_COLLECTION_NAME,
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

# RAG pipeline using LCEL
template = """Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}"""
prompt = PromptTemplate.from_template(template)

rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"],
}) | {
    "question": lambda x: x["question"],
    "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
} | prompt | llm | StrOutputParser()

# Run for all samples
results = []
print("Running RAG pipeline on test set...")
for i, row in df.iterrows():
    question = row["user_input"]
    reference = row["reference"]
    print(f"Processing question {i+1}/{len(df)}: {question[:50]}...")
    response = rag_chain.invoke({"question": question})
    context_docs = retriever.invoke(question)
    chunks = [doc.page_content for doc in context_docs]

    results.append({
        "question": question,
        "response": response,
        "retrieved_chunks": chunks,
        "ground_truth": reference
    })

Dataset.from_list(results).to_csv("data/ragas_outputs/rag_predictions.csv", index=False)
print("RAG results saved to rag_predictions.csv")

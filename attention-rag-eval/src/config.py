# === Paths ===
PDF_PATH = "data/paper/Attention Is All You Need.pdf"
CHROMA_PATH = "embeddings/attention_chroma"

# === Chunking Params ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === Model Names Only (keys are loaded via .env)
# OpenAI
OPENAI_LLM_MODEL = "gpt-4o-mini"  # For RAGAS or LangSmith eval
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"  # Fallback for RAG generation

# DeepSeek
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

# === Chroma Vector DB
PERSIST_DIRECTORY = CHROMA_PATH
EMBEDDING_COLLECTION_NAME = "attention_paper_chunks"

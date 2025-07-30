# src/generate_testset.py

from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

load_dotenv()

# Load and chunk PDF
print("📄 Loading and chunking paper...")
loader = PyPDFLoader(config.PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"{len(chunks)} chunks ready for testset generation")

# Setup LLM + embeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model=config.OPENAI_LLM_MODEL))
generator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Initialize testset generator
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_emb)

# Generate testset
print("Generating test questions...")
testset = generator.generate_with_langchain_docs(chunks, testset_size=10)

# Save to disk
testset.to_pandas().to_csv("data/ragas_outputs/generated_testset.csv", index=False)
print("Testset saved to data/ragas_outputs/generated_testset.csv")

from pathlib import Path
import yaml
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.agents.state import AgentState
CURRENT_DIR = Path.cwd()
with open(CURRENT_DIR / "config" / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
    embedding_model_name = config['EMBEDDING_MODEL']

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

pdf_path = CURRENT_DIR / "reports" / "_10-K-2025-As-Filed.pdf"
loader = PyPDFLoader(pdf_path).load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(loader)

vector_db = Chroma.from_documents(texts, embeddings)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def retriever_node(state: AgentState):
    docs = retriever.invoke(state["question"])
    combined_text = " ".join([doc.page_content for doc in docs])
    return {"documents": combined_text}
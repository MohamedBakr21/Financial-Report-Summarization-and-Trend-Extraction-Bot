from  pathlib import Path
import yaml
from langchain_groq import ChatGroq
CURRENT_DIR = Path.cwd()
print(CURRENT_DIR)
# Load config
with open(CURRENT_DIR / "config" / "config.yaml", "r") as file:
    config = yaml.safe_load(file)

api_key = config["GROQ_API_KEY"]
# Initialize LLM - Using a standard Groq model ID
llm = ChatGroq(
    api_key=api_key, 
    model="llama-3.3-70b-versatile", 
    temperature=0
)
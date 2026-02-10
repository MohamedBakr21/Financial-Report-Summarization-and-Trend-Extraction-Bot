# Financial-Report-Summarization-and-Trend-Extraction-Bot

An AI-powered agentic system designed to ingest complex financial documents (like 10-K filings), retrieve specific data points, and perform high-level analysis. The bot utilizes a **StateGraph** (via LangGraph) to intelligently decide between direct answering, concise summarization, or dynamic data visualization.

## 🚀 Features

* **Agentic Orchestration**: Uses `LangGraph` to manage a stateful workflow that routes queries between a reasoning engine and specialized tools.
* **Intelligent RAG**: Implements `Chroma DB` and `RecursiveCharacterTextSplitter` to process long-form PDFs and retrieve contextually relevant chunks.
* **Query Optimization**: Features a `transform_query` node that rewrites conversational user input into search-engine-friendly queries for better retrieval accuracy.
* **Specialized Summarization**: Integrates a local `distilbart-cnn-12-6` HuggingFace pipeline for high-compression financial summaries.
* **Dynamic Visualizations**: Includes a custom tool that uses Llama 3.3 to write and execute `matplotlib` code on-the-fly, generating trend charts directly from retrieved financial data.

## 🛠️ Tech Stack

* **LLM Orchestration**: LangGraph, LangChain
* **Large Language Model**: Llama-3.3-70b (via Groq)
* **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database**: Chroma DB
* **Summarization**: BART (HuggingFace Transformers)
* **Data Viz**: Matplotlib, Seaborn

## 📂 Project Structure

```text
Financial-Report-Summarization-and-Trend-Extraction-Bot/
├── config/
│   └── config.yaml          # API Keys and Model configurations
├── reports/
│   └── 10-K-2025.pdf        # Source financial PDF documents
├── charts/                  # Auto-generated directory for output charts (PNG)
├── rag.ipynb                # Main agent implementation and logic
├── .gitignore               # Excludes config/ and local vector store
└── README.md                # Project documentation

```

## ⚙️ Setup & Installation

1. **Clone the Repository**
```bash
git clone https://github.com/MohamedBakr21/Financial-Report-Summarization-and-Trend-Extraction-Bot.git
cd Financial-Report-Summarization-and-Trend-Extraction-Bot

```


2. **Configuration**
Create a `config/config.yaml` file (one level above or inside your project root as per your notebook paths) with your Groq API key:
```yaml
GROQ_API_KEY: "your_groq_api_key"
EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"

```


3. **Install Dependencies**
```bash
pip install langchain langchain-chroma langchain-huggingface \
            langchain-groq langgraph transformers torch \
            matplotlib PyPDF2 pyyaml

```


4. **Place Data**
Ensure your financial reports are placed in the `../reports/` directory as expected by the `PyPDFLoader` in the notebook.

## 🧠 Workflow Logic

The agent operates as a **State Machine**:

1. **Query Transformation**: The user question is optimized.
2. **Retrieval**: Relevant document context is fetched from the vector store.
3. **Reasoner**: The LLM analyzes the context and the question.
* If a summary is needed, it calls the `summarize_tool`.
* If data trends are identified, it calls the `generate_and_run_chart` tool.


4. **Tool Execution**: The code is executed, and results (or chart paths) are returned to the agent for a final response.

## 📊 Example Queries

* *"What are the main risk factors for the upcoming fiscal year? Summarize them."*
* *"Extract the revenue growth data for the last 3 quarters and create a bar chart."*

## 📝 License

Distributed under the MIT License.

import os
import re
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from transformers import pipeline
from src.helpers.llm import llm
from langgraph.graph import END
# Load config
from  pathlib import Path
CURRENT_DIR = Path.cwd()
summarization_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 
)

@tool
def summarize_tool(text: str) -> str:
    """Summarizes the input text into a concise and easy-to-read summary."""
    input_len = len(text.split())
    max_len = min(150, max(30, input_len))
    summary = summarization_pipeline(text, max_length=max_len, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

@tool
def generate_and_run_chart(text_segment: str) -> str:
    """
    Generates financial charts from text and saves the PNG file 
    with a name corresponding to the chart's title.
    """
    os.makedirs(CURRENT_DIR/"charts", exist_ok=True)
    
    # Updated prompt to specifically ask for a dynamic filename based on content
    prompt = (
        f"Based on this data: {text_segment}, generate Python matplotlib code to create a professional bar chart. "
        f"1. Determine a short, descriptive filename based on the chart title (e.g., 'revenue_trends.png'). "
        f"2. Save the chart using plt.savefig(os.path.join('charts', filename)). "
        f"3. Return ONLY the python code."
    )
    
    llm_response = llm.invoke([HumanMessage(content=prompt)])
    # Clean the response to get raw code
    code = re.sub(r"```(?:python)?", "", llm_response.content, flags=re.IGNORECASE).strip()

    try:
        # We pass os and plt into the exec globals so the generated code can use them
        exec_globals = {"plt": plt, "os": os}
        exec(code, exec_globals)
        
        # Optionally, try to find what filename the LLM chose to return a better confirmation
        return "Chart generated and saved successfully in the 'charts/' directory."
    except Exception as e:
        return f"Error executing chart code: {e}"

tools = [generate_and_run_chart, summarize_tool]
router_llm = llm.bind_tools(tools)

def check_next(state):
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    
    # If the LLM didn't return tool_calls, it has provided a final answer
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    # Route to the tools node
    return "tools"
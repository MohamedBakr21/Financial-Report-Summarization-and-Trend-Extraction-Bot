from src.agents.state import AgentState
from langchain_core.messages import HumanMessage
from src.tools.tools import router_llm
def reasoner(state: AgentState):
    print("---REASONER---")
    context = state.get("documents", "")
    question = state.get("question", "")
    system_msg = (
        "You are a financial analyst. Use the provided context to answer the user. "
        "If you see a summary from a tool in the history, use it to give a final answer. "
        "Do not call a tool if the information is already present in the messages."
    )
    
    # Construct the message list including the history
    input_content = f"Context: {context}\n\nQuestion: {question}"
    messages = [
        HumanMessage(content=f"{system_msg}\n\n{input_content}")
    ] + list(state.get("messages", [])) # Include previous tool outputs
    
    response = router_llm.invoke(messages)
    
    # Return only the NEW response to be appended to state
    return {"messages": [response]}

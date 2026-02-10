from src.agents.state import AgentState
from langgraph.graph import StateGraph
from src.agents.reasoner_agent import reasoner
from src.agents.transform_query import transform_query
from src.agents.retriever_agent import retriever_node
from src.tools.tools import tools ,check_next
from langgraph.prebuilt import ToolNode
from langgraph.graph import START ,END

workflow = StateGraph(AgentState)
workflow.add_node("transform_query", transform_query)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "retriever")
workflow.add_edge("retriever", "reasoner")

workflow.add_conditional_edges("reasoner", check_next, {"tools": "tools", END: END})
workflow.add_edge("tools", "reasoner")

app = workflow.compile()

# Test Run
inputs = {"question":"""accroding to the pdf  draw the graph for Segment Operating Performance """, "messages": []}

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"--- Node: {key} ---")
        if "messages" in value:
            # Print the text content of the latest message
            msg = value["messages"][-1]
            if msg.content:
                print(msg.content)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"Tool Calls: {[t['name'] for t in msg.tool_calls]}")
    print("\n")
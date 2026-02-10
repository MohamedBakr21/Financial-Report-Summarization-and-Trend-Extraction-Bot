from src.prompts.prompts import query_optimizer_prompt
from langchain_core.output_parsers import StrOutputParser
from src.agents.state import AgentState
from src.helpers.llm import llm
def transform_query(state: AgentState):
    prompt = query_optimizer_prompt
    chain = prompt | llm | StrOutputParser()
    transformed_query = chain.invoke({"question": state['question']})
    return {"question": transformed_query}
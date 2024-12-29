# Import relevant functionality
import os
from langchain_ollama import ChatOllama
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Connection LangSmith Tracings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "LangChain-SimpleLLMAgent-Tutorial"

# Create the agent
memory = MemorySaver()
model = ChatOllama(
    model = "llama3.2:1b",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)
search = SearxSearchWrapper(searx_host="http://192.168.0.64:8888")

search_results = search.run("what is the weather in SF")
print(search_results)
'''
tools = [search]
model_with_tools = model.bind_tools(tools)
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")'''
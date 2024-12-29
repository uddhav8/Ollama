# Import relevant functionality
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import TavilySearchResults, SearxSearchResults
from langgraph.prebuilt import create_react_agent

# Connection LangSmith Tracings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "LangChain-SimpleLLMAgent-Tutorial"

# Referencing the model
model = ChatOllama(
    model = "llama3.2:1b",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

memory = MemorySaver()

# Initialize the SearxSearchWrapper
wrapper = SearxSearchWrapper(
    searx_host="http://192.168.0.64:8888",  # URL of the Searx instance
)

# Define Tools
# search = TavilySearchResults(max_results=2)
search = SearxSearchResults(
    wrapper=wrapper,
    max_results=2,
    description="A meta search engine. Useful for answering questions about current events.",
)
#print(search.invoke("what is the weather in SF"))

# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

# Using Tool with Language Models
model_with_tools = model.bind_tools(tools)

'''response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")'''

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

response["messages"]

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
response["messages"]

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")
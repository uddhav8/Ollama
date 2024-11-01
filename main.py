from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Define LLM Provider.
model = ChatOllama(
    model = "llama3.2:latest", # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

# Mesages to the LLM and the User.
messages = [
    SystemMessage(content="Translate the following from English into Spanish"),
    HumanMessage(content="Hello! How are you?"),
]

print(model.invoke(messages))
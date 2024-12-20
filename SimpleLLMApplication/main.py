# Create a prompt template and chain it together 

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "LangChain-SimpleLLMApp-Tutorial"

# Define LLM Provider.
model = ChatOllama(
    model = "llama3.2:1b",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

# Define the message template to provide to the llm.
system_template = "Translate the following into {language}:" # language will be replaced by user input

# The messages to send to the LLM.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template), 
        ("user", "{text}")
    ]
)

# Define Output Parser
parser = StrOutputParser()  # This parses the output of the LLM into a more human readable.

# Chain the LLM and Parser to that every time this chain is called the model and parser will be called.
chain = prompt_template | model | parser    # What this says: Chain together in the order whenever calling Chain, first run prompt_template Then model Then parser 

# print output in terminal
print(chain.invoke({"language": "italian", "text": "Hello! How are you?"}))
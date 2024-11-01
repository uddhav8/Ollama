from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

'''
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-kindly-address-51"
'''
'''
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
'''

# Define LLM Provider.
model = ChatOllama(
    model = "llama3.2:latest",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

# Define the message template to provide to the llm.
system_template = "Translate the following into {language}:" # language will be replaced by user input

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template), 
        ("user", "{text}")
    ] # The messages to send to the LLM.
)

result = prompt_template.invoke({"language": "Spanish", "text": "Hello! How are you?"})
print(result)

'''
# Define Output Parser
parser = StrOutputParser()  # This parses the output of the LLM into a more human readable.

# Chain the LLM and Parser to that every time this chain is called the model and parser will be called.
chain = model | parser

# print output in terminal
print(chain.invoke(messages))'''
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model = "llama3.2:latest",
    base_url= "http://192.168.0.64:11434/",
    # other params ...
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

'''messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]'''

print(llm.invoke(messages))


'''
parser = StrOutputParser()

result = llm.invoke(messages)

chain = llm | parser

chain.invoke(result)'''
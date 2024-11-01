# execute this file we should see our chain being served at http://localhost:8000.

#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langserve import add_routes
from langserve import RemoteRunnable


# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
# Define LLM Provider.
model = ChatOllama(
    model = "llama3.2:latest",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

'''
# This codeline is used to interact the server as a client directly via code. i.e. as if using from http://localhost:8000/chain/playground/ 
remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
'''
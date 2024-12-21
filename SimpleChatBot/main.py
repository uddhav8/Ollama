import os
from getpass import getpass
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Connection LangSmith Tracings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "LangChain-SimpleChatBot-Tutorial"

# Define LLM Provider.
model = ChatOllama(
    model = "llama3.2:1b",  # The model name and version to use for inference.
    base_url= "http://192.168.0.64:11434/", # The URL of the LLM.
    # other params ...
)

# Define the trimmer for conversation history
trimmer = trim_messages(
    max_tokens=65, # Adjust token limit based on your model's context window
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

 # Define Prompt Template.
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You area helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define A class called State of the type TypedDict.
# This dictionary consists of two keys "messages" and "language".
# This keys we will use to pass a message and ask it to respond in the set language. 
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define a new graph
# A Graph is like a whiteboard. 
# A Schema is like a sticky note template that can be stuck on the whiteboard.
workflow = StateGraph(state_schema=State)

initial_messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Define the function that calls the model
def call_model(state: State):
    # Trim messages to fit within context window
    trimmed_messages = trimmer.invoke(state["messages"])
    # Build the prompt with the trimmed messages
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

# Define the (single) node in the graph
# Define which node (each sticky note on the whiteboard) should be the first and how the chain of notes will follow.
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()

# Run the workflow
app = workflow.compile(checkpointer=memory)


# Run the app
'''config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc123"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc345"}}

query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc456"}}

query = "Hi! I'm Bob."

language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": language}, config,)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config,)
output["messages"][-1].pretty_print()
'''
config = {"configurable": {"thread_id": "abc567"}}

query = "What is my name?"

language = "English"

input_messages = initial_messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = initial_messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()
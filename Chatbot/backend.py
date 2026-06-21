from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

################### Load environment variables from .env file ###################
load_dotenv()

################### Initialize language model ###################
# llm = ChatGroq(model="qwen/qwen3-32b")
llm = OllamaLLM(
    model="gemma4:31b-cloud",  # Or any model you've loaded in Ollama
)

################### Define Chat State ###################
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

################### Define Chat Node ###################
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

################### Define Checkpointer ###################
checkpointer = InMemorySaver()

################### Create Graph ###################
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
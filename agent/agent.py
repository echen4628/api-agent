from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


from langchain_openai import ChatOpenAI
from utils.constants import GPT_4o


class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def api_chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def decide_next(state: State):
    # decides whether I need to call a tool or not
    # check if it is an internal tool

    # if so call_tools

    # else end
    pass

def call_tools(state: State):
    # call the tool
    pass

def decide_start(state: State):
    # check if the message is a user message or a tool response
    # 
    # if it is a user message, call the chatbot
    # else call the the api chatbot
    pass


graph_builder = StateGraph(State)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    def stream_graph_updates(user_input: str):
        for events in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                                config,
                                stream_mode="updates"):
            # import pdb; pdb.set_trace()
            for value in events.values():
                print("Assistant:", value["messages"][-1].content)


    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

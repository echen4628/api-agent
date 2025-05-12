import json
from langchain_core.messages import ToolMessage
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.base import StructuredTool

from langchain_openai import ChatOpenAI
from utils.constants import (GPT_4o,
                             TEXT_EMBEDDING_3_LARGE,
                             INPUTS_DESC_VECTOR_STORE_PATH,
                             OUTPUT_DESC_VECTOR_STORE_PATH,
                             FUNC_DESC_VECTOR_STORE_PATH,
                             NAME_TO_FUNCTION_JSON_PATH)
from langchain_core.tools import tool

from utils.FunctionTools import FunctionDatabase

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage



@tool
def add(a: int, b: int) -> int:
    '''
    Adds two intergers a and b
    '''
    return a+b


functionDatabase = FunctionDatabase(TEXT_EMBEDDING_3_LARGE,
                                    INPUTS_DESC_VECTOR_STORE_PATH,
                                    OUTPUT_DESC_VECTOR_STORE_PATH,
                                    FUNC_DESC_VECTOR_STORE_PATH,
                                    NAME_TO_FUNCTION_JSON_PATH )

tools = [add, 
         StructuredTool.from_function(functionDatabase.search),
         StructuredTool.from_function(functionDatabase.find_dependency)]


class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)
llm_with_tools = llm.bind_tools(tools)

prompt_template = ChatPromptTemplate([
    ("system", """You are a helpful coding assistant that reasons and acts step-by-step using the ReAct pattern.

Follow this format for each step:

Thought: Describe what you’re trying to do or figure out.
Action: invoke a tool
Observation: Reflect on what the tool returned and decide the next step.

Repeat the loop until the user’s question is fully answered.

Here is an example:

User: I want to analyze a dataset of GitHub repos and find which users have contributed to the most repositories.

Thought: To solve this, I need a function that can help with analyzing GitHub contribution data.
Action: [something like search(query='analyze GitHub user contributions')]

Observation: The function `analyze_user_contributions` matches your query.

Thought: I should now find what inputs or setup this function depends on.
Action: [something like find_dependency(function_names=['analyze_user_contributions'])]

Observation: The function `analyze_user_contributions` depends on: `load_github_dataset`, `parse_contributions_log`.

Thought: These functions may require specific file inputs or user-provided parameters. I should ask the user about available data or let them specify the input file path.

--- End of example ---

Now begin helping the user. Think step by step. Use tools when necessary.
"""),
    MessagesPlaceholder("msgs")
])

react_llm = prompt_template | llm_with_tools


def chatbot(state: State):
    return {"messages": [react_llm.invoke({"msgs": state["messages"]})]}

def decide_next(state: State):
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "call_tools"
    return END


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        print('there was a tool call')
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# def decide_start(state: State):
#     # check if the message is a user message or a tool response
#     #
#     # if it is a user message, call the chatbot
#     # else call the the api chatbot
#     pass


graph_builder = StateGraph(State)

# nodes

graph_builder.add_node("chatbot", chatbot)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("call_tools", tool_node)

# edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", decide_next)
graph_builder.add_edge("call_tools", "chatbot")

# compile with memory for messages
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

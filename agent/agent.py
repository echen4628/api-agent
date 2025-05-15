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
                             NAME_TO_FUNCTION_JSON_PATH,
                             PLANNING_AGENT_PROMPT)
from langchain_core.tools import tool, InjectedToolArg

from utils.FunctionTools import FunctionDatabase

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage



@tool
def add(a: int, b: int) -> int:
    '''
    Adds two intergers a and b
    '''
    return a+b

from pydantic import BaseModel, Field
from typing import Literal, Dict, Union, List
class APIStep(TypedDict):
    id: int
    var: str
    action: Literal['call']  # 'call', 'extract', 'ask_user'
    tool: str
    args: Dict[str, Union[str, float, int, bool]]

class ExtractStep(TypedDict):
    id: int
    var: str
    action: Literal['extract'] 
    source: str
    path: str

class AskUserStep(TypedDict):
    id: int
    var: str
    action: Literal['ask_user'] 
    query: str

Step = Union[APIStep, ExtractStep, AskUserStep]


functionDatabase = FunctionDatabase(TEXT_EMBEDDING_3_LARGE,
                                    INPUTS_DESC_VECTOR_STORE_PATH,
                                    OUTPUT_DESC_VECTOR_STORE_PATH,
                                    FUNC_DESC_VECTOR_STORE_PATH,
                                    NAME_TO_FUNCTION_JSON_PATH )


plan = []
class State(TypedDict):
    plan: List[Step]
    messages: Annotated[list, add_messages]

def extract_last_k_steps(k: int, steps: List[Step]) -> str:
    last_x = min(k, len(steps))
    response = []
    if last_x < k:
        response.append("--Beginning of Plan--")
    for i in range(len(steps)-last_x, len(steps)):
        response.append(str(steps[i]))
    return "\n\n".join(response)


class AddStepToPlan(BaseModel):
    """
    Tool that adds a subsequent PlanStep object to the linear execution plan.
    """

    plan_step: Step = Field(..., description="The next step to add to the plan")
    steps: Annotated[List, InjectedToolArg] = Field(..., description="The overall plan")


@tool(args_schema=AddStepToPlan)
def add_step_to_plan(plan_step, steps) ->str:
    try:
        steps.append(plan_step)
        return f"Successfully added step. Now, the plan has {len(steps)} steps. The latest few steps are:\n{extract_last_k_steps(3, steps)}", steps
    except Exception as e:
        return str(e), steps

@tool
def finish_plan() ->str:
    '''
    Notify the user the plan has been finished
    '''
    return "plan finished"

tools = [add, 
         StructuredTool.from_function(functionDatabase.search),
         StructuredTool.from_function(functionDatabase.find_dependency),
         add_step_to_plan,
         finish_plan]

memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)
llm_with_tools = llm.bind_tools(tools)
with open(PLANNING_AGENT_PROMPT, "r") as f:
    planning_prompt = f.read()

prompt_template = ChatPromptTemplate([
    ("system", planning_prompt),
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

    def __call__(self, state: dict):
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        print('there was a tool call')
        for tool_call in message.tool_calls:
            if tool_call["name"] == "add_step_to_plan":
                tool_call["args"]["steps"] = state.get("plan", [])
                tool_result, steps = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                state["plan"] = steps
            else:
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
        return {"messages": outputs, "plan": state["plan"]}

# def decide_start(state: State):
#     # check if the message is a user message or a tool response
#     #
#     # if it is a user message, call the chatbot
#     # else call the the api chatbot
#     pass
def initialization(state: State):
    state["plan"] = plan
    return state


graph_builder = StateGraph(State)

# nodes
graph_builder.add_node("initialization", initialization)
graph_builder.add_node("chatbot", chatbot)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("call_tools", tool_node)

# edges
graph_builder.add_edge(START, "initialization")
graph_builder.add_edge("initialization", "chatbot")
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
            for value in events.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)

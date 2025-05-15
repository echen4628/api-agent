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
        return f"Successfully added step. Now, the plan has {len(steps)} steps. The latest few steps are:\n{extract_last_k_steps(3, steps)}"
    except Exception as e:
        return str(e)

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
prompt_template = ChatPromptTemplate([
    ("system", """You are a helpful coding assistant that reasons and acts step-by-step using the ReAct pattern. Your job is to search for APIs that will assist the user and then create a plan with a full dependency graph on how those APIs will be called. Note that you do not execute these APIs — your goal is to generate a correct execution plan in JSON.

You will reason step-by-step:

Thought: Describe what you’re trying to do or figure out.
Action: invoke a tool
Observation: Reflect on what the tool returned and decide the next step.
Start by looking up functions using `search` tool. Then use the `find_dependency` tool to learn dependencies. 
As you gather enough information, begin constructing a plan using the `add_step_to_plan` tool. Each step in the plan should be a structured JSON object of one of the following types:

1. Call a tool:
    {{
      "id": 1,
      "var": "${{1}}",
      "action": "call",
      "tool": "tool_name",
      "args": {{
        "arg1": "value1",
        "arg2": 3.5
      }}
    }}

2. Extract a field from a previous step’s output:
    {{
      "id": 2,
      "var": "${{2}}",
      "action": "extract",
      "source": "${{1}}",
      "path": "some.field.name"
    }}

3. Ask the user for input:
    {{
      "id": 3,
      "var": "${{3}}",
      "action": "ask_user",
      "query": "your question here"
    }}

You should create the plan incrementally using the `add_step_to_plan` tool. Once you believe the plan is complete, invoke the `finish_plan` tool.

---

Example 1:
User: I want to analyze a dataset of GitHub repos and find which users have contributed to the most repositories.

Thought: I need a function to analyze GitHub user contributions.
Action: search(query='analyze GitHub user contributions')
Observation: Found `analyze_user_contributions`.

Thought: I should determine its dependencies.
Action: find_dependency(function_names=['analyze_user_contributions'])
Observation: It depends on `load_github_dataset`, `parse_contributions_log`.

Thought: Therefore, I should plan to call `load_github_dataset` and `parse_contributions_log` before `analyze_user_contributions`. I will start by adding `load_github_dataset` to the plan.
Action: add_step_to_plan({{
  "id": 1,
  "var": "${{1}}",
  "action": "call",
  "tool": "load_github_dataset",
  "args": {{ "path": "user_provided_path" }}
}})

Thought: Next, I will add `parse_contribution_log` to the plan.
Action: add_step_to_plan({{
  "id": 2,
  "var": "${{2}}",
  "action": "call",
  "tool": "parse_contributions_log",
  "args": {{ "dataset": "${{1}}" }}
}})

Thought: Finally, I will add `analyze_user_contributions` to the plan.
Action: add_step_to_plan({{
  "id": 3,
  "var": "${{3}}",
  "action": "call",
  "tool": "analyze_user_contributions",
  "args": {{ "parsed_data": "${{2}}" }}
}})

Thought: The plan is complete.
Action: finish_plan()
---

Example 2:
User: I want to find the duration it takes to travel from my current location to Times Square.

Thought: I need a function that gets route duration between two points.
Action: search(query='get travel duration between two points')
Observation: Found `get_travel_estimate`.

Thought: I need a way to get the destination coordinates for Times Square.
Action: search(query='get location for Times Square')
Observation: Found `lookup_place_coordinates`.

Thought: Let’s get the coordinates first.
Action: add_step_to_plan({{
  "id": 1,
  "var": "${{1}}",
  "action": "call",
  "tool": "lookup_place_coordinates",
  "args": {{ "place": "Times Square" }}
}})

Thought: I should extract just the longitude.
Action: add_step_to_plan({{
  "id": 2,
  "var": "${{2}}",
  "action": "extract",
  "source": "${{1}}",
  "path": "longitude"
}})

Thought: I need to ask the user for their current location.
Action: add_step_to_plan({{
  "id": 3,
  "var": "${{3}}",
  "action": "ask_user",
  "query": "What is your current location?"
}})

Thought: Now I call `get_travel_estimate` with both points.
Action: add_step_to_plan({{
  "id": 4,
  "var": "${{4}}",
  "action": "call",
  "tool": "get_travel_estimate",
  "args": {{
    "origin": "${{3}}",
    "destination_longitude": "${{2}}"
  }}
}})

Thought: The plan is done.
Action: finish_plan()
---

Now begin helping the user. Think step by step (and annotate using Thought, Action, and Observation). Use tools when necessary. Format the plan using the structured JSON schema when you start planning."""),
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
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
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
        return {"messages": outputs}

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

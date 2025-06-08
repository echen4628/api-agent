import json
from langchain_core.messages import ToolMessage
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.base import StructuredTool

from langchain_openai import ChatOpenAI
from utils.constants import (GPT_4o,
                             TEXT_EMBEDDING_3_LARGE,
                             INPUTS_DESC_VECTOR_STORE_PATH,
                             OUTPUT_DESC_VECTOR_STORE_PATH,
                             FUNC_DESC_VECTOR_STORE_PATH,
                             NAME_TO_FUNCTION_JSON_PATH,
                             PLANNING_AGENT_PROMPT,
                             PLANNING,
                             EXECUTE)
from langchain_core.tools import tool, InjectedToolArg

from utils.FunctionTools import FunctionDatabase

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from planning.steps import Step

from utils.execute_utils2 import execute_subgraph
from agent.state import State
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    '''
    Adds two intergers a and b
    '''
    return a+b

from pydantic import BaseModel, Field
from typing import Literal, Dict, Union, List



functionDatabase = FunctionDatabase(TEXT_EMBEDDING_3_LARGE,
                                    INPUTS_DESC_VECTOR_STORE_PATH,
                                    OUTPUT_DESC_VECTOR_STORE_PATH,
                                    FUNC_DESC_VECTOR_STORE_PATH,
                                    NAME_TO_FUNCTION_JSON_PATH )

finished_plan = False



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
        # check if the extract makes sense
        if plan_step["action"] == "answer_question" and plan_step["args"] == {}:
            return f"Failed to add step. This answer_question step does not contain any arguments for the agent answering the question. Please fill the 'args' fields with all the necessary variables to answer the question.", steps
        elif plan_step["action"] == "call":
            import pdb; pdb.set_trace()
            function_inputs = functionDatabase.name_to_function[plan_step["tool"]].parameter_leaves

            if len(plan_step["args"]) < len(function_inputs.keys()):
                return f"Failed to add step. This function call is missing at least one arguments. Expected arguments with the following description: {function_inputs}. Received {plan_step['args'].keys()}."
            elif len(plan_step["args"]) > len(function_inputs.keys()):
                return f"Failed to add step. This function call has too many arguments. Expected arguments with the following description: {function_inputs}. Received {plan_step['args'].keys()}."
        steps.append(plan_step)
        return f"Successfully added step. Now, the plan has {len(steps)} steps. The latest few steps are:\n{extract_last_k_steps(3, steps)}", steps
    except Exception as e:
        import pdb; pdb.set_trace()
        return str(e), steps

@tool
def finish_plan() ->str:
    '''
    Notify the user the plan has been finished
    '''
    return "plan finished"

tools = [StructuredTool.from_function(functionDatabase.search),
         StructuredTool.from_function(functionDatabase.find_dependency),
         StructuredTool.from_function(functionDatabase.search_function_outputs),
         add_step_to_plan,
         finish_plan]

memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)
llm_with_tools = llm.bind_tools(tools)
with open(PLANNING_AGENT_PROMPT, "r", errors="ignore") as f:
    planning_prompt = f.read()

prompt_template = ChatPromptTemplate([
    ("system", planning_prompt),
    MessagesPlaceholder("msgs")
])

react_llm = prompt_template | llm_with_tools


def chatbot(state: State):
    # import pdb; pdb.set_trace()
    return {"messages": [react_llm.invoke({"msgs": state["messages"]})]}

def chatbot_decide_next(state: State):
    # import pdb; pdb.set_trace()
    ai_message = state["messages"][-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "call_tools"
    return END


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        print('there was a tool call')
        for tool_call in message.tool_calls:
            try:
                if tool_call["name"] == "add_step_to_plan":
                    tool_call["args"]["steps"] = state.get("plan", [])
                    tool_result, steps = self.tools_by_name[tool_call["name"]].invoke(
                        tool_call["args"]
                    )
                    state["plan"] = steps
                elif tool_call["name"] == "finish_plan":
                    tool_result = self.tools_by_name[tool_call["name"]].invoke(
                        tool_call["args"]
                    )
                    state['mode'] = EXECUTE
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
            except:
                print("ERROR!")
                import pdb; pdb.set_trace()
        return {"messages": outputs, "plan": state["plan"], 'mode': state['mode']}

# def decide_start(state: State):
#     # check if the message is a user message or a tool response
#     #
#     # if it is a user message, call the chatbot
#     # else call the the api chatbot
#     pass
def initialization(state: State):
    # import pdb; pdb.set_trace()
    plan = state.get("plan", [])
    if state.get("mode") != PLANNING and state.get("mode") != EXECUTE:
        mode = PLANNING
    else:
        mode = state.get("mode", PLANNING)
    plan_idx = state.get("plan_idx", 0)
    results_cache = state.get("results_cache", {})

    return {"plan": plan,
            "mode": mode,
            "plan_idx": plan_idx,
            "results_cache": results_cache,
            "failure": "",
            "retry_count": 0}

def tools_decide_next(state: State):
    # import pdb; pdb.set_trace()
    if state["mode"] == EXECUTE:
        return "execute_init"
    elif state["mode"] == PLANNING:
        return "chatbot"
    else:
        raise ValueError

def execute_init(state: State):
    # import pdb; pdb.set_trace()
    print("call the other graph here")
    # user_input = {"plan": state["plan"],
    #               "mode": EXECUTE,
    #               "plan_idx": state["plan_idx"],
    #               "results_cache": {},
    #             "messages": [{"role": "user", "content": ""}]}
    execute_output = execute_subgraph.invoke(state)
    return execute_output
    # state["plan_idx"] = execute_output["plan_idx"]
    # import pdb; pdb.set_trace()
    # return {"plan": execute_output["plan"],
    #         "mode": EXECUTE,
    #         "plan_idx": state["plan_idx"]}

    return None


def planning_execution_router(state: State):
    # import pdb; pdb.set_trace()
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        return "execute_init"
    else:
        return "chatbot"
graph_builder = StateGraph(State)

# nodes
graph_builder.add_node("initialization", initialization)
graph_builder.add_node("chatbot", chatbot)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("call_tools", tool_node)
# graph_builder.add_node("execute_call", execute_call)
# graph_builder.add_node("execute_ask_user", execute_ask_user)
# graph_builder.add_node("execute_extract", execute_extract)
graph_builder.add_node("execute_init", execute_init)

# edges
graph_builder.add_edge(START, "initialization")
graph_builder.add_conditional_edges("initialization", planning_execution_router)
# graph_builder.add_edge("initialization", "chatbot")
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_conditional_edges(START, planning_execution_router)
graph_builder.add_conditional_edges("chatbot", chatbot_decide_next)
graph_builder.add_conditional_edges("call_tools", tools_decide_next)
graph_builder.add_edge("execute_init", END)
# graph_builder.add_edge("call_tools", "chatbot")

# compile with memory for messages
graph = graph_builder.compile(checkpointer=memory)


# state: messages, mode=planning or execution
# when u return from planning subgraph, u should get a state obj, grab the last message
# execute_subgraph = invoke 

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    # def stream_graph_updates(user_input: str):
    #     for events in graph.stream({"messages": [{"role": "user", "content": user_input}]},
    #                                config,
    #                                stream_mode="updates"):
    #         for value in events.values():
    #             print("Assistant:", value["messages"][-1].content)

    def stream_graph_updates(user_input: State):
        for events in graph.stream(user_input,
                                   config,
                                   stream_mode="updates"):
            for value in events.values():
                if "messages" in value:
                    print("Assistant:", value["messages"][-1].content)

    # plan = []
    
    messages = [{'type': 'ai', 'content': 'execute_call', 'tool_calls': 
                 [{'name': 'Search_Car_Location', 
                   'args': {'query': 'San Diego Marriott La Jolla'}, 'id': '${1}', 'type': 'tool_call'}]},
        {'tool_call_id': '${1}', 'role': 'tool', 'name': 'Search_Car_Location', 'content': '{"status": true, "message": "Success", "data": [{"city": "San Diego", "coordinates": {"longitude": -117.215935, "latitude": 32.873055}, "country": "United States", "name": "San Diego Marriott La Jolla"}]}'}]
    # messages.append({"role": "user", "content": query})
    plan = [{'id': 1, 'var': '${1}', 'action': 'call', 'tool': 'Search_Car_Location', 'args': {'query': 'San Diego Marriott La Jolla'}},
    {'id': 2, 'var': '${2}', 'action': 'extract',
        'source': '${1}', 'path': 'coordinates.latitude'},
    {'id': 3, 'var': '${3}', 'action': 'extract',
        'source': '${1}', 'path': 'coordinates.longitude'},
    {'id': 4, 'var': '${4}', 'action': 'call', 'tool': 'Search_Car_Rentals', 'args': {'pick_up_date': '2024-10-14', 'pick_up_time': '08:00', 'drop_off_date': '2024-10-15',
                                                                                        'drop_off_time': '08:00', 'pick_up_latitude': '${2}', 'pick_up_longitude': '${3}', 'drop_off_latitude': '${2}', 'drop_off_longitude': '${3}'}},
    {'id': 5, 'var': '${5}', 'action': 'call', 'tool': 'Search_Car_Rentals', 'args': {'pick_up_date': '2024-10-15', 'pick_up_time': '08:00', 'drop_off_date': '2024-10-16', 'drop_off_time': '08:00', 'pick_up_latitude': '${2}', 'pick_up_longitude': '${3}', 'drop_off_latitude': '${2}', 'drop_off_longitude': '${3}'}}]

    mode="execute"
    plan_idx = 0
    results_cache={}
    user_input: State = {"plan": plan,
                        "mode": mode,
                        "plan_idx": plan_idx,
                        "results_cache": results_cache,
                        "messages": messages}
    # user_input: State = {"messages": [{"role": "user", "content": "Today is October 13th, 2024. I want to rent a car for a day at the San Diego Marriott La Jolla. Could you compare the price differences for picking up the car at 8 AM tomorrow and the day after tomorrow at the same place for a 24-hour rental?"}]}
    stream_graph_updates(user_input)


    # while True:
    #     user_input_message = input("User: ")
    #     if user_input_message.lower() in ["quit", "exit", "q"]:
    #         print("Goodbye!")
    #         break
    #     user_input: State = {"messages": [{"role": "user", "content": user_input_message}]}

    #     stream_graph_updates(user_input)

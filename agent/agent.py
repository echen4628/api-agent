import json
from langchain_core.messages import ToolMessage, AIMessage
from typing import Annotated, Tuple

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
def add_step_to_plan(plan_step, steps) ->Tuple[str, List]:
    try:
        # check if the extract makes sense
        if plan_step["action"] == "answer_question" and plan_step["args"] == {}:
            return f"Failed to add step. This answer_question step does not contain any arguments for the agent answering the question. Please fill the 'args' fields with all the necessary variables to answer the question.", steps
        elif plan_step["action"] == "call":
            function_inputs = functionDatabase.name_to_function[plan_step["tool"]].parameter_leaves

            if len(plan_step["args"]) < len(function_inputs.keys()):
                return f"Failed to add step. This function call is missing at least one argument. Expected arguments with the following description: {function_inputs}. Received {plan_step['args'].keys()}.", steps
            elif len(plan_step["args"]) > len(function_inputs.keys()):
                return f"Failed to add step. This function call has too many arguments. Expected arguments with the following description: {function_inputs}. Received {plan_step['args'].keys()}.", steps
        steps.append(plan_step)
        return f"Successfully added step. Now, the plan has {len(steps)} steps. The latest few steps are:\n{extract_last_k_steps(3, steps)}", steps
    except Exception as e:
        return str(e), steps
    
class AddMultipleStepsToPlan(BaseModel):
    """
    Tool that adds a subsequent PlanStep object to the linear execution plan.
    """

    plan_steps: List[Step] = Field(..., description="A series of next steps in order.")
    steps: Annotated[List, InjectedToolArg] = Field(..., description="The overall plan")

@tool(args_schema=AddMultipleStepsToPlan)
def add_multiple_steps_to_plan(plan_steps, steps) -> Tuple[str, List]:
    completed = 0
    existing_variables = set()
    api_response_variables = set()
    for existing_step in steps:
        existing_variables.add(existing_step["var"])
        if existing_step["action"] == "call":
            api_response_variables.add(existing_step["var"])
    for plan_step in plan_steps:
        try:
            if plan_step["action"] == "answer_question" and plan_step["args"] == {}:
                return f"Failed to add step: {plan_step}, because this `answer_question` step does not contain any arguments for the agent answering the question. Please fill the 'args' fields with all the necessary variables to answer the question.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
            elif plan_step["action"] == "call":
                function_inputs = functionDatabase.name_to_function[plan_step["tool"]].parameter_leaves
                if len(plan_step["args"]) < len(function_inputs.keys()):
                    return f"Failed to add step: {plan_step}, because this function call is missing at least one argument. Expected arguments with the following description: {function_inputs}.  Received {plan_step['args'].keys()}.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
                elif len(plan_step["args"]) > len(function_inputs.keys()):
                    return f"Failed to add step: {plan_step}, because this function call has too many arguments. Expected arguments with the following description: {function_inputs}.  Received {plan_step['args'].keys()}.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
                errors = []
                for arg in plan_step["args"].values(): 
                    if "${" in arg and arg not in existing_variables:
                        errors.append(arg)
                if errors:
                    return f"Failed to add step: {plan_step}, because its argument names do not exist: {errors}. Please make sure to extract from the following existing ones first: {existing_variables}.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
            elif plan_step["action"] == "extract":
                if plan_step["source"] not in api_response_variables:
                    return f"Failed to add step: {plan_step}, because its source ({plan_step['source']}) is not the output of a call step. Please make sure to extract from the relevant call step var: {api_response_variables}.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
            elif plan_step["action"] == "answer_question":
                errors = []
                for arg in plan_step["args"].values(): 
                    if "${" in arg and arg not in existing_variables:
                        errors.append(arg)
                if errors:
                    return f"Failed to add step: {plan_step}, because its argument names do not exist: {errors}. Please make sure to extract from the following existing ones first: {existing_variables}.\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
            steps.append(plan_step)
            existing_variables.add(plan_step["var"])
            if plan_step["action"] == "call":
                api_response_variables.add(plan_step["var"])
            completed += 1
        except Exception as e:
            return str(e)+f"\n\nThe latest successfully added steps are:\n{extract_last_k_steps(3, steps)}.", steps
    return f"Successfully added all steps. Now, the plan has {len(steps)} steps. The latest few steps are:\n{extract_last_k_steps(3, steps)}", steps



class FinishPlan(BaseModel):
    '''
    Notify the user the plan has been finished.
    '''

    steps: Annotated[List, InjectedToolArg] = Field(..., description="The overall plan")

@tool(args_schema=FinishPlan)
def finish_plan(steps) -> Tuple[str, str]:
    if len(steps) == 0:
        return "The plan is empty and does not end with `answer_question`. Please complete the plan and then try again.", PLANNING
    elif steps and steps[-1]["action"] == "answer_question":
        return "successfully, finished plan", EXECUTE
    else:
        return "Does not end the plan with `answer_question`. Please complete the plan with that step and try again.", PLANNING


tools = [StructuredTool.from_function(functionDatabase.search),
         StructuredTool.from_function(functionDatabase.find_dependency),
         StructuredTool.from_function(functionDatabase.search_function_outputs),
        add_multiple_steps_to_plan,
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
    return {"messages": [react_llm.invoke({"msgs": state["messages"]})]}

def chatbot_decide_next(state: State):
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
                if tool_call["name"] == "add_step_to_plan" or tool_call["name"] == "add_multiple_steps_to_plan":
                    tool_call["args"]["steps"] = state.get("plan", [])
                    tool_result, steps = self.tools_by_name[tool_call["name"]].invoke(
                        tool_call["args"]
                    )
                    state["plan"] = steps
                elif tool_call["name"] == "finish_plan":
                    tool_call["args"]["steps"] = state.get("plan", [])
                    tool_result, state['mode'] = self.tools_by_name[tool_call["name"]].invoke(
                        tool_call["args"]
                    )
                    # state['mode'] = EXECUTE
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
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(str(e)),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                print(f"Got an error during tool call {tool_call['name']}: {e}")
        return {"messages": outputs, "plan": state["plan"], 'mode': state['mode']}

# def decide_start(state: State):
#     # check if the message is a user message or a tool response
#     #
#     # if it is a user message, call the chatbot
#     # else call the the api chatbot
#     pass
def initialization(state: State):
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
    if state["mode"] == EXECUTE:
        return "execute_init"
    elif state["mode"] == PLANNING:
        return "chatbot"
    else:
        raise ValueError

def execute_init(state: State):
    print("call the other graph here")
    # user_input = {"plan": state["plan"],
    #               "mode": EXECUTE,
    #               "plan_idx": state["plan_idx"],
    #               "results_cache": {},
    #             "messages": [{"role": "user", "content": ""}]}
    last_message = state["messages"][-1]
    # import pdb; pdb.set_trace()
    if isinstance(last_message, ToolMessage) and last_message.name == "finish_plan":
        state["messages"].append(AIMessage(content="starting execution of plan."))
    execute_output = execute_subgraph.invoke(state)
    return execute_output


def planning_execution_router(state: State):
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
graph_builder.add_node("execute_init", execute_init)

# edges
graph_builder.add_edge(START, "initialization")
graph_builder.add_conditional_edges("initialization", planning_execution_router)
graph_builder.add_conditional_edges("chatbot", chatbot_decide_next)
graph_builder.add_conditional_edges("call_tools", tools_decide_next)
graph_builder.add_edge("execute_init", END)

graph = graph_builder.compile(checkpointer=memory)


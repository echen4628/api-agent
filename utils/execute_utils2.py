from utils.parse_functions import (parse_output_to_basemodel,
                                   extract_from_basemodel,
                                   extract_from_dict)
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from pydantic_core import from_json

import json
from planning.steps import (APIStep,
                            ExtractStep,
                            AskUserStep,
                            AnswerQuestionStep,
                            Step)

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnablePassthrough


from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from utils.constants import (GPT_4o,
                             TEXT_EMBEDDING_3_LARGE,
                             INPUTS_DESC_VECTOR_STORE_PATH,
                             OUTPUT_DESC_VECTOR_STORE_PATH,
                             FUNC_DESC_VECTOR_STORE_PATH,
                             NAME_TO_FUNCTION_JSON_PATH,
                             PLANNING_AGENT_PROMPT,
                             PLANNING,
                             EXECUTE,
                             EXTRACT_RETRY_AGENT_SYSTEM_PROMPT,
                             EXTRACT_RETRY_AGENT_USER_PROMPT,
                             ANSWER_QUESTION_SYSTEM_PROMPT,
                             ANSWER_QUESTION_USER_PROMPT)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from execution.dummy_functions import name_to_functions
from agent.state import State
from itertools import product
from dotenv import load_dotenv



tools = []
load_dotenv()

memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)
llm_with_tools = llm.bind_tools(tools)


def execute_call_by_llm(state: State):
    return state
    # return {"messages": [execute_call_prompt_llm.invoke({"msgs": state["messages"]})]}


def execute_call_by_llm_decide_next(state: State):
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
        has_failures = False
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
                elif tool_call["name"] in name_to_functions:
                    tool_result_as_model: BaseModel = name_to_functions[tool_call["name"]](
                        **tool_call["args"])
                    state['results_cache'][tool_call["id"]] = tool_result_as_model
                    tool_result = tool_result_as_model.model_dump()
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
                has_failures = True
                outputs.append(
                    ToolMessage(
                        content=json.dumps(e),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            # if not has_failures:
            #     state['plan_idx'] += 1

        return {"messages": outputs, "plan": state["plan"], 'mode': state['mode'],
                'plan_idx': state['plan_idx'], "results_cache": state["results_cache"]}



def tools_decide_next(state: State):
    if state["mode"] == EXECUTE:
        return "execute_init"
    elif state["mode"] == PLANNING:
        return "chatbot"
    else:
        raise ValueError


def execution_router(state: State):
    last_message = state["messages"][-1]
    # import pdb; pdb.set_trace()
    # First check if the execution is in progress already
    if isinstance(last_message, ToolMessage):
        if "error" in last_message.content:
            return "execute_call_by_llm" # use the llm to resolve the error
        elif '"plan finished"' != last_message.content: # this is an internal message not a tool response part of execution
            return "handle_tool_responses"

    plan_idx = state['plan_idx']
    if plan_idx >= len(state['plan']):
        print("got to end!!!!")
        return END
    elif state['plan'][plan_idx]['action'] == 'call':
        needs_llm = False
        for arg in state['plan'][plan_idx]['args'].values():
            if isinstance(arg, str) and '$' == arg[0] and "@d" in state["results_cache"][arg][1].split(".")[0]:
                needs_llm = True
                break
        if needs_llm:
            return "execute_call_use_llm"
        print('going to execute_call')
        return "execute_call"
    elif state['plan'][plan_idx]['action'] == 'ask_user':
        return "execute_ask_user"
    elif state['plan'][plan_idx]['action'] == 'extract':
        return "execute_extract"
    elif state['plan'][plan_idx]['action'] == 'answer_question':
        return "execute_answer_question"
    # don't forget to increment

with open(ANSWER_QUESTION_SYSTEM_PROMPT, "r") as f:
    answer_question_system_template = f.read()

with open(ANSWER_QUESTION_USER_PROMPT, "r") as f:
    answer_question_user_template = f.read()

answer_question_prompt_template = ChatPromptTemplate([
    ("system", answer_question_system_template),
    ("user", answer_question_user_template)
])
answer_question_llm_chain =  answer_question_prompt_template | llm


def execute_answer_question(state: State):
    plan_idx = state['plan_idx']
    step: AnswerQuestionStep = state['plan'][plan_idx]  # type: ignore
    all_args = {}
    for arg_name in step["args"]:
        arg = step["args"][arg_name]
        if isinstance(arg, str) and '$' == arg[0]:
            print("here")
            all_args[arg_name] = (state["results_cache"][arg][1], state["results_cache"][arg][0])
        else:
            all_args[arg_name] = (state["results_cache"][arg][1], arg)

    
    response = answer_question_llm_chain.invoke({"query": step["query"],
                "strategy": step["strategy"],
                "information": str(all_args)})
    state['plan_idx'] += 1
    return {"plan_idx": state['plan_idx'], "messages": [response]}


def execute_call(state: State):
    plan_idx = state['plan_idx']
    step: APIStep = state['plan'][plan_idx]  # type: ignore

    all_args = []
    for arg_name in step["args"]:
        arg = step["args"][arg_name]
        if isinstance(arg, str) and '$' == arg[0]:
            if "@d" not in state["results_cache"][arg][1]:
                all_args.append([state["results_cache"][arg][0]])
            else:
                all_args.append(state["results_cache"][arg][0])
        else:
            all_args.append([arg])
    all_args_product = product(*all_args)
    tool_calls = []
    for i, all_arg in enumerate(all_args_product):
        args_dict = {}
        for arg, arg_name in zip(all_arg,  step["args"]):
            args_dict[arg_name] = arg
        if i == 0:
            tool_id = step["var"]
        else:
            tool_id = step["var"] + "_"+str(i)
        tool_call = ToolCall(name=step['tool'],
                        args=args_dict,
                        id=tool_id)
        tool_calls.append(tool_call)

    return {"messages": [AIMessage(content="execute_call", tool_calls=tool_calls)]}



def execute_ask_user(state: State):
    return


def execute_extract(state: State):
    plan_idx = state['plan_idx']
    step: ExtractStep = state['plan'][plan_idx]  # type: ignore

    try:
        # if isinstance(state["results_cache"][step['source']][0], dict):
        print(f"extract from a dictionary, {step['path']}")
        state["results_cache"][step['var']] = extract_from_dict(
        state["results_cache"][step['source']][0], step['path'])
        # else:
        #     print(f"extract from a base model: {step['path']}")
        #     state["results_cache"][step['var']] = extract_from_basemodel(
        #     state["results_cache"][step['source']][0], step['path'])
        state['plan_idx'] += 1
        state['retry_count'] = 0
    except Exception as e:
        state['failure'] = str(e)
    return state

def execute_extract_decide_next(state: State):
    if state["failure"]:
        return "handle_extraction_errors"
    else:
        return "execute_init"




def execute_init(state: State):
    return state

from utils.FunctionTools import FunctionDatabase
functionDatabase = FunctionDatabase(TEXT_EMBEDDING_3_LARGE,
                                    INPUTS_DESC_VECTOR_STORE_PATH,
                                    OUTPUT_DESC_VECTOR_STORE_PATH,
                                    FUNC_DESC_VECTOR_STORE_PATH,
                                    NAME_TO_FUNCTION_JSON_PATH )

def handle_tool_responses(state: State):
    step = state["plan"][state["plan_idx"]]
    assert step['action'] == "call"
    starting_tool_message_id = len(state["messages"])-1
    has_list = False
    temp_outputs = []
    while "_" in state["messages"][starting_tool_message_id].tool_call_id:
        starting_tool_message_id -= 1
        has_list = True
    function_name = step['tool']

    for i in range(starting_tool_message_id, len(state["messages"])):
        tool_message = state["messages"][i]
        tool_message_content = from_json(tool_message.content)
        output_repr = parse_output_to_basemodel({"data": tool_message_content["data"]}, f"{function_name}_output")
        if output_repr:
            temp_outputs.append(output_repr.model_validate({"data": tool_message_content["data"]}).model_dump())
            
        else:
            raise Exception(f"Couldn't create output_repr from response of this step:{step}")
        
    tool_call_id = state["messages"][starting_tool_message_id].tool_call_id
    if has_list:
        state['results_cache'][tool_call_id] = [temp_outputs, tool_call_id+"@d"]
    else:
        state['results_cache'][tool_call_id] = [temp_outputs[0], tool_call_id]
    state["messages"].append(AIMessage(content=f"Successfully called and processed call (ID: {tool_call_id})."))

    
    state["plan_idx"] += 1


    return state


extract_step_llm = llm.with_structured_output(ExtractStep)
with open(EXTRACT_RETRY_AGENT_SYSTEM_PROMPT, "r") as f:
    extract_retry_agent_system_template = f.read()

with open(EXTRACT_RETRY_AGENT_USER_PROMPT, "r") as f:
    extract_retry_user_system_template = f.read()

extract_retry_prompt_template = ChatPromptTemplate([
    ("system", extract_retry_agent_system_template),
    ("user", extract_retry_user_system_template)
])
extract_step_llm_chain =  extract_retry_prompt_template | extract_step_llm
def handle_extraction_errors(state: State):
    if state["retry_count"] >= 3:
        raise Exception(f"Trying to address error {state['failure']} but reached retry maximum of {3}.")
    state["retry_count"] += 1
    plan_idx = state["plan_idx"]
    step: ExtractStep = state['plan'][plan_idx]  # type: ignore
    state['plan'][plan_idx] = extract_step_llm_chain.invoke({"current_extract_step": step,
                                   "error_message": state["failure"],
                                   "overall_plan": state["plan"]})
    
    state["failure"] = ""
    return state

graph_builder = StateGraph(State)

# nodes

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("call_tools", tool_node)
graph_builder.add_node("execute_call_by_llm", execute_call_by_llm)
graph_builder.add_node("execute_call", execute_call)
graph_builder.add_node("execute_ask_user", execute_ask_user)
graph_builder.add_node("execute_extract", execute_extract)
graph_builder.add_node("execute_init", execute_init)
graph_builder.add_node("execute_answer_question", execute_answer_question)
graph_builder.add_node("handle_tool_responses", handle_tool_responses)
graph_builder.add_node("handle_extraction_errors", handle_extraction_errors)

# edges
graph_builder.add_conditional_edges(START, execution_router)
graph_builder.add_conditional_edges("handle_tool_responses", execution_router)
graph_builder.add_conditional_edges("execute_init", execution_router)
graph_builder.add_conditional_edges(
    "execute_call_by_llm", execute_call_by_llm_decide_next)
graph_builder.add_edge("execute_call", END)
graph_builder.add_conditional_edges("call_tools", tools_decide_next)
graph_builder.add_conditional_edges("execute_extract", execute_extract_decide_next)
graph_builder.add_conditional_edges("execute_answer_question", execution_router)
graph_builder.add_edge("handle_extraction_errors", "execute_init")

# TODO: execute_ask_user prob needs to be changed
graph_builder.add_edge("execute_ask_user", "execute_init")

# compile with memory for messages
execute_subgraph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    def stream_graph_updates(user_input: State):
        for events in execute_subgraph.stream(user_input,
                                   config,
                                   stream_mode="updates"):
            for value in events.values():
                print("Assistant:", value["messages"][-1].content)


    user_input_message = input("User: ")
    plan = [{'id': 1, 'var': '${1}', 'action': 'call', 'tool': 'Search_Car_Location', 'args': {'query': 'San Diego Marriott La Jolla'}},
        {'id': 2, 'var': '${2}', 'action': 'extract',
            'source': '${1}', 'path': 'coordinates.latitude'},
        {'id': 3, 'var': '${3}', 'action': 'extract',
            'source': '${1}', 'path': 'coordinates.longitude'},
        {'id': 4, 'var': '${4}', 'action': 'call', 'tool': 'Search_Car_Rentals', 'args': {'pick_up_date': '2024-10-14', 'pick_up_time': '08:00', 'drop_off_date': '2024-10-15',
                                                                                          'drop_off_time': '08:00', 'pick_up_latitude': '${2}', 'pick_up_longitude': '${3}', 'drop_off_latitude': '${2}', 'drop_off_longitude': '${3}'}},
        {'id': 5, 'var': '${5}', 'action': 'call', 'tool': 'Search_Car_Rentals', 'args': {'pick_up_date': '2024-10-15', 'pick_up_time': '08:00', 'drop_off_date': '2024-10-16', 'drop_off_time': '08:00', 'pick_up_latitude': '${2}', 'pick_up_longitude': '${3}', 'drop_off_latitude': '${2}', 'drop_off_longitude': '${3}'}}]
    user_input: State = { "plan": plan,
                  "mode": EXECUTE,
                  "plan_idx": 0,
                  "results_cache": {},
                    "messages": [{"role": "user", "content": user_input_message}]}
    # stream_graph_updates(user_input)
    final_message = execute_subgraph.invoke(user_input,
                                   config)
    


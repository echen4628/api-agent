from utils.parse_functions import (parse_output_to_basemodel,
                                   extract_from_basemodel)
from langgraph.graph.message import add_messages
from pydantic import BaseModel

import json
from planning.steps import (APIStep,
                            ExtractStep,
                            AskUserStep,
                            Step)

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages.tool import ToolCall


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
                             EXECUTE)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from execution.dummy_functions import name_to_functions
from agent.state import State


# class CacheEntry:
#     def __init__(self, signature, data, parent_id):
#         self.signature = signature
#         self.data = data
#         self.parent_id = parent_id


# class ExecutorState(TypedDict):
#     plan: List[Step]
#     messages: Annotated[list, add_messages]
#     mode: str
#     plan_idx: int
#     results_cache: Dict
#     # results_cache: Dict[str, CacheEntry]


tools = []

memory = MemorySaver()
llm = ChatOpenAI(model=GPT_4o)
llm_with_tools = llm.bind_tools(tools)

# TODO: Fix prompt
with open(PLANNING_AGENT_PROMPT, "r") as f:
    execute_prompt = f.read()

prompt_template = ChatPromptTemplate([
    ("system", execute_prompt),
    MessagesPlaceholder("msgs")
])

execute_call_prompt_llm = prompt_template | llm_with_tools


def execute_call_by_llm(state: State):
    return {"messages": [execute_call_prompt_llm.invoke({"msgs": state["messages"]})]}


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
        # import pdb
        # pdb.set_trace()
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
                    # import pdb
                    # pdb.set_trace()
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
            if not has_failures:
                state['plan_idx'] += 1

        return {"messages": outputs, "plan": state["plan"], 'mode': state['mode'],
                'plan_idx': state['plan_idx'], "results_cache": state["results_cache"]}



def initialization(state: State):
    # state["plan"] = plan
    # state["mode"] = EXECUTE
    # state["plan_idx"] = 0
    # state["results_cache"] = {}

    return state


def tools_decide_next(state: State):
    if state["mode"] == EXECUTE:
        # import pdb
        # pdb.set_trace()
        return "execute_init"
    elif state["mode"] == PLANNING:
        return "chatbot"
    else:
        raise ValueError


def execution_router(state: State):
    # if the step is awf
    #   run the code and then store the result in a cache
    # elif the step is ask_user
    #   call an llm, to ask user for the question, then return the exact string used for that value
    # elif the step is extract
    #   run the code to extract the value
    # -> router first
    # router -> a step when the step is done (how to pass the current step? there needs to be another state here) -> router
    # if there are no more steps -> router is done
    # if there is an error, router will call a tool to swap
    # otherwise, the result will be returned to the main agent
    # maybe the planning agent can be planning and answering
    # import pdb
    # pdb.set_trace()
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
        return "execute_call"
    elif state['plan'][plan_idx]['action'] == 'ask_user':
        return "execute_ask_user"
    elif state['plan'][plan_idx]['action'] == 'extract':
        return "execute_extract"
    # don't forget to increment


def execute_call(state: State):
    # import pdb
    # pdb.set_trace()
    plan_idx = state['plan_idx']
    step: APIStep = state['plan'][plan_idx]  # type: ignore

    for arg_name in step["args"]:
        arg = step["args"][arg_name]
        if isinstance(arg, str) and '$' == arg[0] and "@d" not in state["results_cache"][arg][1].split(".")[0]:
            step["args"][arg_name] = state["results_cache"][arg][0]


    tool_call = ToolCall(name=step['tool'],
                         args=step["args"],
                         id=step["var"])
    return {"messages": [AIMessage(content="execute_call", tool_calls=[tool_call])]}

# class APIStep(TypedDict):
#     id: int
#     var: str
#     action: Literal['call']  # 'call', 'extract', 'ask_user'
#     tool: str
#     args: Dict[str, Union[str, float, int, bool]]


def execute_ask_user(state: State):
    return


def execute_extract(state: State):
    plan_idx = state['plan_idx']
    step: ExtractStep = state['plan'][plan_idx]  # type: ignore
    # import pdb
    # pdb.set_trace()
    state["results_cache"][step['var']] = extract_from_basemodel(
        state["results_cache"][step['source']], step['path'])
    state['plan_idx'] += 1
    return state


def execute_init(state: State):
    return state


graph_builder = StateGraph(State)

# nodes

graph_builder.add_node("initialization", initialization)

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("call_tools", tool_node)
graph_builder.add_node("execute_call_by_llm", execute_call_by_llm)
graph_builder.add_node("execute_call", execute_call)
graph_builder.add_node("execute_ask_user", execute_ask_user)
graph_builder.add_node("execute_extract", execute_extract)
graph_builder.add_node("execute_init", execute_init)

# edges
graph_builder.add_edge(START, "initialization")
graph_builder.add_edge("initialization", "execute_init")
graph_builder.add_conditional_edges("execute_init", execution_router)
graph_builder.add_conditional_edges(
    "execute_call_by_llm", execute_call_by_llm_decide_next)
graph_builder.add_edge("execute_call", "call_tools")
graph_builder.add_conditional_edges("call_tools", tools_decide_next)
graph_builder.add_edge("execute_extract", "execute_init")

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

    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() in ["quit", "exit", "q"]:
    #         print("Goodbye!")
    #         break
    #     stream_graph_updates(user_input)
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
    stream_graph_updates(user_input)

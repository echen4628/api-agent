from planning.steps import Step
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

def overwrite_if_nonempty(old, new):
    if new:
        return new
    return old

def int_overwrite_if_nonempty(old, new):
    if new or new == 0:
        return new
    return old

class State(TypedDict):
    plan: Annotated[List[Step], overwrite_if_nonempty]
    messages: Annotated[list, add_messages]
    mode: Annotated[str, overwrite_if_nonempty]
    plan_idx: Annotated[int, int_overwrite_if_nonempty]
    results_cache: Annotated[Dict, overwrite_if_nonempty]
    failure: str
    retry_count: Annotated[int, int_overwrite_if_nonempty]

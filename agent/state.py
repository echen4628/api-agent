from planning.steps import Step
from typing import List, Annotated, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    plan: List[Step]
    messages: Annotated[list, add_messages]
    mode: str
    plan_idx: int
    results_cache: Dict
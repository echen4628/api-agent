from typing import Literal, Dict, Union
from typing_extensions import TypedDict
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

from dataclasses import dataclass
from typing import Literal

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]
    
def node_1(state):
    print("---Node 1---")
    return {"name": state.name + " is ... "}

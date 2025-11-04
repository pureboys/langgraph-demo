from typing import Literal, TypedDict


class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]


import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END


def node_1(state):
    print("---Node 1---")
    return {"name": state["name"] + " is ... "}


def node_2(state):
    print("---Node 2---")
    return {"mood": "happy"}


def node_3(state):
    print("---Node 3---")
    return {"mood": "sad"}


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


# Build graph
builder = StateGraph(TypedDictState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

graph.invoke({"name": "Lance"})

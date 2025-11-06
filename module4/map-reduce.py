import common.customModel

llm = common.customModel.volcengine_doubao()

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}. 

Please return the response in JSON format with a 'subjects' array containing the sub-topics."""
joke_prompt = """Generate a joke about {subject}. 
Please return the response in JSON format with a 'joke' field containing the joke text."""

best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! 
Please return the response in JSON format with an 'id' field containing the index of the best joke (starting from 0).
Jokes: \n\n  {jokes}"""


import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel


class Subjects(BaseModel):
    subjects: list[str]  # 笑话话题列表subjects，由LLM生成


class BestJoke(BaseModel):
    id: int


class OverallState(TypedDict):
    topic: str  # 用户输入的主题
    subjects: list
    jokes: Annotated[list, operator.add]  # 设置了reducer，处理合并
    best_selected_joke: str


def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = llm.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


from langgraph.constants import Send


def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


class JokeState(TypedDict):
    subject: str


class Joke(BaseModel):
    joke: str


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = llm.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = llm.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


from langgraph.graph import END, StateGraph, START

# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# Compile the graph
app = graph.compile()

for s in app.stream({"topic": "animals"}):
    print(s)

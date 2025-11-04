from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator
from typing import Annotated

import common.customModel

llm = common.customModel.volcengine_doubao()


class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults


def search_web(state):
    """Retrieve docs from web search"""
    # Search
    _question = state["question"]
    # mock data
    formatted_search_docs = f"""
    Nvidia's Q2 2024 earnings about 1.5 billion dollars.
    """
    return {"context": [formatted_search_docs]}


def search_wikipedia(state):
    # Search
    _question = state["question"]
    # mock data
    formatted_search_docs = f"""
    Nvidia's Q2 2024 earnings about 2.5 billion dollars.
    """
    return {"context": [formatted_search_docs]}


def generate_answer(state):
    """Node to answer a question"""
    # Get state
    context = state["context"]
    question = state["question"]
    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, context=context)
    # Answer
    answer = llm.invoke(
        [SystemMessage(content=answer_instructions)]
        + [HumanMessage(content=f"Answer the question.")]
    )
    # Append it to state
    return {"answer": answer}


# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

# import common.render
# common.render.renderGraph(graph)

from langgraph.errors import InvalidUpdateError

try:
    result = graph.invoke({"question": "How were Nvidia's Q2 2024 earnings"})
    print(result["answer"].content)
except InvalidUpdateError as e:
    print(f"An error occurred: {e}")

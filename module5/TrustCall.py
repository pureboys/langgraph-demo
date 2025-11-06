import sys
import os
from trustcall import create_extractor
from pydantic import BaseModel, Field, ConfigDict
from typing import List


class UserProfile(BaseModel):
    user_name: str = Field(description="The user's preferred name")
    interests: List[str] = Field(description="A list of the user's interests")


import common.customModel

model = common.customModel.volcengine_doubao()

# Create the extractor
trustcall_extractor = create_extractor(
    model, tools=[UserProfile], tool_choice="UserProfile"
)

# Instruction
system_msg = "Extract the user profile from the following conversation"
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Conversation
conversation = [
    HumanMessage(content="Hi, I'm Lance."),
    AIMessage(content="Nice to meet you, Lance."),
    HumanMessage(content="I really like biking around San Francisco."),
]

# Invoke the extractor
result = trustcall_extractor.invoke(
    {"messages": [SystemMessage(content=system_msg)] + conversation}
)

for m in result["messages"]:
    m.pretty_print()


schema = result["responses"]
print(f"schema:{schema}")
dump = schema[0].model_dump()
print(f"dump:{dump}")
response_metadata = result["response_metadata"]
print(f"response_metadata:{response_metadata}")

print("*" * 50, end="\n")

# Update the conversation
updated_conversation = [
    HumanMessage(content="Hi, I'm Lance."),
    AIMessage(content="Nice to meet you, Lance."),
    HumanMessage(content="I really like biking around San Francisco."),
    AIMessage(content="San Francisco is a great city! Where do you go after biking?"),
    HumanMessage(content="I really like to go to a bakery after biking."),
]

# Update the instruction
system_msg = f"""Update the memory (JSON doc) to incorporate new information from the following conversation"""

# Invoke the extractor with the updated instruction and existing profile with the corresponding tool name (UserProfile)
result = trustcall_extractor.invoke(
    {"messages": [SystemMessage(content=system_msg)] + updated_conversation},
    {"existing": {"UserProfile": schema[0].model_dump()}},
)

for m in result["messages"]:
    m.pretty_print()

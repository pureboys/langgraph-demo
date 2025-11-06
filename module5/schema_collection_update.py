from trustcall import create_extractor
from pydantic import BaseModel, Field

# Initialize the model
import common.customModel

model = common.customModel.volcengine_doubao()



class PersonalInfo(BaseModel):
    """Basic personal information about the user"""
    content: str = Field(
        description="Personal details like name, age, location, or identity"
    )


class Activity(BaseModel):
    """Activities, hobbies, or events the user participates in"""
    content: str = Field(
        description="Activities, hobbies, sports, or things the user did or enjoys doing"
    )


class TravelPlan(BaseModel):
    """Travel plans, destinations, or trip ideas"""
    content: str = Field(
        description="Travel plans, destinations, countries to visit, or trip ideas"
    )


# Create the extractor with multiple schema types
trustcall_extractor = create_extractor(
    model,
    tools=[PersonalInfo, Activity, TravelPlan],
    enable_inserts=True,
    enable_updates=True,
)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Instruction
instruction = """Extract ALL memories from the following conversation using the appropriate tools.

RULES:
- Use PersonalInfo tool for: name, age, location, personal identity
- Use Activity tool for: hobbies, sports, activities, things they did
- Use TravelPlan tool for: travel plans, destinations, trips

Extract EVERY piece of information as a separate memory. Use multiple tool calls if needed."""

# Conversation
conversation = [
    HumanMessage(content="Hi, I'm Lance."),
    AIMessage(content="Nice to meet you, Lance."),
    HumanMessage(content="This morning I had a nice bike ride in San Francisco."),
]

# Invoke the extractor
result = trustcall_extractor.invoke(
    {"messages": [SystemMessage(content=instruction)] + conversation}
)

# Messages contain the tool calls
for m in result["messages"]:
    m.pretty_print()

print("\n" + "=" * 50)
print("第一次提取的记忆:")
print("=" * 50)
if result["responses"]:
    for i, memory in enumerate(result["responses"]):
        print(f"\n记忆 {i+1} [{type(memory).__name__}]:")
        print(f"  {memory.content}")
else:
    print("没有提取到记忆")

# We'll save existing memories, giving them an ID, key (tool name), and value
# Note: Now we have multiple tool types (PersonalInfo, Activity, TravelPlan)
existing_memories = (
    [
        (str(i), type(memory).__name__, memory.model_dump())
        for i, memory in enumerate(result["responses"])
    ]
    if result["responses"]
    else None
)

print(f"\n现有记忆数量: {len(existing_memories) if existing_memories else 0}")

# Update the conversation
updated_conversation = [
    AIMessage(content="That's great, did you do after?"),
    HumanMessage(content="I went to Tartine and ate a croissant."),
    AIMessage(content="What else is on your mind?"),
    HumanMessage(content="I was thinking about my Japan, and going back this winter!"),
]

# Update the instruction
system_msg = """Based on the following conversation, create NEW memories for any new information mentioned.

IMPORTANT RULES:
- ONLY create new memories for NEW information in this conversation
- Use Activity tool for: dining at Tartine, eating croissant
- Use TravelPlan tool for: Japan travel plans
- DO NOT recreate or modify existing memories unless they need updating
- Each new piece of information should be a separate tool call

The existing memories will be preserved automatically."""

# Invoke the extractor with our updated conversation and existing memories
result = trustcall_extractor.invoke(
    {
        "messages": [SystemMessage(content=system_msg)] + updated_conversation,
        "existing": existing_memories,
    }
)

print("\n" + "=" * 50)
print("第二次提取的新记忆:")
print("=" * 50)
if result["responses"]:
    for i, memory in enumerate(result["responses"]):
        print(f"\n新记忆 {i+1} [{type(memory).__name__}]:")
        print(f"  {memory.content}")
else:
    print("没有新记忆")

# 合并所有记忆：现有的 + 新的
print("\n" + "=" * 50)
print("所有记忆的完整列表（现有 + 新增）:")
print("=" * 50)

all_memories = []
# 添加现有记忆
if existing_memories:
    for mem_id, tool_name, mem_data in existing_memories:
        all_memories.append((tool_name, mem_data['content']))

# 添加新记忆
if result["responses"]:
    for memory in result["responses"]:
        all_memories.append((type(memory).__name__, memory.content))

# 显示所有记忆
for i, (mem_type, content) in enumerate(all_memories):
    print(f"\n记忆 {i+1} [{mem_type}]:")
    print(f"  {content}")

print(f"\n总计: {len(all_memories)} 条记忆")

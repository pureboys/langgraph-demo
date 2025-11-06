import sys
import os
from pydantic import BaseModel, Field


class Memory(BaseModel):
    content: str = Field(
        description="The main content of the memory. For example: User expressed interest in learning about French."
    )


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")


from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the model
import common.customModel

model = common.customModel.volcengine_doubao()

# Bind schema to model
model_with_structure = model.with_structured_output(MemoryCollection)

# Create a system message to guide the model
system_message = SystemMessage(
    content="""You are a memory collection system. 
Your task is to extract and store memories about the user in a structured format.
Please output the memories in the following JSON format:
{
    "memories": [
        {
            "content": "memory content here"
        }
    ]
}"""
)

# Invoke the model to produce structured output that matches the schema
memory_collection = model_with_structure.invoke(
    [system_message, HumanMessage("My name is Lance. I like to bike.")]
)
memories = memory_collection.memories


print("memories:{}\n".format(memories))

model_dump = memory_collection.memories[0].model_dump()
print(f"model_dump:{model_dump}")

import uuid
from langgraph.store.memory import InMemoryStore
from typing import List, Dict

# Initialize the in-memory store
in_memory_store = InMemoryStore()

# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memories")


def batch_store_memories(memories: List[Memory], namespace: tuple) -> None:
    """
    批量存储记忆到内存存储中

    Args:
        memories: Memory对象列表
        namespace: 存储的命名空间
    """
    # 为每个记忆创建唯一的键并存储
    for memory in memories:
        key = str(uuid.uuid4())
        value = memory.model_dump()
        in_memory_store.put(namespace, key, value)


# 使用批量存储方法
batch_store_memories(memory_collection.memories, namespace_for_memory)

# 验证存储结果
# Search
for m in in_memory_store.search(namespace_for_memory):
    print(m.dict())

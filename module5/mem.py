import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()


user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())

# The value needs to be a dictionary  
value = {"food_preference" : "I like pizza"}

# Save the memory
in_memory_store.put(namespace_for_memory, key, value)

# Search 
memories = in_memory_store.search(namespace_for_memory)
print(type(memories))
# Metatdata 
print(memories[0].dict())
# The key, value
print(memories[0].key, memories[0].value)

# # 我们也可以使用get方法通过命名空间和键来取得对象。
# # Get the memory by namespace and key
memory = in_memory_store.get(namespace_for_memory, key)
print(memory.dict())
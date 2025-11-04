from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()


def volcengine_doubao():
    return ChatOpenAI(
        model=os.environ.get("MODEL"),
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("BASE_URL"),
    )

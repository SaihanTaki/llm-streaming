import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

def get_chat_llm(
    provider: Literal["openai", "google"]="google",
    temperature: int=0.1,
    max_token: int=1024
):
    if provider == "google":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_token,
            api_key=os.environ["GOOGLE_API_KEY"]
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_token,
            api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        raise ValueError("Provider should be 'openai' or 'google'")
        
    return llm

llm = get_chat_llm(provider="openai")

prompt = "Who are you?"
messages = [
    {"role":"system", "content": "You are a helpful AI chat Bot!"},
    {"role": "user", "content": prompt},
]

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
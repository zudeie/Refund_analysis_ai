from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import pandas as pd 
from dotenv import load_dotenv

load_dotenv()


agent = ChatOllama.from_llm(
    llm=ChatOllama(
        model_name="llama-2-13b-chat-hf",
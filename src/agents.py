from os import getenv
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.chains.openai_functions import create_extraction_chain_pydantic

load_dotenv()

class ResearchTopics(BaseModel):
    topics: List[str] = Field(..., description="List of research topics mentioned in the query")

LLM_MODEL_NAME=getenv('LLM_MODEL_NAME')
chat_llm = init_chat_model(LLM_MODEL_NAME)

system_prompt="Extract user research topics from user query. Be as succint as you can, provide at most 5 topics."
topic_research_agent = Agent(
    getenv('LLM_MODEL'), 
    system_prompt=system_prompt,
    output_type=ResearchTopics
)


from os import getenv
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic_ai import Agent

from src.models import (
    ResearchContext,
    ResearcherToolChoice, 
    ResearchTopics,
    Paper,
    EvaluatedPaper,
    ConversationIntent,
    TranslationResult,
)
from src.utils import ( 
    retrieve_arxiv_papers, 
    explain_paper_relevance,
    prepare_translation_input,
)
from src.exceptions import TranslationError

load_dotenv()

LLM_MODEL_NAME=getenv('LLM_MODEL_NAME')
chat_llm = init_chat_model(LLM_MODEL_NAME)

# Initialize the agent for topic research
LLM_MODEL=getenv('LLM_MODEL')

# ─── Farewell Classifier ─────────────────────────────────────────────────────────────────────────────────
farewell_classifier_prompt = """You identify when the user wants to end the conversation.
You must respond with a JSON object containing:
    - "intent": one of "continue", "quit", or "unsure"
If the user says "bye", "exit", "quit", or similar, set intent to "quit".
If the user asks a question or makes a statement that does not indicate an end to the conversation, set intent to "continue".
If you are unsure, set intent to "unsure".
Respond with no other text or formatting, just the JSON.
"""
farewell_agent = Agent(
    LLM_MODEL, 
    system_prompt=farewell_classifier_prompt,
    output_type=ConversationIntent
)

system_prompt="Extract user research topics from user query. Be as succint as you can, provide at most 5 topics."
topic_research_agent = Agent(
    LLM_MODEL, 
    system_prompt=system_prompt,
    output_type=ResearchTopics
)

# ─── Tool Selection Agent ─────────────────────────────────────────────────────────────────────────────────
tool_selection_prompt = """
You are the router for our multi-tool assistant.  Given the user’s final message,
respond with exactly one JSON field “tool” whose value is the name of the tool
that should handle this request. Your options are:
    • arxiv_research_tool  – for queries about finding or evaluating papers in academic research
    • "__end__"            – when you want to stop or just chat
Do not add any extra commentary.
"""
which_tool_agent = Agent(
    LLM_MODEL,
    system_prompt=tool_selection_prompt,
    output_type=ResearcherToolChoice,
)

# ─── Translation Agent ──────────────────────────────────────────────────────────────────────────────────
translation_prompt = """
You are a multilingual translation assistant. 
Your task is to translate the user's message to or from English.

- If the input is in English, translate it to the user's preferred language (detect it if possible).
- If the input is in another language, translate it into English.
- Respond with a JSON object containing:
    - "source_lang": the detected language code (e.g., "pt", "es", "de", "en")
    - "target_lang": the target language code
    - "translation": the translated text
    - "text": the original text
- Do not include any other explanation or formatting — only the JSON.
"""
translate_tool_agent = Agent(
    LLM_MODEL,
    system_prompt=translation_prompt,
    output_type=TranslationResult,
)

def translate_sync(text: str, preferred_lang="en", force_direction="to_en"):
    input_data = prepare_translation_input(text, preferred_lang, force_direction)
    input_str = str(input_data.model_dump())
    result = translate_tool_agent.run_sync(input_str)
    return result.output

@tool("arxiv_research_tool")
def arxiv_research_tool(research_topics: ResearchTopics) -> ResearchContext:
    """
    Extract and research topics from a query in arXiv with parallel fetch and evaluation.
    """
    # Initialize context
    ctx = ResearchContext(research_topics=research_topics, papers={})

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_input = {executor.submit(translate_sync, text): text for text in research_topics.topics}

        translated_topics = []
        for future in as_completed(future_to_input):
            input_text = future_to_input[future]
            try:
                translation_output = future.result()
                translated_topics.append(translation_output.translation)
            except Exception as e:
                # handle exceptions here
                raise TranslationError(f"Failed to translate: {input_text}") from e

    research_topics.topics = translated_topics

    # 1) Fetch papers concurrently per topic
    raw_papers_per_topic: Dict[str, List[Paper]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(research_topics.topics))) as fetch_executor:
        future_to_topic = {
            fetch_executor.submit(retrieve_arxiv_papers, topic): topic
            for topic in research_topics.topics
        }
        for future in as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                raw_papers_per_topic[topic] = future.result()
            except Exception as e:
                raw_papers_per_topic[topic] = []
                # log warning if needed

    # 2) Evaluate papers concurrently
    evaluated: List[EvaluatedPaper] = []
    def evaluate(paper_tuple: Tuple[str, Paper]) -> EvaluatedPaper:
        _, paper = paper_tuple
        eval_model = explain_paper_relevance(research_topics.user_query, paper)
        return EvaluatedPaper(**paper.model_dump(), evaluation=eval_model)

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as eval_executor:
        for topic, papers in raw_papers_per_topic.items():
            for paper in papers:
                tasks.append(eval_executor.submit(evaluate, (topic, paper)))

        for task in as_completed(tasks):
            try:
                ep = task.result()
                evaluated.append(ep)
            except Exception:
                # skip failing evaluations
                continue

    # 3) Store in context keyed by topic
    ctx['papers'] = {}
    for ep in evaluated:
        ctx['papers'].setdefault(ep.topic, []).append(ep)

    # Sort papers within each topic by evaluation score descending
    for topic, papers in ctx['papers'].items():
        papers.sort(key=lambda p: p.evaluation.score, reverse=True)

    return ctx

research_tools = [arxiv_research_tool]
research_tools_by_name = {tool.name: tool for tool in research_tools}

chat_llm_with_tools = chat_llm.bind_tools(research_tools)

def chatbot_node(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    prompts=[
        SystemMessage(
            content="You are a research assistant that can help users find and evaluate academic papers in arXiv repository."
        )
    ] + state["messages"]
    
    message = chat_llm_with_tools.invoke(prompts)
    
    return {
        "messages": [ message ]
    }

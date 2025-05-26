from os import getenv
from langchain_core.tools import tool
from openai import OpenAI

from models import ResearchContext, PaperEvaluation, Paper
from agents import topic_research_agent
from utils import (
    get_embedding, 
    from_scale_to_scale, 
    retrieve_arxiv_papers, 
    cossine_distance,
    get_scaled_similarity,
    explain_paper_relevance,
    get_last_human_message,
)

@tool("arxiv_research_tool", parse_docstring=True)
def arxiv_research_tool(ctx: ResearchContext) -> ResearchContext:
    """
    Extract the research topics from a query.

    Args:
        ctx: The research context containing the user query.

    Returns:
        The updated research context with extracted topics, papers metadata and evalutation.
    """
    last_human_message = get_last_human_message(ctx)

    ctx.research_topics = topic_research_agent.run_sync(last_human_message)
    ctx.papers=list(
        zip(
            ctx.research_topics,
            map(retrieve_arxiv_papers, ctx.research_topics)
        )
    )
    for paper in ctx.papers:
        paper.evaluation = explain_paper_relevance(query, paper)

    return ctx

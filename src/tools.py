from langchain_core.tools import tool

from src.models import ResearchContext
from src.agents import topic_research_agent
from src.utils import ( 
    retrieve_arxiv_papers, 
    explain_paper_relevance,
    get_last_human_message,
)

@tool("arxiv_research_tool", parse_docstring=True)
def arxiv_research_tool(ctx: ResearchContext) -> ResearchContext:
    """
    Extract and research topics from a query in arXiv.

    Args:
        ctx: The research context containing the user query.

    Returns:
        The updated research context with extracted topics, papers metadata and evaluation.
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
        paper.evaluation = explain_paper_relevance(last_human_message, paper)

    return ctx

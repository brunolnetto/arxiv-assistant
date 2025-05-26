from typing import List 
from functools import lru_cache

from numpy.linalg import norm
from numpy import dot
from openai import OpenAI

from models import PaperEvaluation, Paper, ResearchContext

client = OpenAI()

@lru_cache(maxsize=128)
def get_embedding(text: str, model: str):
    """Fetch and cache text embeddings."""
    return client.embeddings.create(input=text, model=model).data[0].embedding

def get_last_human_message(ctx: ResearchContext) -> str:
    return [msg for msg in ctx.messages if msg.get("role") == "user"][-1]["content"]

def cossine_distance(l_vec: str, r_vec: str):
    return dot(l_vec, r_vec) / (norm(l_vec) * norm(r_vec))

def from_scale_to_scale(value: float, l_scale: list[int], r_scale: list[int]) -> float:
    """
    Convert a value from one scale (l_scale) to another scale (r_scale).

    Args:
        value (float): The value to convert.
        l_scale (list[int]): The original scale as [min, max].
        r_scale (list[int]): The target scale as [min, max].

    Returns:
        float: The value mapped to the target scale.

    Raises:
        ValueError: If input scales are invalid or value is out of bounds.
    """
    # Validate input scales
    if len(l_scale) != 2 or len(r_scale) != 2:
        raise ValueError("Both l_scale and r_scale must have exactly two elements.")
    l_min, l_max = l_scale
    r_min, r_max = r_scale
    if l_min == l_max:
        raise ValueError("l_scale cannot have equal min and max values.")
    if r_min == r_max:
        raise ValueError("r_scale cannot have equal min and max values.")

    # Check if value is within l_scale
    if not (l_min <= value <= l_max):
        raise ValueError(f"Value {value} is outside the l_scale range {l_scale}.")

    # Perform the scaling
    scaled_value = ((value - l_min) * (r_max - r_min)) / (l_max - l_min) + r_min
    return scaled_value

def retrieve_arxiv_papers(topic: str, max_results: int=5) -> List[Paper]:
    import arxiv

    search = arxiv.Search(query=f'all:"{topic}"', max_results=max_results)
    client_arxiv = arxiv.Client()
    papers = []

    return list(map(
        lambda r: Paper(
            topic=topic,
            title=r.title,
            summary=r.summary,
            url=r.entry_id,
            published=r.published.strftime("%Y-%m-%d"),
        ), client_arxiv.results(search)
    ))

def get_scaled_similarity(l_query: str, r_query: str) -> float: 
    q_emb, d_emb = get_embedding(l_query), get_embedding(r_query)
    similarity = cossine_distance(q_emb, d_emb)
    return from_scale_to_scale(similarity, [-1, 1], [0, 1])

def explain_paper_relevance(query: str, paper: Paper) -> PaperEvaluation:
    client=OpenAI()

    # Embeddings
    paper_string=paper.title + "\n" + paper.summary
    scaled_similarity = get_scaled_similarity(query, paper_string)

    # GPT rationale
    system_msg = {
        "role": "system", 
        "content": "You are a research assistant able to evaluate the relevance of provided query topics."
    }
    user_msg = {
        "role": "user",
        "content": (
            f"User query: {query}\n"
            f"Paper Title: {paper.title}\nSummary: {paper.summary}\n"
            "Explain relevance in 1-2 sentences."
        )
    }
    messages=[system_msg, user_msg]
    completion = client.chat.completions.create(model=LLM_MODEL_NAME, messages=messages)
    rationale = completion.choices[0].message.content.strip()

    return PaperEvaluation(topic=paper.topic, score=scaled_similarity, rationale=rationale)
from typing import Dict, List, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class PaperEvaluation(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str

class Paper(BaseModel):
    topic: str
    title: str
    summary: str
    url: str
    published: str
    evaluation: PaperEvaluation

class ResearchTopics(BaseModel):
    topics: list[str]

class ResearchContext(TypedDict):
    messages: Annotated[list, add_messages]
    research_topics: ResearchTopics
    papers: Dict[str, List[Paper]]

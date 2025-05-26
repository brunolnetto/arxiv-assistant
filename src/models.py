from enum import Enum
from typing import Dict, List, Optional
from typing_extensions import TypedDict, List, Literal

from pydantic import BaseModel, Field

class PaperEvaluation(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str

class Paper(BaseModel):
    topic: str
    title: str
    summary: str
    url: str
    published: str

class ConversationIntent(BaseModel):
    intent: Literal["continue", "quit", "unsure"] = Field(
        ...,
        description="The intent of the conversation, indicating whether to continue, quit, or if unsure."
    )

class EvaluatedPaper(Paper):
    evaluation: PaperEvaluation

class ResearchTopics(BaseModel):
    user_query: str = Field(..., description="The original user query from which topics are extracted")
    topics: List[str] = Field(..., description="List of research topics mentioned in the query")

class ResearchContext(TypedDict):
    research_topics: ResearchTopics
    papers: Dict[str, List[Paper]]

class ResearcherToolChoice(BaseModel):
    tool: Literal[
        "__end__",  
        "arxiv_research_tool",
    ]

class TranslationInput(BaseModel):
    text: str
    preferred_lang: Optional[str] = None  # e.g. 'pt'
    force_direction: Literal["to_en", "from_en", "auto"] = "auto"

class TranslationResult(BaseModel):
    source_lang: str  # e.g. "en", "pt", "de"
    target_lang: str
    text: str
    translation: str
    
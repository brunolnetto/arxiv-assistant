from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from src.models import ResearchContext
from src.agents import chat_llm
from src.tools import arxiv_research_tool

def chatbot(ctx: ResearchContext):
    messages=ctx["messages"]
    message=chat_llm.invoke(messages)
    return {"messages": [message]}

graph_builder = StateGraph(ResearchContext)

tools_node = ToolNode([ arxiv_research_tool ])

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools_node)

# Define edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# ---- 5. Build Graph & Agent ----
memory=MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
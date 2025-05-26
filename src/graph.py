from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from typing import Literal

from src.agents import chatbot_node, research_tools_by_name

def should_research(state: MessagesState) -> Literal["research_pool", END]:
    """
    Determines if the last message is related to research and should invoke the research pool.
    If the last message contains tool calls, it indicates a research-related query.
    If not, it returns END to stop the workflow.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return 'research_pool'

    return END

def research_pool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = research_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        tool_message = ToolMessage(content=observation, tool_call_id=tool_call["id"])

        result.append(tool_message)

    return {"messages": result}

def build_research_graph():
    """
    Build the research graph for interacting with the chatbot and tools.
    This function sets up the nodes and edges of the graph, defining how the chatbot interacts with the tools.
    """
    g = StateGraph(MessagesState)

    # Start node
    g.add_edge(START, "chatbot")

    # Nodes
    g.add_node("chatbot", chatbot_node)
    g.add_node("research_pool",  research_pool_node)

    # Conditional: if not research-related → END, else → chatbot
    g.add_conditional_edges("chatbot", should_research)

    # After tools
    g.add_edge("research_pool", "chatbot")

    memory=MemorySaver()
    return g.compile(checkpointer=memory)

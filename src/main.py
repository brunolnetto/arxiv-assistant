from dotenv import load_dotenv
import asyncio

load_dotenv()

from rich.console import Console

from src.display import interact_with_graph
from src.graph import build_research_graph 

console = Console()

if __name__ == "__main__":
    # Clear the console at the start
    console.clear()

    # Build the research graph and start interaction
    console.print("Building research graph...", style="bold green")
    graph = build_research_graph()
    console.print(f"{graph.get_graph().print_ascii()}", style="bold green")
    interact_with_graph(graph)

    # Clear the console after interaction
    console.clear()
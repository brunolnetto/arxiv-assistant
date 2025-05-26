from dotenv import load_dotenv
import asyncio

load_dotenv()

from rich.console import Console

from src.display import interact_with_graph
from src.graph import build_research_graph 

console = Console()

if __name__ == "__main__":
    # The line `console()` is attempting to call the `console` object as a function, which is not
    # valid in this context. The `console` object is an instance of the `rich.console.Console` class,
    # which is typically used for displaying styled and formatted text in the terminal.
    console.clear()

    graph = build_research_graph()

    graph.get_graph().print_ascii()
    interact_with_graph(graph)

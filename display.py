import json
import numpy as np
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.json import JSON
from rich.syntax import Syntax
from rich.spinner import Spinner

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

console = Console()

# ─── Prompt Toolkit Setup ────────────────────────────────────────────────────────
commands = ['exit', 'quit', 'help']
completer = WordCompleter(commands, ignore_case=True)
session = PromptSession(
    completer=completer,
    history=InMemoryHistory(),
    style=Style.from_dict({"": "#00ff00", "prompt": "bold green"}),
    multiline=False,
)

def prepare_config() -> dict:
    return {
        "stream_mode": "values",
        "stream": True,
        "stream_interval": 0.1,
        "max_tokens": 200,
        "temperature": 0.5,
        "configurable": {
            "user_id": "default",
            "thread_id": 1
        }
    }

# ─── Streaming Function ─────────────────────────────────────────────────────────
def stream_graph_updates(graph, user_input: str, config: dict = {}):
    markdown_buffer = ""
    json_blobs: list[str] = []

    def render_panel():
        ts = datetime.now().strftime("%H:%M:%S")
        title = f"[dim]{ts}[/dim] [bold green]AI Response[/bold green]"
        return Panel(Markdown(markdown_buffer), title=title, border_style="cyan")

    # internal helper to resume after human interjection
    def handle_human_pause(pause_call: ToolMessage):
        # pause_call.tool_input might carry context if you need it
        human_resp = console.input("[bold yellow]AI requests your input →[/] ").strip()
        resume_cmd = Command(resume={"data": human_resp})
        # now re-enter the stream with the resume command
        return graph.stream(resume_cmd, config=config, stream_mode=["values"])    

    try:
        with Live(render_panel(), console=console, refresh_per_second=12) as live:
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            
            for event in graph.stream(inputs, config=config):
                for value in event.values():
                    # detect a “human assistance” tool call
                    if isinstance(value, ToolMessage) and value.tool_name == "human_assistance":
                        break

                    # This assumes message content comes as incremental string chunks
                    message = value.get("messages", [])[-1]
                    if not message:
                        continue

                    chunk = getattr(message, "content", "")

                    if not chunk.strip():
                        continue

                    # Check if it's JSON-like — avoid appending raw JSON
                    try:
                        json.loads(chunk)
                        json_blobs.append(chunk)
                        continue
                    except json.JSONDecodeError:
                        pass

                    # Append markdown
                    markdown_buffer += chunk
                    live.update(render_panel())

    except KeyboardInterrupt:
        console.print("\n[bold red]⏹ Aborted current response.[/bold red]")
        return

    if json_blobs:
        console.rule("[bold blue]Tool-Call JSON Outputs")
        for blob in json_blobs:
            try:
                console.print(JSON(blob, indent=2))
            except Exception:
                console.print(Syntax(blob, "json", line_numbers=False))


# ─── Main Loop ──────────────────────────────────────────────────────────────────
def interact_with_graph(graph, config: dict = None):
    if not config:
        config = prepare_config()
    

    console.print(
        Panel(
            "[bold magenta]LangGraph CLI[/bold magenta]\n"
            "Type [green]'exit'[/green] or [green]'help'[/green] to quit",
            title="LangGraph Chat",
            style="bold magenta"
        )
    )

    while True:
        # 1) Prompt for user input
        try:
            user_input = session.prompt(HTML("<prompt>You:</prompt> ")).strip()
        except KeyboardInterrupt:
            console.print("[bold yellow]Input cancelled—sending final goodbye…[/bold yellow]")
            stream_graph_updates(graph, "Goodbye", config)
            break

        cmd = user_input.lower()

        # 2) Handle control commands
        if cmd in {"exit", "quit", "q"}:
            console.print("[bold yellow]Exit requested—sending final goodbye…[/bold yellow]")
            stream_graph_updates(graph, "Goodbye", config)
            break

        if cmd == "help":
            # Use AI to generate dynamic help info, including listing tools and context
            help_prompt = (
                "Please provide a prompt-oriented user perspective list of available tools and relevant usage instructions."
                "for this LangGraph environment. No code leak. Only provide tools you are currently equipped with."
            )
            try:
                stream_graph_updates(graph, help_prompt, config)
            except Exception as e:
                console.print(Panel(f"[bold red]Error fetching help:[/] {e}", style="red"))
            continue

        # 3) Echo the user's message
        timestamp = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{timestamp}[/] [bold cyan]You:[/] {user_input}")

        # 4) Stream the AI response
        try:
            stream_graph_updates(graph, user_input, config)
        except Exception as e:
            console.print(Panel(f"[bold red]Error during stream:[/] {e}", style="red"))
            continue


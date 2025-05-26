import json
from datetime import datetime
import traceback
from time import time
import asyncio

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

from src.agents import farewell_agent

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

# ─── Utility Functions ────────────────────────────────────────────────────────
def handle_exception(e, verbose=False):
    if verbose:
        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        console.print(Panel(tb, title="[red]Detailed Error[/red]", style="red"))
    else:
        msg = f"{type(e).__name__}: {e}"
        if e.__cause__:
            msg += f"\nCaused by {type(e.__cause__).__name__}: {e.__cause__}"
        console.print(Panel(msg, title="[bold red]Error[/bold red]", style="red"))

def print_node_state(node_name: str, message: any, max_length: int = 2000) -> None:
    """Nicely prints different message types with spinner and thinking effect."""    
    try:
        if hasattr(message, "content"):
            content = message.content
            if not isinstance(content, str):
                content = str(content)

            content_trim = content.strip()
            if content_trim.startswith("{") or content_trim.startswith("["):
                try:
                    data = json.loads(content_trim)
                    console.print(
                        Panel(
                            JSON(data, indent=2),
                            title=f"[bold blue]{node_name} Output (parsed JSON)",
                            border_style="blue",
                        )
                    )
                    return
                except json.JSONDecodeError:
                    pass

            if len(content) > max_length:
                content = content[:max_length] + "...[truncated]"
            console.print(
                Panel(content, title=f"[bold blue]{node_name} Output (text)", border_style="blue")
            )
            return

        if hasattr(message, "tool_calls") and getattr(message, "tool_calls"):
            calls_summary = []
            for call in message.tool_calls:
                tool_name = call.get("name", "unknown_tool")
                args = call.get("args", {})
                try:
                    arg_str = json.dumps(args, indent=2)
                    if len(arg_str) > max_length:
                        arg_str = arg_str[:max_length] + "...[truncated]"
                except Exception:
                    arg_str = str(args)
                calls_summary.append({"tool": tool_name, "args": arg_str})
            console.print(
                Panel(
                    JSON(calls_summary, indent=2),
                    title=f"[bold blue]{node_name} Tool Calls",
                    border_style="blue",
                )
            )
            return

        if hasattr(message, "model_dump") and callable(message.model_dump):
            try:
                data = message.model_dump()
                console.print(
                    Panel(
                        JSON(data, indent=2),
                        title=f"[bold blue]{node_name} Output (model_dump)",
                        border_style="blue",
                    )
                )
                return
            except Exception as e:
                console.print(
                    Panel(f"Error dumping model: {e}", title=f"[bold red]{node_name} Dump Error", border_style="red")
                )

        if isinstance(message, dict):
            console.print(
                Panel(
                    JSON(message, indent=2),
                    title=f"[bold blue]{node_name} Output (dict)",
                    border_style="blue",
                )
            )
            return

        if isinstance(message, (bytes, bytearray)):
            try:
                decoded = message.decode("utf-8")
                console.print(
                    Panel(decoded, title=f"[bold blue]{node_name} Output (bytes decoded)", border_style="blue")
                )
                return
            except Exception:
                console.print(
                    Panel(repr(message), title=f"[bold blue]{node_name} Output (bytes raw)", border_style="red")
                )
                return

        s = str(message)
        if len(s) > max_length:
            s = s[:max_length] + "...[truncated]"
        console.print(
            Panel(s, title=f"[bold blue]{node_name} Output (Raw)", border_style="red")
        )

    except Exception as ex:
        console.print(
            Panel(f"Failed to print node state: {ex}", title=f"[bold red]{node_name} Output Error", border_style="red")
        )

# ─── Streaming Function ────────────────────────────────────────────────────────
async def show_thinking(live: Live, node_name: str):
    spinner = Spinner("dots", text=f"{node_name} is thinking...")
    while True:
        live.update(spinner)
        await asyncio.sleep(0.1)

def stream_graph(graph, user_input: str, config: dict = {}, log_json: bool = False):
    markdown_buffer = ""
    json_buffer = ""
    json_blobs = []

    def render_panel():
        ts = datetime.now().strftime("%H:%M:%S")
        title = f"[dim]{ts}[/dim] [bold green]AI Rationale[/bold green]"
        return Panel(Markdown(markdown_buffer), title=title, border_style="cyan")

    try:
        with Live(render_panel(), console=console, refresh_per_second=12) as live:
            inputs = {"messages": [{"role": "user", "content": user_input}]}

            for event in graph.stream(inputs, config=config):
                for node_name, tool_data in event.items():
                    messages = tool_data.get("messages", [])

                    if isinstance(messages, list) and all(hasattr(m, "content") for m in messages):
                        for msg in messages:
                            print_node_state(node_name, msg)

                    # Rationale or partial answer rendering
                    msg = tool_data.get("messages", [])[-1] if tool_data.get("messages") else None
                    if msg and hasattr(msg, "content"):
                        chunk = msg.content
                        if chunk.strip().startswith("{") or chunk.strip().startswith("["):
                            json_buffer += chunk
                            try:
                                obj = json.loads(json_buffer)
                                json_blobs.append(obj)
                                json_buffer = ""  # Reset
                            except json.JSONDecodeError:
                                continue  # Wait for complete
                            continue
                        else:
                            markdown_buffer += chunk
                            live.update(render_panel())

    except KeyboardInterrupt:
        console.print("\n[bold red]⏹ Aborted current response.[/bold red]")
        return

    # Structured logs at the end
    if log_json and json_blobs:
        console.rule("[bold yellow]Tool-Call JSON Outputs")
        for blob in json_blobs:
            try:
                console.print(JSON(blob, indent=2))
            except Exception:
                console.print(Syntax(json.dumps(blob, indent=2), "json"))


# ─── Main Loop ──────────────────────────────────────────────────────────────────
def interact_with_graph(graph, config: dict = None):
    if not config:
        config = prepare_config()

    startup_panel = Panel(
        "[bold magenta]LangGraph CLI[/bold magenta]\n"
        "Type [green]'exit'[/green] or [green]'help'[/green] to quit",
        title="LangGraph Chat",
        style="bold magenta"
    )
    console.print(startup_panel)

    while True:
        # 1) Prompt for user input
        try:
            user_input = session.prompt(HTML("<prompt>You:</prompt> ")).strip()
        except KeyboardInterrupt:
            console.print("[bold yellow]Input cancelled—sending final goodbye…[/bold yellow]")
            stream_graph(graph, "Goodbye", config)
            break

        cmd = user_input.lower()
        chat_intent = farewell_agent.run_sync(cmd).output.intent
        
        # 2) Handle control commands
        if chat_intent in 'quit':
            console.print("[bold yellow]👋 Exit requested—sending final goodbye…[/bold yellow]")
            stream_graph(graph, user_input, config)
            break

        if cmd == "help":
            # Use AI to generate dynamic help info, including listing tools and context
            help_prompt = (
                "Provide a prompt-oriented user perspective list of available tools and relevant usage instructions."
                "for this LangGraph environment. No code leak. Only provide tools you are currently equipped with."
            )
            try:
                stream_graph(graph, help_prompt, config)
            except Exception as e:
                console.print(Panel(f"[bold red]Error fetching help:[/] {e}", style="red"))
            continue

        # 3) Echo the user's message
        timestamp = datetime.now().strftime("%H:%M:%S")
        console.print(f"[dim]{timestamp}[/] [bold cyan]You:[/] {user_input}")

        # 4) Stream the AI response
        try:
            stream_graph(graph, user_input, config)
        except Exception as e:
            handle_exception(e, verbose=True)
            continue


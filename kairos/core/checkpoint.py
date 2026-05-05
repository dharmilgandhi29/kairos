from dataclasses import dataclass
from typing import Callable, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()


@dataclass
class CheckpointResult:
    approved: bool
    final_output: str
    human_edited: bool
    feedback: str = ""


def human_checkpoint(
    task: str,
    output: str,
    confidence: float,
    reasoning_chain: list,
    mode: str,
    on_approve: Optional[Callable] = None,
    on_reject: Optional[Callable] = None
) -> CheckpointResult:
    """
    Shows human a review interface when confidence is low.
    Human can approve, edit, or reject the output.
    Returns CheckpointResult with final decision.
    """

    console.print("\n")
    console.print(Panel.fit(
        f"[yellow]⚠ Human Review Required[/yellow]\n"
        f"Confidence: [red]{confidence:.0%}[/red] — below threshold",
        border_style="yellow"
    ))

    # Show task
    console.print(f"\n[bold]Task:[/bold] {task}")

    # Show reasoning if System 2
    if reasoning_chain:
        console.print(f"\n[bold]Reasoning ({len(reasoning_chain)} steps):[/bold]")
        for i, step in enumerate(reasoning_chain[:5], 1):
            console.print(f"  {i}. {step[:100]}")
        if len(reasoning_chain) > 5:
            console.print(f"  ... and {len(reasoning_chain) - 5} more steps")

    # Show output
    console.print(Panel(
        output,
        title=f"[cyan]Agent Output ({mode})[/cyan]",
        border_style="cyan"
    ))

    # Show confidence table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Mode", f"[cyan]{mode}[/cyan]")
    table.add_row("Confidence", f"[{'green' if confidence >= 0.7 else 'red'}]{confidence:.0%}[/]")
    console.print(table)

    # Human decision
    console.print("\n[bold]What would you like to do?[/bold]")
    console.print("  [green]1. Approve[/green] — use this output as is")
    console.print("  [yellow]2. Edit[/yellow] — modify the output")
    console.print("  [red]3. Reject[/red] — discard this output")

    choice = Prompt.ask("\nChoice", choices=["1", "2", "3"], default="1")

    if choice == "1":
        console.print("[green]✓ Output approved[/green]")
        if on_approve:
            on_approve(output)
        return CheckpointResult(
            approved=True,
            final_output=output,
            human_edited=False
        )

    elif choice == "2":
        console.print("\n[yellow]Enter your edited version:[/yellow]")
        edited = Prompt.ask("Edited output")
        feedback = Prompt.ask("Feedback for agent (optional)", default="")
        console.print("[green]✓ Edited output saved[/green]")
        return CheckpointResult(
            approved=True,
            final_output=edited,
            human_edited=True,
            feedback=feedback
        )

    else:
        feedback = Prompt.ask("Why are you rejecting this? (helps agent learn)", default="")
        console.print("[red]✗ Output rejected[/red]")
        if on_reject:
            on_reject(output, feedback)
        return CheckpointResult(
            approved=False,
            final_output="",
            human_edited=False,
            feedback=feedback
        )
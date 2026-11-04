"""Baudrier: chat with your Apertus agent in the terminal.

Two loops, like Claude Code:
  - outer loop (here): wait for your message, show the answer, repeat;
  - inner loop (Agent.ask in agent.py): tool calls until the answer.

The conversation is kept between messages, so you can ask follow-up
questions ("and tomorrow?", "what about the other file?").

Commands:
    /reset   forget the conversation
    /exit    quit (or Ctrl+D)

Usage:
    python baudrier.py
    python baudrier.py --model models/apertus-v1.5-8b-text-8bit
    python baudrier.py --quiet      # hide the tool calls
    python baudrier.py --workspace examples/notes   # use the demo notes
"""
import argparse

import tools
from agent import DEFAULT_MODEL, Agent

parser = argparse.ArgumentParser(description="Chat with the Baudrier agent (Apertus).")
parser.add_argument("--model", default=DEFAULT_MODEL, help="path of the MLX model folder")
parser.add_argument("--workspace", help="folder the agent can read (default: workspace/)")
parser.add_argument("--quiet", action="store_true", help="hide the tool calls and token counts")
cli = parser.parse_args()

print(f"Loading {cli.model}...")
agent = Agent(cli.model, verbose=not cli.quiet, workspace=cli.workspace)
print(f"Workspace: {tools.WORKSPACE}")
print("Baudrier is ready. Type /exit to quit, /reset to start a new conversation.\n")

while True:
    try:
        question = input("> ").strip()
    except (EOFError, KeyboardInterrupt):  # Ctrl+D or Ctrl+C
        print()
        break

    if not question:
        continue
    if question in ("/exit", "/quit"):
        break
    if question == "/reset":
        agent.reset()
        print("(Conversation forgotten.)\n")
        continue

    answer = agent.ask(question)
    print(f"\n{answer}\n")

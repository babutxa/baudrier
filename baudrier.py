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
    python baudrier.py                                   # local Apertus 8B (4-bit)
    python baudrier.py --model models/apertus-v1.5-8b-text-8bit
    python baudrier.py --backend remote                  # Apertus 70B on a server (see backends.py)
    python baudrier.py --quiet                           # hide the tool calls
    python baudrier.py --workspace examples/notes        # use the demo notes
"""
import argparse
import sys

import tools
from agent import Agent
from backends import make_backend

parser = argparse.ArgumentParser(description="Chat with the Baudrier agent (Apertus).")
parser.add_argument("--backend", choices=["local", "remote"], default="local",
                    help="local: mlx-lm on this Mac; remote: an OpenAI-compatible server")
parser.add_argument("--model", help="local: MLX model folder; remote: model name")
parser.add_argument("--workspace", help="folder the agent can read (default: workspace/)")
parser.add_argument("--quiet", action="store_true", help="hide the tool calls and token counts")
cli = parser.parse_args()

# Check the workspace before loading the model (which takes a few seconds),
# and show a short error instead of a Python traceback.
if cli.workspace is not None:
    try:
        tools.set_workspace(cli.workspace)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

try:
    backend = make_backend(cli.backend, cli.model)
except RuntimeError as e:
    sys.exit(f"Error: {e}")
agent = Agent(backend, verbose=not cli.quiet, workspace=cli.workspace)
print(f"Model: {backend.name}")
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

    try:
        answer = agent.ask(question)
    except RuntimeError as e:  # e.g. the remote server is not reachable
        print(f"\nError: {e}\n")
        continue
    print(f"\n{answer}\n")

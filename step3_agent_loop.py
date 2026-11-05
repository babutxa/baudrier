"""Step 3: the agent loop, for one question.

The loop itself is in agent.py (Agent.ask). The model decides which tools
to use and in which order. The harness only runs them and gives back the
results.

Usage:
    python step3_agent_loop.py "Look at my notes. Where am I going today, and do I need an umbrella?"
    python step3_agent_loop.py --model models/apertus-v1.5-8b-text-8bit "..."
    python step3_agent_loop.py --backend remote "..."        # Apertus 70B on a server
    python step3_agent_loop.py --workspace workspace "..."   # your own notes
"""
import argparse
import sys

import tools
from agent import Agent
from backends import make_backend

# Command-line options (argparse also generates --help).
parser = argparse.ArgumentParser(description="Baudrier agent loop with Apertus, one question.")
parser.add_argument("question", nargs="?",
                    default="Look at my notes. Where am I going today, and do I need an umbrella?")
parser.add_argument("--backend", choices=["local", "remote"], default="local",
                    help="local: mlx-lm on this Mac; remote: an OpenAI-compatible server")
parser.add_argument("--model", help="local: MLX model folder; remote: model name")
# This step is a demo, so it uses the demo notes by default.
parser.add_argument("--workspace", default="examples/notes",
                    help="folder the agent can read (default: examples/notes)")
cli = parser.parse_args()

# Check the workspace before loading the model (which takes a few seconds),
# and show a short error instead of a Python traceback.
try:
    tools.set_workspace(cli.workspace)
except FileNotFoundError as e:
    sys.exit(f"Error: {e}")

try:
    backend = make_backend(cli.backend, cli.model)
except RuntimeError as e:
    sys.exit(f"Error: {e}")
print(f"Model: {backend.name}")
agent = Agent(backend, workspace=cli.workspace)

print(f"\nUSER: {cli.question}\n")
try:
    answer = agent.ask(cli.question)
except RuntimeError as e:  # e.g. the remote server is not reachable
    sys.exit(f"Error: {e}")
print(f"\nASSISTANT: {answer}")

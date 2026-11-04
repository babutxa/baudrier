"""Step 3: the agent loop, for one question.

The loop itself is in agent.py (Agent.ask). The model decides which tools
to use and in which order. The harness only runs them and gives back the
results.

Usage:
    python step3_agent_loop.py "Look at my notes. Where am I going today, and do I need an umbrella?"
    python step3_agent_loop.py --model models/apertus-v1.5-8b-text-8bit "..."
    python step3_agent_loop.py --workspace workspace "..."   # your own notes
"""
import argparse

from agent import DEFAULT_MODEL, Agent

# Command-line options (argparse also generates --help).
parser = argparse.ArgumentParser(description="Baudrier agent loop with Apertus, one question.")
parser.add_argument("question", nargs="?",
                    default="Look at my notes. Where am I going today, and do I need an umbrella?")
parser.add_argument("--model", default=DEFAULT_MODEL, help="path of the MLX model folder")
# This step is a demo, so it uses the demo notes by default.
parser.add_argument("--workspace", default="examples/notes",
                    help="folder the agent can read (default: examples/notes)")
cli = parser.parse_args()

print(f"Loading {cli.model}...")
agent = Agent(cli.model, workspace=cli.workspace)

print(f"\nUSER: {cli.question}\n")
answer = agent.ask(cli.question)
print(f"\nASSISTANT: {answer}")

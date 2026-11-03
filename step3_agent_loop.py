"""Step 3: the agent loop.

Repeat until the model answers without a tool:
  1. the model generates (text, and maybe tool calls);
  2. if there are tool calls: run them, add the results, go back to 1;
  3. if not: this is the final answer, stop.

The model decides which tools to use and in which order. The harness
only runs them and gives back the results.

Usage:
    python step3_agent_loop.py "Look at my notes. Where am I going today, and do I need an umbrella?"
    python step3_agent_loop.py --model models/apertus-v1.5-8b-text-8bit "..."
"""
import argparse
import json
import re
from typing import Any

from mlx_lm import load, stream_generate

from tools import TOOL_DESCRIPTIONS, TOOL_FUNCTIONS, ToolResult

# Type aliases (like "using" in C++):
# a message is a dict like {"role": "user", "content": "..."};
# a tool call is (tool name, arguments), e.g. ("get_weather", {"city": "Zurich"}).
Message = dict[str, Any]
ToolCall = tuple[str, dict[str, Any]]

# Command-line options (argparse also generates --help).
parser = argparse.ArgumentParser(description="Baudrier agent loop with Apertus.")
parser.add_argument("question", nargs="?",
                    default="Look at my notes. Where am I going today, and do I need an umbrella?")
parser.add_argument("--model", default="models/apertus-v1.5-8b-text-4bit",
                    help="path of the MLX model folder")
cli = parser.parse_args()

MODEL: str = cli.model
MAX_STEPS: int = 8  # safety: stop if the model never gives a final answer

TOOLS_PREFIX = "<|tools_prefix|>"
TOOLS_SUFFIX = "<|tools_suffix|>"
ASSISTANT_END = "<|assistant_end|>"

print(f"Loading {MODEL}...")
# load() is declared to return 2 or 3 values (3 with return_config=True),
# so we take the first two by index: correct for the type checker in both cases.
loaded = load(MODEL)
model, tokenizer = loaded[0], loaded[1]
TOOLS_SUFFIX_ID: int = tokenizer.convert_tokens_to_ids(TOOLS_SUFFIX)


def generate(messages: list[Message], add_generation_prompt: bool) -> str:
    """Run the model from the conversation; return the new raw text."""
    # Simple version: rebuild and re-tokenize the whole conversation each time.
    # continue_assistant_message: after tool results, keep the assistant turn
    # OPEN, so the model continues it. Without it, the v1.5 template closes
    # the turn with <|assistant_end|>.
    text = tokenizer.apply_chat_template(
        messages, tools=TOOL_DESCRIPTIONS, tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_assistant_message=not add_generation_prompt,
    )
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)

    new_ids: list[int] = []
    for response in stream_generate(model, tokenizer, prompt_ids, max_tokens=500):
        new_ids.append(response.token)
        if response.token == TOOLS_SUFFIX_ID:
            break  # the model asked for a tool: the harness takes over
        # (stream_generate stops by itself on <|assistant_end|>)

    print(f"  [{len(prompt_ids)} tokens in -> {len(new_ids)} tokens out]")
    return tokenizer.decode(new_ids)


class BadToolCall(Exception):
    """The model wrote a tool call that is not valid JSON."""

    def __init__(self, text: str, call_text: str) -> None:
        super().__init__(call_text)
        self.text = text            # what the model wrote before the call
        self.call_text = call_text  # the broken call


def split_output(raw: str) -> tuple[str, list[ToolCall]]:
    """Split the raw output into (text, tool_calls).

    'Let me check.<|tools_prefix|>[{"get_weather": {"city": "X"}}]<|tools_suffix|>'
    -> ('Let me check.', [("get_weather", {"city": "X"})])

    Raises BadToolCall if the call cannot be parsed.
    """
    raw = raw.replace(ASSISTANT_END, "")
    if TOOLS_PREFIX not in raw:
        return raw.strip(), []
    text, rest = raw.split(TOOLS_PREFIX, 1)
    inside = rest.split(TOOLS_SUFFIX, 1)[0]
    # Repair a known small-model mistake: a tool without arguments written as
    # {"list_files": } instead of {"list_files": {}}. In valid JSON a key is
    # always followed by a value, so '": }' only appears in this broken case.
    inside = re.sub(r'":\s*}', '": {}}', inside)
    try:
        calls = [(name, args) for call in json.loads(inside) for name, args in call.items()]
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        raise BadToolCall(text.strip(), inside) from e
    return text.strip(), calls


def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Run one tool. Errors become text for the model, not a crash."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOL_FUNCTIONS)}."
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:  # wrong arguments, network error, ...
        return f"Error running {name}: {e}"


question: str = cli.question
# The system prompt tells the model to do the work itself with the tools,
# instead of asking the user to do it (small models often stop too early).
SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Use the tools to find all the "
    "information needed to answer completely. Never ask the user to look up "
    "something that one of your tools can find."
)
messages: list[Message] = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]
print(f"\nUSER: {question}")

# First generation: a new assistant turn starts.
add_generation_prompt = True

for step in range(1, MAX_STEPS + 1):
    print(f"\n--- Step {step}: Apertus generates")
    raw = generate(messages, add_generation_prompt)
    # After the first step, the assistant turn stays open (the tool results
    # are inside it), so we must not add a new <|assistant_start|>.
    add_generation_prompt = False

    try:
        text, calls = split_output(raw)
    except BadToolCall as bad:
        # The model wrote a broken tool call. Don't stop: close its turn and
        # tell it what went wrong, so it can try again. (A "tool" message
        # needs a valid call before it, so the error comes as a user message.)
        print(f"  BAD TOOL CALL: {bad.call_text!r} -> asking the model to retry")
        messages.append({"role": "assistant", "content": bad.text})
        messages.append({"role": "user", "content":
                         f"Your tool call was not valid JSON: {bad.call_text}. "
                         "Please call the tool again with valid JSON."})
        add_generation_prompt = True  # a new assistant turn starts
        continue

    if text:
        print(f"  TEXT: {text}")

    if not calls:
        # No tool call: this is the final answer.
        print(f"\nASSISTANT: {text}")
        break

    # Record what the model wrote: its text AND its tool calls.
    messages.append({
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for name, args in calls
        ],
    })

    # Run each tool and add its result to the conversation.
    for name, args in calls:
        result = run_tool(name, args)
        print(f"  TOOL: {name}({args}) -> {result!r}")
        messages.append({"role": "tool", "content": json.dumps(result)})
else:
    # The for loop ended without break: too many steps.
    print(f"\nStopped after {MAX_STEPS} steps without a final answer.")

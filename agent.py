"""The agent: a model backend + tools + the inner loop.

One call to Agent.ask(question) is one user turn. Inside it, repeat until
the model answers without a tool:
  1. the model generates (text, and maybe tool calls);
  2. if there are tool calls: run them, add the results, go back to 1;
  3. if not: this is the final answer, return it.

The conversation (self.messages) is kept between calls, so the model
remembers the earlier questions and answers.

The model can be local (mlx-lm) or remote (an OpenAI-compatible server):
see backends.py. The loop is the same for both.
"""
import json
from typing import Any

import tools
from backends import Backend, BadToolCall, Message
from tools import TOOL_DESCRIPTIONS, TOOL_FUNCTIONS, ToolResult

# The system prompt tells the model:
#  - to do the work itself with the tools (small models often stop too early);
#  - where the user's plans are, so it reads the notes instead of guessing
#    (without this, it once invented "New York" as the user's city).
# Keep it short: a small model follows a few simple rules, but more rules can
# make it worse. A rule "never state facts before the tool result" did not
# stop it from inventing the weather, and made it call get_weather in a loop.
SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. Use the tools to find all the "
    "information needed to answer completely. Never ask the user to look up "
    "something that one of your tools can find.\n"
    "The user's plans and personal information are in their notes. If you "
    "don't know where the user is or where they are going, read the notes "
    "first (list_files, then read_file). Never invent a city."
)


def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Run one tool. Errors become text for the model, not a crash."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOL_FUNCTIONS)}."
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:  # wrong arguments, network error, ...
        return f"Error running {name}: {e}"


class Agent:
    def __init__(self, backend: Backend, max_steps: int = 8,
                 verbose: bool = True, workspace: str | None = None) -> None:
        self.backend = backend
        self.max_steps = max_steps  # safety: stop if the model never gives a final answer
        self.verbose = verbose      # print each step (tool calls, token counts)
        if workspace is not None:
            tools.set_workspace(workspace)
        tools.WORKSPACE.mkdir(exist_ok=True)  # the default workspace/ may not exist yet
        self.messages: list[Message] = []
        self.reset()

    def reset(self) -> None:
        """Forget the conversation (keep only the system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def ask(self, question: str) -> str:
        """One user turn: the inner loop. Returns the final answer."""
        self.messages.append({"role": "user", "content": question})
        # The tool calls already done in this turn, to detect repetitions.
        done_calls: set[str] = set()

        for step in range(1, self.max_steps + 1):
            self.log(f"--- Step {step}")
            try:
                turn = self.backend.generate(self.messages, TOOL_DESCRIPTIONS)
            except BadToolCall as bad:
                # The model wrote a broken tool call. Don't stop: close its turn and
                # tell it what went wrong, so it can try again. (A "tool" message
                # needs a valid call before it, so the error comes as a user message.)
                self.log(f"  BAD TOOL CALL: {bad.call_text!r} -> asking the model to retry")
                self.messages.append({"role": "assistant", "content": bad.text})
                self.messages.append({"role": "user", "content":
                                      f"Your tool call was not valid JSON: {bad.call_text}. "
                                      "Please call the tool again with valid JSON."})
                continue
            self.log(f"  [{turn.tokens_in} tokens in -> {turn.tokens_out} tokens out]")

            if not turn.tool_calls:
                # No tool call: this is the final answer. Keep it in the
                # conversation, so the next question can refer to it.
                self.messages.append({"role": "assistant", "content": turn.text})
                return turn.text

            if turn.text:
                self.log(f"  TEXT: {turn.text}")

            # Record what the model wrote: its text AND its tool calls.
            self.messages.append({
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                    for call in turn.tool_calls
                ],
            })

            # Run each tool and add its result to the conversation.
            for call in turn.tool_calls:
                # Protection against loops: a small model sometimes repeats
                # exactly the same call. Don't run it again; tell it to answer.
                key = call.name + json.dumps(call.arguments, sort_keys=True)
                if key in done_calls:
                    result: ToolResult = ("You already called this tool with the same "
                                          "arguments; the result is above. Now answer the user.")
                    self.log(f"  REPEATED: {call.name}({call.arguments}) -> not run again")
                else:
                    done_calls.add(key)
                    result = run_tool(call.name, call.arguments)
                    self.log(f"  TOOL: {call.name}({call.arguments}) -> {result!r}")
                # tool_call_id links the result to its call (needed by remote servers).
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                      "content": json.dumps(result)})

        # The loop ended without an answer: too many steps.
        answer = f"(Stopped after {self.max_steps} steps without a final answer.)"
        self.messages.append({"role": "assistant", "content": answer})
        return answer

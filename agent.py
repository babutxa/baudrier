"""The agent: Apertus + tools + the inner loop.

One call to Agent.ask(question) is one user turn. Inside it, repeat until
the model answers without a tool:
  1. the model generates (text, and maybe tool calls);
  2. if there are tool calls: run them, add the results, go back to 1;
  3. if not: this is the final answer, return it.

The conversation (self.messages) is kept between calls, so the model
remembers the earlier questions and answers.
"""
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

DEFAULT_MODEL = "models/apertus-v1.5-8b-text-4bit"

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

TOOLS_PREFIX = "<|tools_prefix|>"
TOOLS_SUFFIX = "<|tools_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


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


class Agent:
    def __init__(self, model_path: str = DEFAULT_MODEL, max_steps: int = 8,
                 verbose: bool = True) -> None:
        self.max_steps = max_steps  # safety: stop if the model never gives a final answer
        self.verbose = verbose      # print each step (tool calls, token counts)
        # load() is declared to return 2 or 3 values (3 with return_config=True),
        # so we take the first two by index: correct for the type checker in both cases.
        loaded = load(model_path)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.tools_suffix_id: int = self.tokenizer.convert_tokens_to_ids(TOOLS_SUFFIX)
        self.messages: list[Message] = []
        self.reset()

    def reset(self) -> None:
        """Forget the conversation (keep only the system prompt)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def generate(self, add_generation_prompt: bool) -> str:
        """Run the model from the conversation; return the new raw text."""
        # Simple version: rebuild and re-tokenize the whole conversation each time.
        # continue_assistant_message: after tool results, keep the assistant turn
        # OPEN, so the model continues it. Without it, the template closes the
        # turn with <|assistant_end|>.
        text = self.tokenizer.apply_chat_template(
            self.messages, tools=TOOL_DESCRIPTIONS, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_assistant_message=not add_generation_prompt,
        )
        prompt_ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False)

        new_ids: list[int] = []
        for response in stream_generate(self.model, self.tokenizer, prompt_ids, max_tokens=500):
            new_ids.append(response.token)
            if response.token == self.tools_suffix_id:
                break  # the model asked for a tool: the harness takes over
            # (stream_generate stops by itself on <|assistant_end|>)

        self.log(f"  [{len(prompt_ids)} tokens in -> {len(new_ids)} tokens out]")
        return self.tokenizer.decode(new_ids)

    def ask(self, question: str) -> str:
        """One user turn: the inner loop. Returns the final answer."""
        self.messages.append({"role": "user", "content": question})

        # First generation: a new assistant turn starts.
        add_generation_prompt = True
        # The tool calls already done in this turn, to detect repetitions.
        done_calls: set[str] = set()

        for step in range(1, self.max_steps + 1):
            self.log(f"--- Step {step}")
            raw = self.generate(add_generation_prompt)
            # After the first step, the assistant turn stays open (the tool results
            # are inside it), so we must not add a new <|assistant_start|>.
            add_generation_prompt = False

            try:
                text, calls = split_output(raw)
            except BadToolCall as bad:
                # The model wrote a broken tool call. Don't stop: close its turn and
                # tell it what went wrong, so it can try again. (A "tool" message
                # needs a valid call before it, so the error comes as a user message.)
                self.log(f"  BAD TOOL CALL: {bad.call_text!r} -> asking the model to retry")
                self.messages.append({"role": "assistant", "content": bad.text})
                self.messages.append({"role": "user", "content":
                                      f"Your tool call was not valid JSON: {bad.call_text}. "
                                      "Please call the tool again with valid JSON."})
                add_generation_prompt = True  # a new assistant turn starts
                continue

            if not calls:
                # No tool call: this is the final answer. Keep it in the
                # conversation, so the next question can refer to it.
                self.messages.append({"role": "assistant", "content": text})
                return text

            if text:
                self.log(f"  TEXT: {text}")

            # Record what the model wrote: its text AND its tool calls.
            self.messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    for name, args in calls
                ],
            })

            # Run each tool and add its result to the conversation.
            for name, args in calls:
                # Protection against loops: a small model sometimes repeats
                # exactly the same call. Don't run it again; tell it to answer.
                key = name + json.dumps(args, sort_keys=True)
                if key in done_calls:
                    result: ToolResult = ("You already called this tool with the same "
                                          "arguments; the result is above. Now answer the user.")
                    self.log(f"  REPEATED: {name}({args}) -> not run again")
                else:
                    done_calls.add(key)
                    result = run_tool(name, args)
                    self.log(f"  TOOL: {name}({args}) -> {result!r}")
                self.messages.append({"role": "tool", "content": json.dumps(result)})

        # The loop ended without an answer: too many steps.
        answer = f"(Stopped after {self.max_steps} steps without a final answer.)"
        self.messages.append({"role": "assistant", "content": answer})
        return answer

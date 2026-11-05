"""The model backends: how the agent asks the model for its next turn.

Both backends have the same interface:

    backend.generate(messages, tool_descriptions) -> ModelTurn(text, tool_calls)

- LocalBackend: Apertus on this Mac with mlx-lm. WE apply the chat template,
  generate tokens and find the tool calls in the raw text
  (<|tools_prefix|>[...]<|tools_suffix|>).
- RemoteBackend: an OpenAI-compatible server (Hugging Face Inference
  Providers, or the CSCS Keera gateway at the hackathon). The SERVER applies
  the chat template and returns the tool calls as structured JSON.

The agent loop (agent.py) does not know which backend it uses.
"""
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

# A message is a dict like {"role": "user", "content": "..."}.
Message = dict[str, Any]

DEFAULT_LOCAL_MODEL = "models/apertus-v1.5-8b-text-4bit"
# Remote defaults: Apertus v1.5 70B through Hugging Face Inference Providers.
# At the hackathon: BAUDRIER_API_BASE=https://gateway.keera.ch/api/v1 and
# BAUDRIER_MODEL=apertus-v1-5-70b (and the team key in BAUDRIER_API_KEY).
DEFAULT_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_REMOTE_MODEL = "swiss-ai/Apertus-v1.5-70B:publicai"


@dataclass
class ToolCall:
    id: str                    # links the call to its result (needed by remote servers)
    name: str                  # e.g. "get_weather"
    arguments: dict[str, Any]  # e.g. {"city": "Zurich"}


@dataclass
class ModelTurn:
    text: str                                          # what the model wrote
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


class BadToolCall(Exception):
    """The model wrote a tool call that is not valid JSON."""

    def __init__(self, text: str, call_text: str) -> None:
        super().__init__(call_text)
        self.text = text            # what the model wrote before the call
        self.call_text = call_text  # the broken call


class Backend(Protocol):
    """What the agent needs from a backend (like an abstract class in C++)."""

    name: str

    def generate(self, messages: list[Message], tool_descriptions: list[dict[str, Any]]) -> ModelTurn:
        ...


def repair_json(text: str) -> str:
    """Repair a known small-model mistake: a tool without arguments written as
    {"list_files": } instead of {"list_files": {}}. In valid JSON a key is
    always followed by a value, so '": }' only appears in this broken case."""
    return re.sub(r'":\s*}', '": {}}', text)


# ---------------------------------------------------------------------------
# Local: Apertus with mlx-lm
# ---------------------------------------------------------------------------

TOOLS_PREFIX = "<|tools_prefix|>"
TOOLS_SUFFIX = "<|tools_suffix|>"
ASSISTANT_END = "<|assistant_end|>"


class LocalBackend:
    def __init__(self, model_path: str = DEFAULT_LOCAL_MODEL) -> None:
        # Imported here, so the remote backend works without loading mlx.
        from mlx_lm import load

        self.name = f"local: {model_path}"
        # load() is declared to return 2 or 3 values (3 with return_config=True),
        # so we take the first two by index: correct for the type checker in both cases.
        loaded = load(model_path)
        self.model, self.tokenizer = loaded[0], loaded[1]
        self.tools_suffix_id: int = self.tokenizer.convert_tokens_to_ids(TOOLS_SUFFIX)
        self.call_count = 0  # to give each tool call a unique id

    def generate(self, messages: list[Message], tool_descriptions: list[dict[str, Any]]) -> ModelTurn:
        from mlx_lm import stream_generate

        # After tool results, the assistant turn is still open: continue it
        # (continue_assistant_message), don't start a new one. Without it, the
        # template closes the turn with <|assistant_end|>.
        continuing = messages[-1]["role"] == "tool"
        # Simple version: rebuild and re-tokenize the whole conversation each time.
        text = self.tokenizer.apply_chat_template(
            messages, tools=tool_descriptions, tokenize=False,
            add_generation_prompt=not continuing,
            continue_assistant_message=continuing,
        )
        prompt_ids: list[int] = self.tokenizer.encode(text, add_special_tokens=False)

        new_ids: list[int] = []
        for response in stream_generate(self.model, self.tokenizer, prompt_ids, max_tokens=500):
            new_ids.append(response.token)
            if response.token == self.tools_suffix_id:
                break  # the model asked for a tool: the harness takes over
            # (stream_generate stops by itself on <|assistant_end|>)

        turn = self.parse(self.tokenizer.decode(new_ids))
        turn.tokens_in, turn.tokens_out = len(prompt_ids), len(new_ids)
        return turn

    def parse(self, raw: str) -> ModelTurn:
        """'Let me check.<|tools_prefix|>[{"get_weather": {"city": "X"}}]<|tools_suffix|>'
        -> ModelTurn('Let me check.', [ToolCall(..., "get_weather", {"city": "X"})])

        Raises BadToolCall if the call cannot be parsed.
        """
        raw = raw.replace(ASSISTANT_END, "")
        if TOOLS_PREFIX not in raw:
            return ModelTurn(raw.strip())
        text, rest = raw.split(TOOLS_PREFIX, 1)
        inside = repair_json(rest.split(TOOLS_SUFFIX, 1)[0])
        try:
            pairs = [(name, args) for call in json.loads(inside) for name, args in call.items()]
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            raise BadToolCall(text.strip(), inside) from e
        calls = []
        for name, args in pairs:
            self.call_count += 1
            calls.append(ToolCall(f"call_{self.call_count}", name, args))
        return ModelTurn(text.strip(), calls)


# ---------------------------------------------------------------------------
# Remote: an OpenAI-compatible server
# ---------------------------------------------------------------------------

class RemoteBackend:
    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        self.name = f"remote: {model} at {api_base}"
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key  # never printed
        self.model = model

    @staticmethod
    def fix_empty_arguments(messages: list[Message]) -> list[Message]:
        """Workaround for a server bug (seen with Apertus 70B via publicai, Nov 2026):
        a tool call with empty arguments ("{}") in the history is rejected with
        "Invalid tool call", although it is valid OpenAI format. Any non-empty
        object is accepted, so we send {"_": null} instead. Only the copy we
        send is changed; the agent's conversation stays as it is."""
        fixed = []
        for m in messages:
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [
                    {**tc, "function": {**tc["function"], "arguments": '{"_": null}'}}
                    if json.loads(tc["function"]["arguments"] or "{}") == {} else tc
                    for tc in m["tool_calls"]
                ]}
            fixed.append(m)
        return fixed

    def generate(self, messages: list[Message], tool_descriptions: list[dict[str, Any]]) -> ModelTurn:
        # The OpenAI format wraps each tool description: {"type": "function", "function": {...}}.
        body = {
            "model": self.model,
            "messages": self.fix_empty_arguments(messages),
            "tools": [{"type": "function", "function": d} for d in tool_descriptions],
            "max_tokens": 1000,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     # Python's default User-Agent ("Python-urllib/3.x") is often
                     # blocked by Cloudflare as a bot, so we name our program.
                     "User-Agent": "baudrier/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            # Show the server's error message (it never contains our key).
            raise RuntimeError(f"Server error {e.code}: {e.read().decode(errors='replace')[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach {self.url}: {e.reason}") from e

        message = data["choices"][0]["message"]
        text = (message.get("content") or "").strip()
        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            function = tc["function"]
            arguments_text = repair_json(function.get("arguments") or "{}")
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError as e:
                raise BadToolCall(text, f'{function["name"]}: {arguments_text}') from e
            calls.append(ToolCall(tc.get("id") or f"call_{i}", function["name"], arguments))

        usage = data.get("usage") or {}
        return ModelTurn(text, calls, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def make_backend(kind: str, model: str | None = None) -> Backend:
    """Create a backend from the command-line choice and the environment.

    Remote settings come from environment variables, so the key is never in
    the code or in git:
        BAUDRIER_API_BASE  server address (default: Hugging Face router)
        BAUDRIER_API_KEY   the key (default for Hugging Face: your `hf auth login` token)
        BAUDRIER_MODEL     the model name (default: Apertus v1.5 70B via publicai)
    """
    if kind == "local":
        return LocalBackend(model or DEFAULT_LOCAL_MODEL)

    api_base = os.environ.get("BAUDRIER_API_BASE", DEFAULT_API_BASE)
    api_key = os.environ.get("BAUDRIER_API_KEY")
    if not api_key and "huggingface.co" in api_base:
        from huggingface_hub import get_token  # the token saved by `hf auth login`
        api_key = get_token()
    if not api_key:
        raise RuntimeError("No API key: set BAUDRIER_API_KEY (or run `hf auth login` for Hugging Face).")
    return RemoteBackend(api_base, api_key, model or os.environ.get("BAUDRIER_MODEL", DEFAULT_REMOTE_MODEL))

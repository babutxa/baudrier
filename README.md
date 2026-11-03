# Baudrier

*Baudrier* is the word we use at home for a climbing harness.

This is a small **agent harness** written from scratch, to learn how AI agents
work. The model is [Apertus](https://huggingface.co/swiss-ai), the open Swiss
LLM, running **locally** on an Apple Silicon Mac with
[MLX](https://github.com/ml-explore/mlx-lm). No agent framework, no API:
just the model, a chat template, some Python tools, and a loop.

An agent is:

```
loop:
    text → LLM → new text
    if the new text asks for a tool:
        run the tool (normal Python code)
        add the result to the text
    else:
        final answer → stop
```

## Files

| File | What it does |
|---|---|
| `step1_template.py` | Shows the raw text the model reads: special tokens, roles and tool descriptions. |
| `step2_one_cycle.py` | One full agent cycle: the model asks for `get_weather`, the harness calls the real weather API ([Open-Meteo](https://open-meteo.com)), the model answers. |
| `baudrier.py` | **The chat**, like Claude Code: you write, the agent answers (using tools), you write again. It remembers the conversation. |
| `agent.py` | The `Agent` class: the model, the inner loop (`ask()`: tool calls until the answer), tool-call parsing and the protections. |
| `step3_agent_loop.py` | The agent loop for one question (uses `agent.py`). |
| `tools.py` | The tools: `get_weather` (conditions, rain, clouds, sunshine, UV), `list_files`, `read_file` (only inside `workspace/`). Each tool is a typed function; the description the model sees is generated from the type hints and the docstring. |
| `workspace/` | Example notes for the file tools. |
| `convert_v15_to_mlx.py` | Extracts the text model from the official Apertus v1.5 8B and converts it to MLX 4-bit. |
| `experiments/` | Tests used to compare Apertus v1.0 and v1.5 on tool calling. |

## What we learned

With the same tool and question, the model must choose the first token of its answer.
`<|tools_prefix|>` starts a tool call:

| Model (4-bit) | P(`<\|tools_prefix\|>`) | Result |
|---|---|---|
| Apertus v1.0 8B Instruct | ~3%, rank 4 | writes Python code instead of calling the tool |
| Apertus v1.5 8B | ~12%, rank 2 (100% with one example) | correct tool call |

Other lessons:

- **Tool results need the right template option.** The Apertus templates close
  the assistant turn after a tool result, unless you pass
  `continue_assistant_message=True`. Without it, the model continues from a
  closed turn and the next tool call breaks.
- **The system prompt matters.** Without instructions, the model asked the user
  to check the weather; with "never ask the user to look up something that one
  of your tools can find", it called the tool itself.
- **Small models make reasoning mistakes, and quantization matters.** In the
  notes task ("Where am I going today, and do I need an umbrella?"), the note says
  "Day trip to Lucerne ... 10:00 train from Zurich":

  | Apertus v1.5 8B | Size | Tool calls | Result |
  |---|---|---|---|
  | 4-bit | 4.3 GB | list_files → read today.md → read shopping.txt → weather **Zurich** | wrong city |
  | 8-bit | 8.0 GB | list_files → read today.md → weather **Lucerne** | correct |

  One task and one greedy run each: a strong hint, not a proof. The 8-bit
  model is about 2× slower, and it is not better at everything: it once wrote
  a tool call without arguments as `{"list_files": }` (invalid JSON).
- **The harness must tolerate small format errors.** `{"list_files": }` is
  repaired to `{"list_files": {}}` before parsing. If a call is still invalid,
  the harness sends the error back to the model and lets it try again.
- **Without context, the model invents.** Asked "Where am I going today? Do I
  need sunglasses?", it called `get_weather("New York")`. Later, it kept
  reusing that old New York result even after finding Lucerne in the notes.
  Telling it in the system prompt that the user's plans are in the notes fixed it.
- **Give the tools the data the question needs.** For sunglasses, rain
  probability is not enough: `get_weather` now also returns cloud cover,
  sunshine hours and the UV index.
- **More prompt rules can make a small model worse.** A rule "never state
  facts before the tool result" did not stop it from guessing the weather
  ("sunny, 22°C") before calling the tool, and made it call `get_weather`
  in a loop. We removed the rule and handle loops in the harness instead:
  a repeated identical tool call is not run again.
- **Text written before a tool call is a guess.** Trust the tool results and
  the final answer, not what the model says before the call.

## Setup

Requirements: an Apple Silicon Mac (16 GB RAM is enough) and Python 3.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Models

The models are not in this repository (several GB). Create them in `models/`:

**Apertus v1.0** (open access):

```
.venv/bin/mlx_lm.convert --hf-path swiss-ai/Apertus-8B-Instruct-2509 \
    -q --q-bits 4 --q-group-size 64 --mlx-path models/apertus-v1.0-8b-4bit
```

**Apertus v1.5** (gated: accept the conditions on
[the model page](https://huggingface.co/swiss-ai/Apertus-v1.5-8B), then run `.venv/bin/hf auth login`):

```
.venv/bin/python convert_v15_to_mlx.py
```

The script downloads the needed files (~16 GB) if they are not in the Hugging Face
cache yet, then creates `models/apertus-v1.5-8b-text-4bit/`. For the 8-bit version
(8 GB, fewer mistakes, slower), run `.venv/bin/python convert_v15_to_mlx.py 8`.

This is a text-only conversion: image and audio input are removed. It is my own
adaptation, not an official method.

## Run

Run the scripts from the project folder.

**Chat with the agent:**

```
.venv/bin/python baudrier.py
.venv/bin/python baudrier.py --model models/apertus-v1.5-8b-text-8bit
.venv/bin/python baudrier.py --quiet     # hide the tool calls
```

In the chat, `/reset` forgets the conversation and `/exit` (or Ctrl+D) quits.

**The learning steps:**

```
.venv/bin/python step1_template.py
.venv/bin/python step2_one_cycle.py "Do I need an umbrella in Barcelona today?"
.venv/bin/python step3_agent_loop.py "Look at my notes. Where am I going today, and do I need an umbrella?"
.venv/bin/python step3_agent_loop.py --model models/apertus-v1.5-8b-text-8bit
```

## Type checking

All scripts have type hints. To check them (optional):

```
.venv/bin/pip install mypy
.venv/bin/mypy --ignore-missing-imports tools.py agent.py baudrier.py
```

Check the standalone scripts one by one (they define the same names).

## License

MIT. The Apertus models have their own license (Apache 2.0) and acceptable use policy.

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
| `convert_v15_to_mlx.py` | Extracts the text model from the official Apertus v1.5 8B and converts it to MLX 4-bit. |
| `experiments/` | Tests used to compare Apertus v1.0 and v1.5 on tool calling. |

## What we learned

With the same tool and question, the model must choose the first token of its answer.
`<|tools_prefix|>` starts a tool call:

| Model (4-bit) | P(`<\|tools_prefix\|>`) | Result |
|---|---|---|
| Apertus v1.0 8B Instruct | ~3%, rank 4 | writes Python code instead of calling the tool |
| Apertus v1.5 8B | ~12%, rank 2 (100% with one example) | correct tool call |

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
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('swiss-ai/Apertus-v1.5-8B', allow_patterns=['model-apertus-model-*.safetensors', '*.json', 'chat_template.jinja', 'tokenizer*'])"
.venv/bin/python convert_v15_to_mlx.py
```

This is a text-only conversion: image and audio input are removed. It is my own
adaptation, not an official method.

## Run

Run the scripts from the project folder:

```
.venv/bin/python step1_template.py
.venv/bin/python step2_one_cycle.py "Do I need an umbrella in Barcelona today?"
```

## License

MIT. The Apertus models have their own license (Apache 2.0) and acceptable use policy.

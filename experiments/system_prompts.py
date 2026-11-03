"""Try several system prompts and see which ones make Apertus call the tool.

The question and the tool are always the same. Only the system message
changes. This shows how much the model depends on the exact input text.

Usage:
    python experiments/system_prompts.py                                    # v1.0
    python experiments/system_prompts.py models/apertus-v1.5-8b-text-4bit   # v1.5
"""

# sys gives access to the command-line arguments (sys.argv).
import sys

# load: reads the model weights and the tokenizer from disk.
# stream_generate: generates tokens one by one, and gives us each new token.
from mlx_lm import load, stream_generate

# Choose the model from the command line (default: v1.0).
MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/apertus-v1.0-8b-4bit"
print("Model:", MODEL, "\n")

# Load the model (the neural network) and its tokenizer (text <-> token IDs).
model, tokenizer = load(MODEL)

# The ID of the special token that ends a tool call. We stop there,
# because in a real harness our code would now run the tool.
SUFFIX_ID = tokenizer.convert_tokens_to_ids("<|tools_suffix|>")

# Description of the tool, in the format the Apertus chat template expects.
# The model only sees this text. There is no real function behind it here.
TOOLS = [
    {
        "name": "get_weather",                               # the name the model must write
        "description": "Get the current weather for a city.",  # helps the model decide when to use it
        "parameters": {                                      # the inputs, in JSON Schema format
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Barcelona"}
            },
            "required": ["city"],                            # "city" is mandatory
        },
    }
]

# The question. It is the same for all variants.
QUESTION = "Do I need an umbrella in Barcelona today?"

# The variants to compare. Each one is a list with 0 or 1 system message.
variants = {
    # A: a short, generic system message.
    "A: helpful assistant": [
        {"role": "system", "content": "You are a helpful assistant."},
    ],
    # B: no system message. The template then adds its own default one
    #    (for v1.0: "You are Apertus, a helpful assistant created by the SwissAI initiative...").
    "B: no system (template default)": [],
    # C: tell the model explicitly to use tools, and not to write code.
    "C: tell it to use tools": [
        {"role": "system", "content": "You are a helpful assistant. You have access to tools. "
                                      "When a tool can help answer the question, call it instead "
                                      "of guessing or writing code."},
    ],
}

# .items() gives each (name, list of system messages) pair.
for name, system_messages in variants.items():
    # The conversation: the system message (if any), then the user question.
    msgs = system_messages + [{"role": "user", "content": QUESTION}]

    # Build the raw text with the chat template (special tokens + tool list).
    # add_generation_prompt=True adds <|assistant_start|> at the end:
    # "now it is the assistant's turn to write".
    text = tokenizer.apply_chat_template(
        msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )

    # Text -> list of token IDs.
    # add_special_tokens=False because the text already contains <s>.
    ids = tokenizer.encode(text, add_special_tokens=False)

    out = []  # the new token IDs written by the model
    # Generate up to 120 tokens. By default this is greedy: always the most
    # probable token. It stops by itself at the end token <|assistant_end|>.
    for r in stream_generate(model, tokenizer, ids, max_tokens=120):
        out.append(r.token)
        # If the model closes a tool call, stop here.
        if r.token == SUFFIX_ID:
            break

    # Show the variant name and the raw output (with special tokens).
    # !r prints the text with quotes and visible \n, so we see exactly what was written.
    print(f"--- {name}\n{tokenizer.decode(out)!r}\n")

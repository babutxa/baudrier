"""Few-shot test: does an example help Apertus to call a tool?

We put one example conversation before the real question. In the
example, the assistant uses the tool. Then we check if the model
copies this behaviour ("in-context learning").
"""

# Any: "any type", for values whose type we don't describe in detail.
from typing import Any, cast

# mlx.core is Apple's array library (like numpy, but runs on the M1 GPU).
import mlx.core as mx

# load: reads the model weights and the tokenizer from disk.
# stream_generate: generates tokens one by one, and gives us each new token.
from mlx_lm import load, stream_generate

# Type alias: a message is a dict like {"role": "user", "content": "..."}.
Message = dict[str, Any]

# Load the model (the neural network) and its tokenizer (text <-> token IDs).
# load() is declared to return 2 or 3 values (3 with return_config=True),
# so we take the first two by index: correct for the type checker in both cases.
# loaded = load("models/apertus-v1.0-8b-4bit")
loaded = load("models/apertus-v1.5-8b-text-4bit")
model, tokenizer = loaded[0], loaded[1]

# Description of the tool, in the format the Apertus chat template expects.
# The model only sees this text. There is no real function behind it here.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",  # the name the model must write
        "description": "Get the current weather for a city.",  # helps the model decide when to use it
        "parameters": {  # the inputs, in JSON Schema format
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Barcelona"}
            },
            "required": ["city"],  # "city" is mandatory
        },
    }
]

# The token IDs of the two special tokens that surround a tool call:
#   <|tools_prefix|>[{"get_weather": {"city": "..."}}]<|tools_suffix|>
# We need the IDs because the model works with numbers, not text.
PREFIX_ID: int = tokenizer.convert_tokens_to_ids("<|tools_prefix|>")
SUFFIX_ID: int = tokenizer.convert_tokens_to_ids("<|tools_suffix|>")

# The example conversation (the "shot" in "few-shot").
# It shows the model a full tool cycle: question -> tool call -> result -> answer.
example: list[Message] = [
    # 1) A user asks something that needs the tool.
    {"role": "user", "content": "Is it cold in Oslo right now?"},
    # 2) The assistant does not answer with text: it calls the tool.
    #    "arguments" must be a string with JSON inside (the template pastes it as text).
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
            },
        ],
    },
    # 3) The tool result (invented by us for the example).
    {
        "role": "tool",
        "content": '"Oslo (Norway): overcast, -3°C, wind 12 km/h, rain probability today 10%"',
    },
    # 4) The assistant uses the result to answer.
    {"role": "assistant", "content": "Yes, it is cold in Oslo: -3°C and overcast."},
]

# Three test questions:
#   - two that need the weather tool (one in English, one in Catalan),
#   - one that does NOT need it (to check the model does not call tools for everything).
questions: list[str] = [
    "Do I need an umbrella in Barcelona today?",
    "Quin temps fa a Girona?",
    "What is the capital of France?",
]

for q in questions:
    # The full conversation: the example first, then the real question.
    msgs: list[Message] = example + [{"role": "user", "content": q}]

    # Build the raw text with the chat template (special tokens + tool list).
    # add_generation_prompt=True adds <|assistant_start|> at the end:
    # "now it is the assistant's turn to write".
    text = tokenizer.apply_chat_template(
        msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )

    # Text -> list of token IDs.
    # add_special_tokens=False because the text already contains <s>.
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

    # --- Part 1: probability of the first token ---------------------------
    # Run the network once on all the tokens.
    # mx.array([ids]) makes a batch of 1 sequence (shape: 1 x number_of_tokens).
    logits = model(mx.array([ids]))
    # logits has one score per vocabulary token, for each position.
    # [0, -1] = first (only) sequence, last position = the NEXT token to predict.
    last_logits = logits[0, -1]
    # softmax turns the scores into probabilities (all positive, sum = 1).
    # float32 gives a more precise result than the model's 16-bit numbers.
    probs = mx.softmax(last_logits.astype(mx.float32))
    # The probability that the next token is <|tools_prefix|> (start of a tool call).
    # .item() is declared as int | float | complex. A probability is a float:
    # cast() tells the type checker (like static_cast, but nothing changes at run time).
    p = cast(float, probs[PREFIX_ID].item())

    # --- Part 2: let the model write -----------------------------------------
    out: list[int] = []  # the new token IDs written by the model
    # Generate up to 80 tokens. By default this is greedy: always the most
    # probable token. It stops by itself at the end token <|assistant_end|>.
    for r in stream_generate(model, tokenizer, ids, max_tokens=80):
        out.append(r.token)
        # If the model closes a tool call, stop: in a real harness, this is
        # where our code would run the tool.
        if r.token == SUFFIX_ID:
            break

    # Show the question, the probability, and the raw output (with special tokens).
    # !r prints the text with quotes and visible \n, so we see exactly what was written.
    print(f"--- {q}\n   P(<|tools_prefix|>) = {p:.1%}\n   {tokenizer.decode(out)!r}\n")

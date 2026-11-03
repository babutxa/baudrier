"""Show the model's probabilities for the FIRST token of its answer.

The model does not write anything here. We run the network only once
and look at its "decision" for the first token: does it start a tool
call (<|tools_prefix|>) or does it start writing text?

Usage:
    python experiments/first_token.py                                    # v1.0
    python experiments/first_token.py models/apertus-v1.5-8b-text-4bit   # v1.5
"""

# sys gives access to the command-line arguments (sys.argv).
import sys

# Any: "any type", for values whose type we don't describe in detail.
from typing import Any, cast

# mlx.core is Apple's array library (like numpy, but runs on the M1 GPU).
import mlx.core as mx

# load: reads the model weights and the tokenizer from disk.
from mlx_lm import load

# Choose the model from the command line.
# sys.argv is the list of words in the command:
#   python experiments/first_token.py models/apertus-v1.5-8b-text-4bit
#   sys.argv[0] = "experiments/first_token.py", sys.argv[1] = "models/apertus-..."
# If no model is given, use v1.0.
MODEL: str = sys.argv[1] if len(sys.argv) > 1 else "models/apertus-v1.0-8b-4bit"
print("Model:", MODEL)

# Load the model (the neural network) and its tokenizer (text <-> token IDs).
# load() is declared to return 2 or 3 values (3 with return_config=True),
# so we take the first two by index: correct for the type checker in both cases.
loaded = load(MODEL)
model, tokenizer = loaded[0], loaded[1]

# Description of the tool, in the format the Apertus chat template expects.
# The model only sees this text. There is no real function behind it here.
TOOLS: list[dict[str, Any]] = [
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

# The ID of the special token that starts a tool call.
# If the model chooses this token first, it wants to call a tool.
PREFIX_ID: int = tokenizer.convert_tokens_to_ids("<|tools_prefix|>")

# Run the test twice: with Deliberation (thinking mode) disabled, then enabled.
for think in (False, True):
    # Only one user message. No system message, so the template adds its default one.
    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Do I need an umbrella in Barcelona today?"}]

    # Build the raw text with the chat template.
    # enable_thinking is passed to the template: it writes
    # "Deliberation: enabled" or "Deliberation: disabled" in the developer block.
    text = tokenizer.apply_chat_template(
        msgs, tools=TOOLS, tokenize=False,
        add_generation_prompt=True,  # ends with <|assistant_start|>: the assistant's turn
        enable_thinking=think,
    )

    # Text -> list of token IDs -> MLX array with a batch of 1 sequence.
    # Shape: (1, number_of_tokens). add_special_tokens=False because <s> is already in the text.
    ids = mx.array([tokenizer.encode(text, add_special_tokens=False)])

    # Run the network ONCE on all the tokens (no generation loop).
    # The output has one score ("logit") per vocabulary token, for every position:
    # shape (1, number_of_tokens, 131072).
    # [0, -1] = first (only) sequence, last position = the scores for the NEXT token.
    logits = model(ids)[0, -1]

    # softmax turns the 131072 scores into probabilities (all positive, sum = 1).
    # float32 gives a more precise result than the model's 16-bit numbers.
    probs = mx.softmax(logits.astype(mx.float32))

    # The 8 most probable tokens.
    # argsort sorts from smallest to largest, so we sort -probs (negative)
    # to get the largest first. [:8] keeps the first 8 token IDs.
    # .tolist() is declared to return a number or nested lists; for a 1D array
    # of indices it is a list[int]. cast() tells the type checker
    # (like static_cast, but nothing changes at run time).
    top = cast(list[int], mx.argsort(-probs)[:8].tolist())

    # The rank of <|tools_prefix|> among all 131072 tokens.
    # (probs > probs[PREFIX_ID]) is True for each token MORE probable than it.
    # Counting them (+1) gives its position: 1 = most probable.
    rank: int = cast(int, (probs > probs[PREFIX_ID]).sum().item()) + 1

    # Print the top 8: probability and the token as text.
    # .item() converts a 1-value MLX array into a normal Python number.
    # !r shows the token with quotes, so spaces like ' Let' are visible.
    print(f"Deliberation {'enabled' if think else 'disabled'}:")
    for i in top:
        print(f"   {probs[i].item():7.2%}  {tokenizer.decode([i])!r}")

    # Print the probability and rank of <|tools_prefix|>, even if it is not in the top 8.
    print(f"   <|tools_prefix|>: {probs[PREFIX_ID].item():.4%}  (rank {rank} of {probs.size})\n")

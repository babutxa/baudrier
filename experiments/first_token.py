"""Show the model's probabilities for the FIRST token of its answer."""
import mlx.core as mx
from mlx_lm import load

import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/apertus-v1.0-8b-4bit"
print("Model:", MODEL)
model, tokenizer = load(MODEL)
TOOLS = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Barcelona"}}, "required": ["city"]}}]
PREFIX_ID = tokenizer.convert_tokens_to_ids("<|tools_prefix|>")

for think in (False, True):
    msgs = [{"role": "user", "content": "Do I need an umbrella in Barcelona today?"}]
    text = tokenizer.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=think)
    ids = mx.array([tokenizer.encode(text, add_special_tokens=False)])
    logits = model(ids)[0, -1]                    # scores for the next token (131073 values)
    probs = mx.softmax(logits.astype(mx.float32))  # -> probabilities
    top = mx.argsort(-probs)[:8].tolist()
    rank = int((probs > probs[PREFIX_ID]).sum().item()) + 1
    print(f"Deliberation {'enabled' if think else 'disabled'}:")
    for i in top:
        print(f"   {probs[i].item():7.2%}  {tokenizer.decode([i])!r}")
    print(f"   <|tools_prefix|>: {probs[PREFIX_ID].item():.4%}  (rank {rank} of {probs.size})\n")

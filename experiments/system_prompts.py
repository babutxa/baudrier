"""Try several system prompts and see which ones make Apertus call the tool."""
import importlib.util, sys
sys.argv = ["x", "-"]  # dummy
from mlx_lm import load, stream_generate

MODEL = "models/apertus-v1.0-8b-4bit"
model, tokenizer = load(MODEL)
SUFFIX_ID = tokenizer.convert_tokens_to_ids("<|tools_suffix|>")
TOOLS = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. Barcelona"}}, "required": ["city"]}}]
Q = "Do I need an umbrella in Barcelona today?"

variants = {
    "A: helpful assistant": [{"role": "system", "content": "You are a helpful assistant."}],
    "B: no system (template default)": [],
    "C: tell it to use tools": [{"role": "system", "content": "You are a helpful assistant. You have access to tools. When a tool can help answer the question, call it instead of guessing or writing code."}],
}
for name, sys_msgs in variants.items():
    msgs = sys_msgs + [{"role": "user", "content": Q}]
    text = tokenizer.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    for r in stream_generate(model, tokenizer, ids, max_tokens=120):
        out.append(r.token)
        if r.token == SUFFIX_ID: break
    print(f"--- {name}\n{tokenizer.decode(out)!r}\n")

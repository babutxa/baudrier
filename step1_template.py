"""Step 1: show the raw text that Apertus receives.

No model is run here. We only load the tokenizer and ask it to turn a
structured conversation (roles + tool list) into the single long text
that the network actually reads.
"""
from transformers import AutoTokenizer

MODEL = "models/apertus-v1.0-8b-4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# A tool description: name, what it does, and its parameters.
# (Apertus's template expects this flat shape.)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Barcelona"}
            },
            "required": ["city"],
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Do I need an umbrella in Barcelona today?"},
]


def show(title, messages, add_generation_prompt):
    # Turn the structured conversation into the raw text the network reads.
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(text)
    print(f"\n({len(tokenizer.encode(text))} tokens)\n")


# 1) What the model receives before answering.
show("BEFORE: the input, ending where the assistant starts writing",
     messages, add_generation_prompt=True)

# 2) The same conversation after the model asked for a tool and the
#    harness added the result. The model would continue from here.
messages_after = messages + [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Barcelona"}'},
        }],
    },
    {"role": "tool", "content": '"Rain, 80% probability, 14C"'},
]
show("AFTER: tool call + tool result added by the harness",
     messages_after, add_generation_prompt=False)

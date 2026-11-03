"""Step 2: one full agent cycle with Apertus.

  1. Build the input text with the chat template (as in step 1).
  2. Let Apertus generate, token by token, until it either:
       - asks for a tool  (<|tools_suffix|>), or
       - finishes its answer (<|assistant_end|>).
  3. If it asked for a tool: parse the request, run the REAL Python
     function, add the result, and let Apertus continue.
"""
import json
import sys
import urllib.parse
import urllib.request

from mlx_lm import load, stream_generate

MODEL = "models/apertus-v1.5-8b-text-4bit"


# ---------------------------------------------------------------------------
# The real tool: plain Python, nothing to do with AI.
# ---------------------------------------------------------------------------

def fetch_json(url, params):
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params)) as r:
        return json.load(r)


# A few WMO weather codes used by Open-Meteo, as words.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 95: "thunderstorm",
}


def get_weather(city):
    # 1) City name -> coordinates.
    places = fetch_json("https://geocoding-api.open-meteo.com/v1/search",
                        {"name": city, "count": 1})
    if not places.get("results"):
        return f"Error: city '{city}' not found."
    place = places["results"][0]

    # 2) Coordinates -> current weather and today's rain probability.
    w = fetch_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "daily": "precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 1,
    })
    now = w["current"]
    return (f"{place['name']} ({place.get('country', '')}): "
            f"{WEATHER_CODES.get(now['weather_code'], 'code ' + str(now['weather_code']))}, "
            f"{now['temperature_2m']}°C, wind {now['wind_speed_10m']} km/h, "
            f"rain probability today {w['daily']['precipitation_probability_max'][0]}%")


# The harness's link between tool NAMES (what the model writes)
# and tool FUNCTIONS (what actually runs).
TOOL_FUNCTIONS = {"get_weather": get_weather}

# What the model is told about the tools (same as in step 1).
TOOL_DESCRIPTIONS = [
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


# ---------------------------------------------------------------------------
# The model side.
# ---------------------------------------------------------------------------

print("Loading Apertus...")
model, tokenizer = load(MODEL)

TOOLS_PREFIX = "<|tools_prefix|>"
TOOLS_SUFFIX = "<|tools_suffix|>"
TOOLS_SUFFIX_ID = tokenizer.convert_tokens_to_ids(TOOLS_SUFFIX)


def generate(messages, add_generation_prompt):
    """Run the model from the conversation; return the new raw text."""
    text = tokenizer.apply_chat_template(
        messages, tools=TOOL_DESCRIPTIONS, tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    # The text already contains <s>, so don't add special tokens again.
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)

    new_ids = []
    for response in stream_generate(model, tokenizer, prompt_ids, max_tokens=300):
        new_ids.append(response.token)
        if response.token == TOOLS_SUFFIX_ID:
            break  # the model asked for a tool: the harness takes over
        # (stream_generate stops by itself on <|assistant_end|>, the eos token)

    raw = tokenizer.decode(new_ids)
    print(f"  [{len(prompt_ids)} tokens in -> {len(new_ids)} tokens out, "
          f"{response.generation_tps:.1f} tokens/s]")
    return raw


def parse_tool_calls(raw):
    """'<|tools_prefix|>[{"get_weather": {"city": "X"}}]<|tools_suffix|>'
    -> [("get_weather", {"city": "X"})]"""
    inside = raw.split(TOOLS_PREFIX, 1)[1].split(TOOLS_SUFFIX, 1)[0]
    return [(name, args) for call in json.loads(inside) for name, args in call.items()]


# ---------------------------------------------------------------------------
# One cycle.
# ---------------------------------------------------------------------------

question = sys.argv[1] if len(sys.argv) > 1 else "Do I need an umbrella in Barcelona today?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": question},
]
print(f"\nUSER: {question}\n")

print("1) Apertus generates:")
raw = generate(messages, add_generation_prompt=True)
print(f"  RAW OUTPUT: {raw!r}\n")

if TOOLS_PREFIX not in raw:
    print("Apertus answered directly, without using a tool.")
    sys.exit()

print("2) The harness runs the tool(s):")
calls = parse_tool_calls(raw)
tool_messages = []
for name, args in calls:
    result = TOOL_FUNCTIONS[name](**args)
    print(f"  {name}({args}) -> {result!r}")
    tool_messages.append({"role": "tool", "content": json.dumps(result)})

# Record the model's request and the results in the conversation.
messages.append({
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for name, args in calls
    ],
})
messages.extend(tool_messages)

print("\n3) Apertus continues, now with the result:")
# The assistant turn is still open, so no new <|assistant_start|>.
raw = generate(messages, add_generation_prompt=False)
print(f"  RAW OUTPUT: {raw!r}\n")
print("ASSISTANT:", raw.replace("<|assistant_end|>", "").strip())

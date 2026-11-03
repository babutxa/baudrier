"""The tools of the agent: plain Python functions, nothing to do with AI.

Each tool is written ONCE, as a typed Python function with a docstring.
The description the model sees (JSON Schema) is generated from the type
hints and the docstring, so the function and its description cannot
disagree.

The docstring format matters: the first line is the tool description,
and the "Args:" section gives one description per parameter.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from transformers.utils import get_json_schema

# The only folder the file tools can access.
WORKSPACE: Path = Path(__file__).parent / "workspace"


# ---------------------------------------------------------------------------
# Weather (Open-Meteo: free, no account or key)
# ---------------------------------------------------------------------------

def fetch_json(url: str, params: dict[str, Any]) -> Any:
    with urllib.request.urlopen(url + "?" + urllib.parse.urlencode(params)) as r:
        return json.load(r)


# A few WMO weather codes used by Open-Meteo, as words.
WEATHER_CODES: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
    82: "violent rain showers", 95: "thunderstorm",
}


def get_weather(city: str) -> str:
    """Get the current weather and today's rain probability for a city.

    Args:
        city: City name, e.g. Barcelona
    """
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


# ---------------------------------------------------------------------------
# Files (only inside WORKSPACE)
# ---------------------------------------------------------------------------

def list_files() -> list[str]:
    """List the files in the user's notes folder."""
    return sorted(p.name for p in WORKSPACE.iterdir() if p.is_file())


def read_file(path: str) -> str:
    """Read a file from the user's notes folder.

    Args:
        path: File name, as given by list_files
    """
    # Safety: the model chooses the path, so it could ask for "../../.ssh/id_rsa".
    # resolve() turns the path into a real absolute path (".." removed),
    # then we check it is still inside the workspace.
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE.resolve()):
        return f"Error: '{path}' is outside the workspace."
    if not target.is_file():
        return f"Error: file '{path}' not found. Use list_files to see the files."
    return target.read_text()


# ---------------------------------------------------------------------------
# For the harness
# ---------------------------------------------------------------------------

# The type of a tool result: what the functions above can return.
ToolResult = str | list[str]

# The tools given to the agent.
TOOLS: list[Callable[..., ToolResult]] = [get_weather, list_files, read_file]

# Tool NAME (what the model writes) -> FUNCTION (what really runs).
TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {f.__name__: f for f in TOOLS}

# What the model is told about the tools, generated from the type hints and
# docstrings. get_json_schema returns {"type": "function", "function": {...}};
# the Apertus template wants the inner part (name, description, parameters).
TOOL_DESCRIPTIONS: list[dict[str, Any]] = [get_json_schema(f)["function"] for f in TOOLS]

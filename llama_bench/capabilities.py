import os
import tomllib
from typing import Literal, TypeAlias, TypedDict, cast

from llama_bench.results import PROJECT_ROOT

CAPABILITIES_FILE = os.path.join(PROJECT_ROOT, "model-capabilities.toml")

CapabilityName: TypeAlias = Literal[
    "coding",
    "reasoning",
    "tools",
    "vision",
    "writing",
    "multilingual",
]

CAPABILITY_NAMES: tuple[CapabilityName, ...] = (
    "coding",
    "reasoning",
    "tools",
    "vision",
    "writing",
    "multilingual",
)


class ModelCapabilities(TypedDict):
    coding: int
    reasoning: int
    tools: int
    vision: int
    writing: int
    multilingual: int


CapabilityMap: TypeAlias = dict[str, ModelCapabilities]


def load_capabilities(path: str = CAPABILITIES_FILE) -> CapabilityMap:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    capabilities: CapabilityMap = {}
    for group, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"capability group {group!r} must be a table")
        capabilities[group] = _parse_model_capabilities(group, cast(dict[str, object], values))
    return capabilities


def _parse_model_capabilities(group: str, values: dict[str, object]) -> ModelCapabilities:
    missing = [name for name in CAPABILITY_NAMES if name not in values]
    if missing:
        raise ValueError(f"capability group {group!r} is missing: {', '.join(missing)}")

    unknown = sorted(set(values) - set(CAPABILITY_NAMES))
    if unknown:
        raise ValueError(f"capability group {group!r} has unknown fields: {', '.join(unknown)}")

    parsed: dict[str, int] = {}
    for name in CAPABILITY_NAMES:
        value = values[name]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
            raise ValueError(f"capability {group}.{name} must be an integer from 0 to 5")
        parsed[name] = value
    return cast(ModelCapabilities, parsed)


def capability_label(value: int) -> str:
    labels = {
        0: "none",
        1: "weak",
        2: "usable",
        3: "good",
        4: "strong",
        5: "excellent",
    }
    try:
        return labels[value]
    except KeyError as exc:
        raise ValueError(f"capability value must be an integer from 0 to 5: {value!r}") from exc

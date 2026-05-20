from pathlib import Path

import pytest

from llama_bench.capabilities import capability_label, load_capabilities
from llama_bench.results import load_models


def test_load_capabilities_parses_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "model-capabilities.toml"
    path.write_text(
        "[foo]\n"
        "coding = 5\n"
        "reasoning = 4\n"
        "tools = 3\n"
        "vision = 2\n"
        "writing = 1\n"
        "multilingual = 0\n",
        encoding="utf-8",
    )

    assert load_capabilities(str(path)) == {
        "foo": {
            "coding": 5,
            "reasoning": 4,
            "tools": 3,
            "vision": 2,
            "writing": 1,
            "multilingual": 0,
        }
    }


def test_load_capabilities_rejects_missing_unknown_and_invalid_values(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    missing.write_text("[foo]\ncoding = 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_capabilities(str(missing))

    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        "[foo]\n"
        "coding = 5\n"
        "reasoning = 4\n"
        "tools = 3\n"
        "vision = 2\n"
        "writing = 1\n"
        "multilingual = 0\n"
        "speed = 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_capabilities(str(unknown))

    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        "[foo]\n"
        "coding = 6\n"
        "reasoning = 4\n"
        "tools = 3\n"
        "vision = 2\n"
        "writing = 1\n"
        "multilingual = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integer from 0 to 5"):
        load_capabilities(str(invalid))


def test_capability_label_formats_values() -> None:
    assert [capability_label(value) for value in range(6)] == [
        "none",
        "weak",
        "usable",
        "good",
        "strong",
        "excellent",
    ]
    with pytest.raises(ValueError, match="0 to 5"):
        capability_label(6)


def test_default_capabilities_cover_all_model_groups() -> None:
    capabilities = load_capabilities()
    groups = {group for _, _, group, _ in load_models()}

    assert groups - set(capabilities) == set()

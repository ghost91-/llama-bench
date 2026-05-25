import pytest

from llama_bench.model_identity import (
    ModelIdentity,
    display_name_from_repo,
    identity_from_tag,
    render_model_tag,
    result_key_from_parts,
    result_key_from_tag,
)


def test_identity_from_tag_parses_repo_quant_and_provider() -> None:
    identity = identity_from_tag("unsloth/Qwen3.5-9B-GGUF:Q4_K_M")

    assert identity == ModelIdentity(
        repo="unsloth/Qwen3.5-9B-GGUF",
        quant="Q4_K_M",
        provider="unsloth",
    )


def test_identity_from_tag_allows_missing_quant_only_when_requested() -> None:
    with pytest.raises(ValueError):
        identity_from_tag("unsloth/Foo-GGUF")

    identity = identity_from_tag("unsloth/Foo-GGUF", require_quant=False)

    assert identity.repo == "unsloth/Foo-GGUF"
    assert identity.quant == ""
    assert identity.provider == "unsloth"


@pytest.mark.parametrize(
    ("tag", "display"),
    [
        ("google_gemma-4-26B-A4B-GGUF:Q4_K_M", "gemma-4-26B-A4B"),
        ("unsloth/Qwen_MyModel-GGUF:Q4_K_M", "MyModel"),
        ("unsloth/qwen_MyModel-GGUF:Q4_K_M", "MyModel"),
        ("zai-org_GLM-4.6-GGUF:Q4_K_M", "GLM-4.6"),
        ("mistralai_Mistral-Small-it:Q4_K_M", "Mistral-Small"),
        ("nvidia_Nemotron-GGUF:Q4_K_M", "Nemotron"),
        ("NVIDIA-Nemotron-GGUF:Q4_K_M", "Nemotron"),
        ("mudler/Qwen3.6-35B-A3B-APEX-GGUF:APEX-I-Compact", "Qwen3.6-35B-A3B"),
    ],
)
def test_display_name_normalises_known_prefixes_and_suffixes(tag: str, display: str) -> None:
    repo, _, _ = tag.partition(":")
    assert display_name_from_repo(repo) == display


def test_render_model_tag_and_result_keys() -> None:
    identity = identity_from_tag("bartowski/Foo-GGUF:Q5_K_M")
    tag = render_model_tag("bartowski/Foo-GGUF", "Q5_K_M")

    assert tag == "bartowski/Foo-GGUF:Q5_K_M"
    assert identity.result_key == ("Foo", "Q5_K_M", "bartowski")
    assert result_key_from_tag(tag) == identity.result_key
    assert result_key_from_parts("bartowski/Foo-GGUF", "Q5_K_M") == identity.result_key

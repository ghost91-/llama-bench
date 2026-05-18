from pathlib import Path

from pytest import MonkeyPatch

import llama_bench.gguf_cache as gguf_cache
import llama_bench.gguf_utils as gguf_utils


class FakeField:
    def __init__(self, value: object, exc: Exception | None = None) -> None:
        self.value = value
        self.exc = exc

    def contents(self) -> object:
        if self.exc is not None:
            raise self.exc
        return self.value


class FakeReader:
    def __init__(self, fields: dict[str, FakeField]) -> None:
        self.fields = fields

    def get_field(self, key: str) -> FakeField | None:
        return self.fields.get(key)


def _patch_reader(monkeypatch: MonkeyPatch, reader: FakeReader) -> None:
    def make_reader(_path: str) -> FakeReader:
        return reader

    def find_path(_tag: str) -> Path:
        return Path("/tmp/model.gguf")

    monkeypatch.setattr(gguf_utils, "GGUFReader", make_reader)
    monkeypatch.setattr(gguf_utils, "find_local_gguf_path", find_path)


def test_get_max_ctx_from_gguf_prefers_arch_specific_context(monkeypatch: MonkeyPatch) -> None:
    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "general.architecture": FakeField("qwen"),
                "qwen.context_length": FakeField("131072"),
                "general.context_length": FakeField(4096),
            }
        ),
    )

    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") == 131072


def test_get_max_ctx_from_gguf_falls_back_to_general_and_suffix(monkeypatch: MonkeyPatch) -> None:
    def find_missing(_tag: str) -> None:
        return None

    monkeypatch.setattr(gguf_utils, "find_local_gguf_path", find_missing)
    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") is None

    _patch_reader(monkeypatch, FakeReader({"general.context_length": FakeField(8192)}))
    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") == 8192

    _patch_reader(monkeypatch, FakeReader({"llama.context_length": FakeField("32768")}))
    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") == 32768

    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "general.architecture": FakeField("qwen"),
                "general.context_length": FakeField("16384"),
            }
        ),
    )
    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") == 16384

    _patch_reader(monkeypatch, FakeReader({}))
    assert gguf_utils.get_max_ctx_from_gguf("repo/model:Q4_K_M") is None


def test_is_moe_model_detects_arch_specific_and_suffix_expert_fields(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "general.architecture": FakeField("qwen"),
                "qwen.expert_count": FakeField(0),
                "qwen.expert_used_count": FakeField(8),
            }
        ),
    )
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is True

    _patch_reader(monkeypatch, FakeReader({"foo.expert_count": FakeField(1)}))
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is True

    _patch_reader(monkeypatch, FakeReader({"foo.expert_count": FakeField(0)}))
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is False

    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "general.architecture": FakeField("qwen"),
                "qwen.expert_count": FakeField("2"),
            }
        ),
    )
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is True

    _patch_reader(monkeypatch, FakeReader({}))
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is False

    def find_missing(_tag: str) -> None:
        return None

    monkeypatch.setattr(gguf_utils, "find_local_gguf_path", find_missing)
    assert gguf_utils.is_moe_model("repo/model:Q4_K_M") is False


def test_detect_capabilities_from_chat_template_and_tags(monkeypatch: MonkeyPatch) -> None:
    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "tokenizer.chat_template": FakeField(
                    "{{ image_url }} {{ reasoning_effort }} low medium high"
                )
            }
        ),
    )

    assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
        "vision": True,
        "reasoning": {"switchable": True, "efforts": "low|medium|high"},
    }

    _patch_reader(
        monkeypatch,
        FakeReader(
            {
                "tokenizer.chat_template": FakeField(None, ValueError("bad template")),
                "general.tags": FakeField(["text-generation", "Vision"]),
            }
        ),
    )
    assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
        "vision": True,
        "reasoning": False,
    }

    def find_missing(_tag: str) -> None:
        return None

    monkeypatch.setattr(gguf_utils, "find_local_gguf_path", find_missing)
    assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
        "vision": False,
        "reasoning": False,
    }


def test_detect_capabilities_reasoning_template_variants(monkeypatch: MonkeyPatch) -> None:
    cases = [
        ("{{ enable_thinking }}", {"switchable": True, "efforts": None}),
        ("{% generation %}[think]hidden{% endgeneration %}", {"switchable": False, "efforts": None}),
        ("<think>hidden", {"switchable": False, "efforts": None}),
        ("</think>", {"switchable": False, "efforts": None}),
        ("{{ low_effort }}", {"switchable": True, "efforts": "low|high"}),
        ("{{ reasoning_effort }} none high", {"switchable": True, "efforts": "none|high"}),
        ("{{ reasoning_effort }} low high", {"switchable": True, "efforts": "low|high"}),
    ]

    for template, expected in cases:
        _patch_reader(monkeypatch, FakeReader({"tokenizer.chat_template": FakeField(template)}))
        assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
            "vision": False,
            "reasoning": expected,
        }

    _patch_reader(monkeypatch, FakeReader({"tokenizer.chat_template": FakeField("plain chat")}))
    assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
        "vision": False,
        "reasoning": False,
    }


def test_detect_capabilities_vision_template_tokens_and_tags(monkeypatch: MonkeyPatch) -> None:
    for token in ["<|image", "image_url", "boi_token", "eoi_token", "<image>", "image_pad", "<|vision"]:
        _patch_reader(monkeypatch, FakeReader({"tokenizer.chat_template": FakeField(token)}))
        assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
            "vision": True,
            "reasoning": False,
        }

    for tag in ["image-text-to-text", "computer-vision", "any-to-any"]:
        _patch_reader(monkeypatch, FakeReader({"general.tags": FakeField(["text", tag])}))
        assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
            "vision": True,
            "reasoning": False,
        }


def test_detect_capabilities_ignores_non_string_chat_templates(monkeypatch: MonkeyPatch) -> None:
    _patch_reader(monkeypatch, FakeReader({"tokenizer.chat_template": FakeField(["<think>"])}))

    assert gguf_utils.detect_capabilities("repo/model:Q4_K_M") == {
        "vision": False,
        "reasoning": False,
    }


def test_get_mmproj_size_mib_uses_local_cache_and_best_projector(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots" / "abc"
    nested = snapshots / "nested"
    nested.mkdir(parents=True)
    model = nested / "model-Q4_K_M.gguf"
    mmproj = nested / "mmproj-Q4_K.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"0" * (3 * 1024 * 1024 + 17))
    monkeypatch.setattr(gguf_cache, "HF_CACHE_DIR", tmp_path)

    assert gguf_utils.get_mmproj_size_mib("org/repo:Q4_K_M") == 3
    assert gguf_utils.get_mmproj_size_mib("org/missing:Q4_K_M") == 0


def test_get_mmproj_size_mib_returns_zero_without_matching_model_or_projector(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots" / "abc"
    snapshots.mkdir(parents=True)
    (snapshots / "model-Q5_K_M.gguf").write_bytes(b"model")
    monkeypatch.setattr(gguf_cache, "HF_CACHE_DIR", tmp_path)

    assert gguf_utils.get_mmproj_size_mib("org/repo:Q4_K_M") == 0

    (snapshots / "model-Q4_K_M.gguf").write_bytes(b"model")
    assert gguf_utils.get_mmproj_size_mib("org/repo:Q4_K_M") == 0


def test_get_mmproj_size_mib_supports_repo_tag_without_quant(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots" / "abc"
    snapshots.mkdir(parents=True)
    (snapshots / "model.gguf").write_bytes(b"model")
    (snapshots / "mmproj.gguf").write_bytes(b"0" * (2 * 1024 * 1024))
    monkeypatch.setattr(gguf_cache, "HF_CACHE_DIR", tmp_path)

    assert gguf_utils.get_mmproj_size_mib("org/repo") == 2


def test_find_local_gguf_path_returns_first_matching_quant(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    snapshots = tmp_path / "models--org--repo" / "snapshots"
    first = snapshots / "first" / "model-Q4_K_M.gguf"
    second = snapshots / "second" / "model-Q5_K_M.gguf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    monkeypatch.setattr(gguf_cache, "HF_CACHE_DIR", tmp_path)

    assert gguf_utils.find_local_gguf_path("org/repo:Q4_K_M") == first
    assert gguf_utils.find_local_gguf_path("org/repo:Q8_0") is None


def test_find_local_gguf_path_handles_absent_repo_and_duplicate_snapshot_paths(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gguf_cache, "HF_CACHE_DIR", tmp_path)

    assert gguf_utils.find_local_gguf_path("org/missing:Q4_K_M") is None

    snapshots = tmp_path / "models--org--repo" / "snapshots"
    first = snapshots / "first" / "model-Q4_K_M.gguf"
    second = snapshots / "second" / "model-Q4_K_M.gguf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    assert gguf_utils.find_local_gguf_path("org/repo:Q4_K_M") in {first, second}

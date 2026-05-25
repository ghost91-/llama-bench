from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from pytest import MonkeyPatch

import llama_bench.scan_cache as scan_cache
from llama_bench.schema_types import ScanCache


def test_set_and_get_scan_entries() -> None:
    cache: ScanCache = {}
    tag = "unsloth/Foo:Q4_K_M"

    scan_cache.set_scan_entry(
        cache,
        tag,
        vision_mode=False,
        ubatch=512,
        scan_result={
            "fit_target": 128,
            "ctx": 5000,
            "ngl": -1,
            "offload": 2,
            "ot": "0,1",
            "scan_ts": "2026-01-01T00:00:00+00:00",
        },
        mmproj="64M",
        caps={"vision": True, "reasoning": {"switchable": True, "efforts": "low"}},
    )
    scan_cache.set_scan_entry(
        cache,
        tag,
        vision_mode=False,
        ubatch=1024,
        scan_result={
            "fit_target": 128,
            "ctx": 8000,
            "ngl": -1,
            "offload": 3,
            "ot": "0,1,2",
            "scan_ts": "2026-01-02T00:00:00+00:00",
        },
    )
    scan_cache.set_scan_entry(
        cache,
        tag,
        vision_mode=True,
        ubatch=512,
        scan_result={
            "fit_target": 192,
            "ctx": 6000,
            "ngl": 72,
            "offload": 2,
            "ot": "0,1",
            "scan_ts": "2026-01-03T00:00:00+00:00",
        },
    )

    assert scan_cache.get_scan_entry(cache, tag, vision_mode=False, ubatch=512) == {
        "fit_target": 128,
        "ctx": 5000,
        "ngl": -1,
        "offload": 2,
        "ot": "0,1",
        "scan_ts": "2026-01-01T00:00:00+00:00",
    }
    assert scan_cache.get_scan_entry(cache, tag, vision_mode=False, ubatch=1024) is not None
    assert scan_cache.get_scan_entry(cache, tag, vision_mode=True, ubatch=512) is not None
    assert scan_cache.get_capabilities(cache, tag) == {
        "vision": True,
        "reasoning": {"switchable": True, "efforts": "low"},
    }
    assert scan_cache.get_model_moe(cache, tag) is None
    assert cache[tag].get("mmproj") == "64M"


def test_get_reusable_scan_entry_requires_matching_fit_target_and_fresh_timestamp() -> None:
    cache: ScanCache = {
        "repo/model:Q4_K_M": {
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": 128,
                        "ctx": 4096,
                        "ngl": -1,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-02T00:00:00+00:00",
                    },
                    "1024": {
                        "fit_target": 256,
                        "ctx": 4096,
                        "ngl": -1,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-02T00:00:00+00:00",
                    },
                    "2048": {
                        "fit_target": 128,
                        "ctx": 4096,
                        "ngl": -1,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "not a timestamp",
                    },
                }
            }
        }
    }

    assert scan_cache.get_reusable_scan_entry(
        cache,
        "repo/model:Q4_K_M",
        False,
        512,
        128,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) is not None
    assert scan_cache.get_reusable_scan_entry(
        cache,
        "repo/model:Q4_K_M",
        False,
        512,
        128,
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    ) is None
    assert scan_cache.get_reusable_scan_entry(
        cache, "repo/model:Q4_K_M", False, 1024, 128
    ) is None
    assert scan_cache.get_reusable_scan_entry(
        cache,
        "repo/model:Q4_K_M",
        False,
        2048,
        128,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) is None


def test_cached_max_ctx_requires_valid_value_and_fresh_timestamp() -> None:
    cache: ScanCache = {}
    tag = "repo/model:Q4_K_M"

    scan_cache.set_cached_max_ctx(cache, tag, 131072, "2026-01-02T00:00:00+0000")

    assert scan_cache.get_cached_max_ctx(cache, tag) == 131072
    assert scan_cache.get_cached_max_ctx(
        cache, tag, datetime(2026, 1, 1, tzinfo=timezone.utc)
    ) == 131072
    assert scan_cache.get_cached_max_ctx(
        cache, tag, datetime(2026, 1, 3, tzinfo=timezone.utc)
    ) is None

    cache[tag]["max_ctx"] = 0
    assert scan_cache.get_cached_max_ctx(cache, tag) is None


def test_set_model_moe_mutates_cache_entry() -> None:
    cache: ScanCache = {}

    scan_cache.set_model_moe(cache, "repo/model:Q4_K_M", True)

    assert cache == {"repo/model:Q4_K_M": {"moe": True}}


def test_get_capabilities_rejects_missing_reasoning() -> None:
    cache = cast(ScanCache, {"repo/model:Q4_K_M": {"caps": {"vision": True}}})

    assert scan_cache.get_capabilities(cache, "repo/model:Q4_K_M") is None


def test_save_scan_cache_prunes_unknown_tags(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "scan-cache.json"
    monkeypatch.setattr(scan_cache, "SCAN_CACHE_FILE", str(cache_file))
    monkeypatch.setattr(scan_cache, "load_tags", lambda: ["unsloth/Foo:Q4_K_M"])

    cache: ScanCache = {
        "unsloth/Foo:Q4_K_M": {
            "mmproj": "64M",
            "moe": True,
            "caps": {"vision": True, "reasoning": False},
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": 128,
                        "ctx": 5000,
                        "ngl": -1,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            },
            "vision": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": 192,
                        "ctx": 4096,
                        "ngl": 24,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-02T00:00:00+00:00",
                    }
                }
            },
        },
        "unsloth/Bar:Q4_K_M": {},
    }

    scan_cache.save_scan_cache(cache)
    loaded = scan_cache.load_scan_cache()

    assert loaded == {
        "unsloth/Foo:Q4_K_M": {
            "mmproj": "64M",
            "moe": True,
            "caps": {"vision": True, "reasoning": False},
            "text": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": 128,
                        "ctx": 5000,
                        "ngl": -1,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-01T00:00:00+00:00",
                    }
                }
            },
            "vision": {
                "ubatch_sizes": {
                    "512": {
                        "fit_target": 192,
                        "ctx": 4096,
                        "ngl": 24,
                        "offload": 1,
                        "ot": "0",
                        "scan_ts": "2026-01-02T00:00:00+00:00",
                    }
                }
            }
        }
    }


def test_load_scan_cache_reads_existing_json(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "scan-cache.json"
    cache_file.write_text(
        """
{
  "unsloth/Foo:Q4_K_M": {
    "mmproj": "64M",
    "caps": {"vision": true, "reasoning": false},
    "text": {"ubatch_sizes": {}}
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(scan_cache, "SCAN_CACHE_FILE", str(cache_file))

    assert scan_cache.load_scan_cache() == {
        "unsloth/Foo:Q4_K_M": {
            "mmproj": "64M",
            "caps": {"vision": True, "reasoning": False},
            "text": {"ubatch_sizes": {}},
        }
    }


def test_load_scan_cache_returns_empty_when_file_missing(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(scan_cache, "SCAN_CACHE_FILE", str(tmp_path / "missing.json"))

    assert scan_cache.load_scan_cache() == {}


def test_scan_cache_missing_and_invalid_values_return_none() -> None:
    cache = cast(
        ScanCache,
        {
            "repo/model:Q4_K_M": {
                "text": {
                    "ubatch_sizes": {
                        "512": {
                            "fit_target": 128,
                            "ctx": None,
                            "ngl": None,
                            "offload": None,
                            "ot": None,
                            "scan_ts": "not a timestamp",
                        },
                        "1024": {
                            "fit_target": 128,
                            "ctx": 4096,
                            "ngl": -1,
                            "offload": 1,
                            "ot": "0",
                            "scan_ts": "",
                        },
                        "2048": {
                            "fit_target": 128,
                            "ctx": 4096,
                            "ngl": -1,
                            "offload": 1,
                            "ot": "0",
                        },
                    }
                },
                "vision": {"ubatch_sizes": {}},
            }
        },
    )

    assert scan_cache.get_scan_entry(cache, "missing:Q4_K_M", vision_mode=False, ubatch=512) is None
    assert scan_cache.get_scan_entry(cache, "repo/model:Q4_K_M", vision_mode=True, ubatch=512) is None
    assert scan_cache.get_capabilities(cache, "repo/model:Q4_K_M") is None
    assert scan_cache.get_model_moe(cache, "repo/model:Q4_K_M") is None


def test_get_all_ubatch_entries_returns_mode_entries() -> None:
    cache: ScanCache = {}
    scan_cache.set_scan_entry(
        cache,
        "repo/model:Q4_K_M",
        vision_mode=False,
        ubatch=512,
        scan_result={
            "fit_target": 128,
            "ctx": 4096,
            "ngl": -1,
            "offload": None,
            "ot": None,
            "scan_ts": "2026-01-01T00:00:00+00:00",
        },
        caps={"vision": False, "reasoning": False},
    )

    assert scan_cache.get_all_ubatch_entries(cache, "repo/model:Q4_K_M", vision_mode=False) == {
        "512": {
            "fit_target": 128,
            "ctx": 4096,
            "ngl": -1,
            "offload": None,
            "ot": None,
            "scan_ts": "2026-01-01T00:00:00+00:00",
        }
    }
    assert scan_cache.get_all_ubatch_entries(cache, "repo/model:Q4_K_M", vision_mode=True) == {}
    assert scan_cache.get_capabilities(cache, "repo/model:Q4_K_M") == {
        "vision": False,
        "reasoning": False,
    }
    assert scan_cache.get_model_moe(cache, "repo/model:Q4_K_M") is None

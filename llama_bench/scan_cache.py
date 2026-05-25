import json
import os
from datetime import datetime
from typing import cast

from llama_bench.results import (
    load_tags,
    parse_timestamp_utc,
)
from llama_bench.schema_types import (
    Capabilities,
    ModelScanCacheEntry,
    ModeCacheEntry,
    ScanCache,
    ScanEntry,
    UbatchEntries,
)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
SCAN_CACHE_FILE = os.path.join(PROJECT_ROOT, "scan-cache.json")


def load_scan_cache() -> ScanCache:
    if not os.path.exists(SCAN_CACHE_FILE):
        return {}
    with open(SCAN_CACHE_FILE) as f:
        return cast(ScanCache, json.load(f))


def save_scan_cache(cache: ScanCache) -> None:
    valid_tags = set(load_tags())
    pruned = {tag: entry for tag, entry in cache.items() if tag in valid_tags}
    with open(SCAN_CACHE_FILE, "w") as f:
        json.dump(pruned, f, indent=2)
        f.write("\n")


def get_scan_entry(cache: ScanCache, tag: str, vision_mode: bool, ubatch: int) -> ScanEntry | None:
    ubatch_sizes = get_all_ubatch_entries(cache, tag, vision_mode)
    return ubatch_sizes.get(str(ubatch))


def get_reusable_scan_entry(
    cache: ScanCache,
    tag: str,
    vision_mode: bool,
    ubatch: int,
    fit_target: int,
    rescan_cutoff: datetime | None = None,
) -> ScanEntry | None:
    entry = get_scan_entry(cache, tag, vision_mode, ubatch)
    if entry is None or entry.get("fit_target") != fit_target:
        return None
    if rescan_cutoff is not None:
        scan_ts = parse_timestamp_utc(entry.get("scan_ts"))
        if scan_ts is None or scan_ts < rescan_cutoff:
            return None
    return entry


def get_cached_max_ctx(
    cache: ScanCache, tag: str, rescan_cutoff: datetime | None = None
) -> int | None:
    entry = cache.get(tag)
    if entry is None:
        return None
    max_ctx = entry.get("max_ctx")
    if not isinstance(max_ctx, int) or max_ctx <= 0:
        return None
    if rescan_cutoff is not None:
        max_ctx_ts = parse_timestamp_utc(entry.get("max_ctx_ts"))
        if max_ctx_ts is None or max_ctx_ts < rescan_cutoff:
            return None
    return max_ctx


def set_cached_max_ctx(cache: ScanCache, tag: str, max_ctx: int, max_ctx_ts: str) -> None:
    entry = cache.setdefault(tag, ModelScanCacheEntry())
    entry["max_ctx"] = max_ctx
    entry["max_ctx_ts"] = max_ctx_ts


def set_scan_entry(
    cache: ScanCache,
    tag: str,
    vision_mode: bool,
    ubatch: int,
    scan_result: ScanEntry,
    mmproj: str | None = None,
    caps: Capabilities | None = None,
) -> None:
    entry = cache.setdefault(tag, ModelScanCacheEntry())
    if mmproj is not None:
        entry["mmproj"] = mmproj
    if caps is not None:
        entry["caps"] = {
            "vision": caps["vision"],
            "reasoning": caps["reasoning"],
        }
    mode = "vision" if vision_mode else "text"
    mode_entry = entry.setdefault(mode, ModeCacheEntry(ubatch_sizes={}))
    mode_entry.setdefault("ubatch_sizes", {})
    mode_entry["ubatch_sizes"][str(ubatch)] = scan_result


def get_capabilities(cache: ScanCache, tag: str) -> Capabilities | None:
    entry = cache.get(tag)
    if entry is None:
        return None
    caps = entry.get("caps")
    if caps is None:
        return None
    if "vision" not in caps or "reasoning" not in caps:
        return None
    return caps


def get_model_moe(cache: ScanCache, tag: str) -> bool | None:
    entry = cache.get(tag)
    if entry is None or "moe" not in entry:
        return None
    return bool(entry["moe"])


def set_model_moe(cache: ScanCache, tag: str, is_moe: bool) -> None:
    cache.setdefault(tag, ModelScanCacheEntry())["moe"] = is_moe


def get_all_ubatch_entries(cache: ScanCache, tag: str, vision_mode: bool) -> UbatchEntries:
    entry = cache.get(tag)
    if entry is None:
        return {}
    mode = "vision" if vision_mode else "text"
    mode_data = entry.get(mode)
    if mode_data is None:
        return {}
    return mode_data.get("ubatch_sizes", {})

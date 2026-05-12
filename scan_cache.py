import csv
import json
import os
from datetime import datetime, timezone

from results import (
    RESULTS_FILE,
    display_name_from_tag,
    load_models,
    load_tags,
    parse_ctx,
    parse_int_field,
    parse_ngl_field,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_CACHE_FILE = os.path.join(SCRIPT_DIR, "scan-cache.json")


def load_scan_cache():
    if not os.path.exists(SCAN_CACHE_FILE):
        return {}
    with open(SCAN_CACHE_FILE) as f:
        return json.load(f)


def save_scan_cache(cache):
    valid_tags = set(load_tags())
    pruned = {tag: entry for tag, entry in cache.items() if tag in valid_tags}
    with open(SCAN_CACHE_FILE, "w") as f:
        json.dump(pruned, f, indent=2)
        f.write("\n")


def get_scan_entry(cache, tag, vision_mode):
    entry = cache.get(tag)
    if entry is None:
        return None
    mode = "vision" if vision_mode else "text"
    return entry.get(mode)


def set_scan_entry(cache, tag, vision_mode, scan_result, mmproj=None, vision=None):
    if tag not in cache:
        cache[tag] = {}
    if mmproj is not None:
        cache[tag]["mmproj"] = mmproj
    if vision is not None:
        cache[tag]["has_vision"] = vision
    mode = "vision" if vision_mode else "text"
    cache[tag][mode] = dict(scan_result)


def get_scan_ts(cache, tag, vision_mode):
    entry = get_scan_entry(cache, tag, vision_mode)
    if entry is None:
        return None

    ts_str = entry.get("scan_ts")
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    except (ValueError, OSError):
        return None


def migrate_from_csv():
    if not os.path.exists(RESULTS_FILE):
        return {}
    all_models = load_models()
    tag_lookup = {}
    for repo, quant, group, pinned in all_models:
        tag = f"{repo}:{quant}"
        display = display_name_from_tag(tag)
        provider = repo.split("/")[0]
        tag_lookup[(display, quant, provider)] = tag

    cache = {}
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model", ""), row.get("quant", ""), row.get("provider", ""))
            tag = tag_lookup.get(key)
            if tag is None:
                continue

            if tag not in cache:
                cache[tag] = {}

            mmproj = row.get("mmproj", "").strip()
            if mmproj:
                cache[tag]["mmproj"] = mmproj
            vision = row.get("vision", "").strip()
            if vision:
                cache[tag]["has_vision"] = vision

            text_ctx = parse_ctx(row.get("ctx", ""))
            if text_ctx is not None:
                text_entry = {
                    "fit_target": parse_int_field(row.get("fit_target", "")),
                    "ctx": text_ctx,
                    "ngl": parse_ngl_field(row.get("ngl", "")),
                    "ubatch": parse_int_field(row.get("ubatch", "")),
                    "moe_cpu": row.get("moe_cpu", "").strip() or "no",
                    "moe_cpu_raw": row.get("moe_cpu_raw", "").strip() or None,
                    "scan_ts": "2026-04-28T23:00:00+0200",
                }
                cache[tag]["text"] = text_entry

            vctx = parse_ctx(row.get("vctx", ""))
            if vctx is not None:
                vision_entry = {
                    "fit_target": parse_int_field(row.get("vfit_target", "")),
                    "ctx": vctx,
                    "ngl": parse_ngl_field(row.get("vngl", "")),
                    "ubatch": parse_int_field(row.get("vubatch", "")),
                    "moe_cpu": row.get("vmoe_cpu", "").strip() or "no",
                    "moe_cpu_raw": row.get("vmoe_cpu_raw", "").strip() or None,
                    "scan_ts": "2026-04-28T23:00:00+0200",
                }
                cache[tag]["vision"] = vision_entry

    save_scan_cache(cache)
    return cache

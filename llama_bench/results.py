import csv
import os
import tomllib
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping

from llama_bench.model_identity import identity_from_tag, render_model_tag, result_key_from_parts
from llama_bench.quant_order import quant_sort_key
from llama_bench.schema_types import ModelRecord, ResultRow

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
RESULTS_FILE = os.path.join(PROJECT_ROOT, "fit-bench-results.csv")
CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
MODELS_FILE = os.path.join(CONFIG_HOME, "llama.cpp", "models.ini")
MODELS_TOML = os.path.join(PROJECT_ROOT, "models.toml")

BENCH_PP = 4096
BENCH_TG = 128

PP_COL = f"pp{BENCH_PP}_tps"
PP_STDDEV_COL = f"pp{BENCH_PP}_stddev_tps"
TG_COL = f"tg{BENCH_TG}_tps"
TG_STDDEV_COL = f"tg{BENCH_TG}_stddev_tps"

PROVIDER_ORDER = {
    "unsloth": 0,
    "bartowski": 1,
    "AesSedai": 2,
    "mistralai": 3,
    "ggml-org": 4,
    "Jackrong": 5,
    "AaryanK": 6,
}

CSV_FIELDNAMES = [
    "model",
    "quant",
    "provider",
    "mode",
    "size_gib",
    "params",
    "moe",
    "fit_target",
    "ctx",
    "ngl",
    "ubatch",
    "offload",
    PP_COL,
    PP_STDDEV_COL,
    TG_COL,
    TG_STDDEV_COL,
    "reps",
    "bench_ts",
]

def load_models() -> list[ModelRecord]:
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    return [(m["repo"], m["quant"], m["group"]) for m in data.get("models", [])]


def load_tags() -> list[str]:
    return [render_model_tag(repo, quant) for repo, quant, _ in load_models()]


def parse_ctx(value: str | None) -> int | None:
    if not value:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value.endswith("k"):
        thousands = value[:-1]
        if not thousands.isdecimal():
            raise ValueError(f"invalid ctx value: {value!r}")
        return int(thousands) * 1000
    if not value.isdecimal():
        raise ValueError(f"invalid ctx value: {value!r}")
    return int(value)


def format_ctx(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def format_ngl(ngl: int | None) -> str:
    if ngl is None:
        return "?"
    if ngl == -1:
        return "all"
    return str(ngl)


def format_params(n: int | str | None) -> str:
    if not n:
        return "?"
    n = int(n)
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        b = n / 1e9
        if b == int(b):
            return f"{int(b)}B"
        return f"{b:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


def format_mmproj(mib: int | None) -> str:
    if not mib:
        return ""
    return f"{int(mib)}M"


def sort_results_file() -> None:
    if not os.path.exists(RESULTS_FILE):
        return
    rows: list[ResultRow] = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(_normalize_result_row(row))

    def param_sort_val(p: str) -> tuple[int, float]:
        if not p or p == "?":
            return (1, 0)
        n = float(p.rstrip("BTM"))
        if "T" in p:
            n *= 1000
        elif "M" in p:
            n /= 1000
        return (0, n)

    def sort_key(row: ResultRow) -> tuple[tuple[int, float], str, int, int, int, int]:
        params = row.get("params", "?")
        model = row.get("model", "")
        quant = quant_sort_key(row.get("quant", ""))
        provider = PROVIDER_ORDER.get(row.get("provider", ""), 99)
        mode = 0 if row.get("mode") == "text" else 1
        ubatch = int(row.get("ubatch", "0"))
        return (param_sort_val(params), model, quant, provider, mode, ubatch)

    rows.sort(key=sort_key)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("model", ""),
        row.get("quant", ""),
        row.get("provider", ""),
        row.get("mode", ""),
        row.get("ubatch", ""),
    )


def _normalize_result_row(row: Mapping[str, str | None]) -> ResultRow:
    return {key: "" if value is None else value for key, value in row.items()}


def _merge_nonempty_fields(
    merged: MutableMapping[str, str], incoming: Mapping[str, str], columns: Iterable[str]
) -> None:
    for col in columns:
        value = incoming.get(col)
        if value not in (None, ""):
            merged[col] = value


def append_result_row(row_dict: Mapping[str, str | None]) -> None:
    row_dict = _normalize_result_row(row_dict)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row_dict)
        return

    rows: list[ResultRow] = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(_normalize_result_row(r))

    incoming_key = _row_key(row_dict)

    merge_idx = None
    for i, existing_row in enumerate(rows):
        if _row_key(existing_row) == incoming_key:
            merge_idx = i
            break

    if merge_idx is not None:
        merged = dict(rows[merge_idx])
        _merge_nonempty_fields(
            merged,
            row_dict,
            [
                "size_gib",
                "params",
                "moe",
                "fit_target",
                "ctx",
                "ngl",
                "ubatch",
                "offload",
                PP_COL,
                PP_STDDEV_COL,
                TG_COL,
                TG_STDDEV_COL,
                "reps",
                "bench_ts",
            ],
        )
        rows[merge_idx] = merged
    else:
        rows.append(row_dict)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_timestamp_utc(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    except (ValueError, OSError):
        return None


def get_bench_ts(tag: str, mode: str = "text", ubatch: int | None = None) -> datetime | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    identity = identity_from_tag(tag, require_quant=False)
    with open(RESULTS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("model") == identity.display_name
                and row.get("quant") == identity.quant
                and row.get("provider") == identity.provider
                and row.get("mode") == mode
                and (ubatch is None or row.get("ubatch") == str(ubatch))
            ):
                return parse_timestamp_utc(row.get("bench_ts", ""))
    return None


def model_groups() -> dict[tuple[str, str, str], str]:
    groups: dict[tuple[str, str, str], str] = {}
    for repo, quant, group in load_models():
        groups[result_key_from_parts(repo, quant)] = group
    return groups

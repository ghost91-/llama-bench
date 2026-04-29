import csv
import os
import tomllib

from quant_order import QUANT_ORDER

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "fit-bench-results.csv")
CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
MODELS_FILE = os.path.join(CONFIG_HOME, "llama.cpp", "models.ini")
MODELS_TOML = os.path.join(SCRIPT_DIR, "models.toml")

BENCH_PP = 1024
BENCH_TG = 128

PP_COL = f"pp{BENCH_PP}_tps"
PP_STDDEV_COL = f"pp{BENCH_PP}_stddev_tps"
TG_COL = f"tg{BENCH_TG}_tps"
TG_STDDEV_COL = f"tg{BENCH_TG}_stddev_tps"
VPP_COL = f"vpp{BENCH_PP}_tps"
VPP_STDDEV_COL = f"vpp{BENCH_PP}_stddev_tps"
VTG_COL = f"vtg{BENCH_TG}_tps"
VTG_STDDEV_COL = f"vtg{BENCH_TG}_stddev_tps"

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
    "size_gib",
    "params",
    "model_type",
    "mmproj",
    "fit_target",
    "ctx",
    "ngl",
    "ubatch",
    "moe_cpu",
    "moe_cpu_raw",
    PP_COL,
    PP_STDDEV_COL,
    TG_COL,
    TG_STDDEV_COL,
    "reps",
    "vfit_target",
    "vctx",
    "vngl",
    "vubatch",
    "vmoe_cpu",
    "vmoe_cpu_raw",
    VPP_COL,
    VPP_STDDEV_COL,
    VTG_COL,
    VTG_STDDEV_COL,
    "vreps",
    "vision",
    "reason",
    "switch",
    "effort",
]

KNOWN_RESULT_COLS = set(CSV_FIELDNAMES)
VISION_RESULT_COLS = (
    "vfit_target",
    "vctx",
    "vngl",
    "vubatch",
    "vmoe_cpu",
    "vmoe_cpu_raw",
    VPP_COL,
    VPP_STDDEV_COL,
    VTG_COL,
    VTG_STDDEV_COL,
    "vreps",
)


def load_models():
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    return [(m["repo"], m["quant"], m["group"]) for m in data.get("models", [])]


def load_tags():
    return [f"{repo}:{quant}" for repo, quant, _ in load_models()]


def parse_ctx(value):
    if not value or value in ("-", "?"):
        return None
    value = value.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)
    return int(value)


def format_ctx(n):
    if n is None:
        return "?"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def format_ngl(ngl):
    if ngl is None:
        return "?"
    if ngl == -1:
        return "all"
    return str(ngl)


def format_params(n):
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


def format_mmproj(mib):
    if not mib:
        return ""
    return f"{int(mib)}M"


def display_name_from_tag(tag):
    repo = tag.split(":")[0] if ":" in tag else tag
    name = repo.split("/")[-1]
    for prefix in (
        "google_",
        "Qwen_",
        "qwen_",
        "zai-org_",
        "mistralai_",
        "nvidia_",
        "NVIDIA-",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("-GGUF"):
        name = name[:-5]
    if name.endswith("-it"):
        name = name[:-3]
    return name


def sort_results_file():
    if not os.path.exists(RESULTS_FILE):
        return
    rows = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = _result_fieldnames(reader.fieldnames)
        for row in reader:
            rows.append(_normalize_result_row(row))

    def param_sort_val(p):
        if not p or p == "?":
            return (1, 0)
        n = float(p.rstrip("BTM"))
        if "T" in p:
            n *= 1000
        elif "M" in p:
            n /= 1000
        return (0, n)

    def sort_key(row):
        params = row.get("params", "?")
        model = row.get("model", "")
        quant = QUANT_ORDER.get(row.get("quant", ""), 99)
        provider = PROVIDER_ORDER.get(row.get("provider", ""), 99)
        return (param_sort_val(params), model, quant, provider)

    rows.sort(key=sort_key)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row):
    return (row.get("model", ""), row.get("quant", ""), row.get("provider", ""))


def _result_fieldnames(existing_fieldnames=None, extra_keys=None):
    fieldnames = list(CSV_FIELDNAMES)
    for names in (existing_fieldnames or [], extra_keys or []):
        for name in names:
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


def _normalize_result_row(row):
    normalized = dict(row)
    for col in VISION_RESULT_COLS:
        if normalized.get(col, "") == "-":
            normalized[col] = ""
    return normalized


def _merge_nonempty_fields(merged, incoming, columns):
    for col in columns:
        value = incoming.get(col)
        if value not in (None, ""):
            merged[col] = value


def _has_vision_data(row):
    return row.get("vctx", "") not in ("", "-")


def append_result_row(row_dict):
    row_dict = _normalize_result_row(row_dict)
    if not os.path.exists(RESULTS_FILE):
        fieldnames = _result_fieldnames(extra_keys=row_dict.keys())
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_dict)
        return

    rows = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = _result_fieldnames(reader.fieldnames, row_dict.keys())
        for r in reader:
            rows.append(_normalize_result_row(r))

    incoming_key = _row_key(row_dict)
    incoming_has_vision = _has_vision_data(row_dict)

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
                    "model_type",
                    "mmproj",
                    "vision",
                    "reason",
                    "switch",
                    "effort",
                ],
            )
        if incoming_has_vision:
            _merge_nonempty_fields(
                merged,
                row_dict,
                [
                    "vctx",
                    "vngl",
                    "vubatch",
                    "vmoe_cpu",
                    "vmoe_cpu_raw",
                    VPP_COL,
                    VPP_STDDEV_COL,
                    VTG_COL,
                    VTG_STDDEV_COL,
                    "vreps",
                ],
            )
        else:
            _merge_nonempty_fields(
                merged,
                row_dict,
                [
                    "fit_target",
                    "ctx",
                    "ngl",
                    "ubatch",
                    "moe_cpu",
                    "moe_cpu_raw",
                    PP_COL,
                    PP_STDDEV_COL,
                    TG_COL,
                    TG_STDDEV_COL,
                    "reps",
                ],
            )
            if not _has_vision_data(merged):
                _merge_nonempty_fields(merged, row_dict, VISION_RESULT_COLS)
        for col, value in row_dict.items():
            if col not in KNOWN_RESULT_COLS and value not in (None, ""):
                merged[col] = value
        rows[merge_idx] = merged
    else:
        rows.append(row_dict)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

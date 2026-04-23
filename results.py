import csv
import os
import tomllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "fit-bench-results.csv")
CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
MODELS_FILE = os.path.join(CONFIG_HOME, "llama.cpp", "models.ini")
MODELS_TOML = os.path.join(SCRIPT_DIR, "models.toml")

BENCH_PP = 2048
BENCH_TG = 512

QUANT_ORDER = {
    "IQ4_XS": 0,
    "Q3_K_M": 1,
    "Q3_K_XL": 2,
    "Q4_K_M": 3,
    "Q4_K_XL": 4,
    "Q5_K_M": 5,
    "Q5_K_XL": 6,
    "Q6_K_XL": 7,
    "Q8_0": 8,
    "Q8_K_XL": 9,
    "MXFP4": 10,
}

PROVIDER_ORDER = {"unsloth": 0, "bartowski": 1, "AesSedai": 2, "ggml-org": 3}

CSV_FIELDNAMES = [
    "model",
    "quant",
    "provider",
    "size_gib",
    "params",
    "model_type",
    "mmproj",
    "ctx",
    "ngl",
    "moe_cpu",
    "pp2048_tps",
    "tg512_tps",
    "vctx",
    "vngl",
    "vpp2048_tps",
    "vtg512_tps",
    "vision",
    "reason",
    "switch",
    "effort",
]

TEXT_COLS = [7, 8, 9, 10, 11]
VISION_COLS = [12, 13, 14, 15]


def load_models():
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    return [(m["repo"], m["quant"], m["group"]) for m in data.get("models", [])]


def load_tags():
    return [
        f"{m['repo']}:{m['quant']}"
        for m in tomllib.load(open(MODELS_TOML, "rb")).get("models", [])
    ]


def format_ctx(n):
    if n is None:
        return "?"
    if n >= 1000:
        k = n / 1000
        if k == int(k):
            return f"{int(k)}k"
        return f"{k:.0f}k"
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


def load_tags_from_models_ini():
    tags = []
    if not os.path.exists(MODELS_FILE):
        raise FileNotFoundError(f"models preset file not found: {MODELS_FILE}")
    with open(MODELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[") or not line.endswith("]"):
                continue
            section = line[1:-1]
            if section == "*":
                continue
            tags.append(section)
    return tags


def sort_results_file():
    if not os.path.exists(RESULTS_FILE):
        return
    rows = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

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
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row):
    return (row.get("model", ""), row.get("quant", ""), row.get("provider", ""))


def _has_vision_data(row):
    return row.get("vctx", "") not in ("", "-")


def append_result_row(row_dict):
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerow(row_dict)
        return

    rows = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    incoming_key = _row_key(row_dict)
    incoming_has_vision = _has_vision_data(row_dict)

    merge_idx = None
    for i, existing_row in enumerate(rows):
        if _row_key(existing_row) == incoming_key:
            merge_idx = i
            break

    if merge_idx is not None:
        merged = dict(rows[merge_idx])
        for col in [
            "size_gib",
            "params",
            "model_type",
            "mmproj",
            "vision",
            "reason",
            "switch",
            "effort",
        ]:
            if col in row_dict:
                merged[col] = row_dict[col]
        if incoming_has_vision:
            for col in ["vctx", "vngl", "vpp2048_tps", "vtg512_tps"]:
                if col in row_dict:
                    merged[col] = row_dict[col]
        else:
            for col in ["ctx", "ngl", "moe_cpu", "pp2048_tps", "tg512_tps"]:
                if col in row_dict:
                    merged[col] = row_dict[col]
            if not _has_vision_data(merged):
                for col in ["vctx", "vngl", "vpp2048_tps", "vtg512_tps"]:
                    if col in row_dict:
                        merged[col] = row_dict[col]
        rows[merge_idx] = merged
    else:
        rows.append(row_dict)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

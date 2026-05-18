import re


def split_gguf_path(path: str) -> tuple[str, str, int, int]:
    prefix = path[:-5] if path.endswith(".gguf") else path
    match = re.match(r"^(.+)-([0-9]{5})-of-([0-9]{5})$", prefix, re.IGNORECASE)
    index = 1
    count = 1
    if match:
        prefix = match.group(1)
        index = int(match.group(2))
        count = int(match.group(3))

    match = re.search(r"[-.](UD-[A-Z0-9_]+|[A-Z0-9_]+)$", prefix, re.IGNORECASE)
    tag = match.group(1).upper() if match else ""
    return prefix, tag, index, count


def extract_quant_bits(path: str) -> int:
    _prefix, tag, _index, _count = split_gguf_path(path)
    match = re.search(r"\d+", tag)
    return int(match.group(0)) if match else 0


def is_model_file(path: str) -> bool:
    lower = path.lower()
    return path.endswith(".gguf") and "mmproj" not in lower and "imatrix" not in lower


def find_matching_model_files(repo_files: list[str], tag: str) -> list[str]:
    want = tag.upper()
    matches = [f for f in repo_files if is_model_file(f) and split_gguf_path(f)[1] == want]
    return sorted(matches, key=_model_sort_key)


def find_best_mmproj_file(repo_files: list[str], model_path: str) -> str | None:
    best = None
    best_depth = -1
    best_diff = 0
    model_bits = extract_quant_bits(model_path)
    model_parts = model_path.split("/")
    model_dir = model_parts[:-1]

    for path in repo_files:
        if not path.endswith(".gguf") or "mmproj" not in path.lower():
            continue

        mmproj_parts = path.split("/")
        mmproj_dir = mmproj_parts[:-1]
        if model_dir[: len(mmproj_dir)] != mmproj_dir:
            continue

        depth = len(mmproj_dir)
        diff = abs(extract_quant_bits(path) - model_bits)
        if best is None or depth > best_depth or (depth == best_depth and diff < best_diff):
            best = path
            best_depth = depth
            best_diff = diff

    return best


def _model_sort_key(path: str) -> tuple[bool, str]:
    _prefix, _tag, index, count = split_gguf_path(path)
    return (count > 1 and index != 1, path)

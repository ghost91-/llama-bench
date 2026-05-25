#!/usr/bin/env python3
import argparse
import csv
import os
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, cast

from llama_bench.gguf_utils import find_local_gguf_path
from llama_bench.model_identity import canonical_result_model, render_model_tag, result_key_from_parts
from llama_bench.quant_order import quant_sort_key
from llama_bench.results import MODELS_FILE, MODELS_TOML, PP_COL, RESULTS_FILE, TG_COL, load_models, parse_ctx
from llama_bench.sampler_config import SAMPLER_CONFIG
from llama_bench.scan_cache import get_scan_entry, load_scan_cache

SERVER_PARALLEL = 1
SERVER_BATCH_FLOOR = 4096


class IniSection(TypedDict):
    name: str
    props: list[tuple[str, str]]
    comment: str


Mode: TypeAlias = Literal["text", "vision"]


class MissingSelectedModelError(RuntimeError):
    pass


class InvalidSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class IniSelection:
    group: str
    model: str
    repo: str
    provider: str
    quant: str
    mode: Mode
    ubatch: int


@dataclass(frozen=True)
class BenchConfig:
    group: str
    model: str
    repo: str
    provider: str
    quant: str
    mode: Mode
    ubatch: int
    ctx: int
    fit_target: int | None
    pp_tps: float
    tg_tps: float

def _server_batch_size(ubatch: int) -> int:
    return max(SERVER_BATCH_FLOOR, SERVER_PARALLEL * ubatch)


def _append_fit_props(
    props: list[tuple[str, str]],
    ctx: int | None,
    fit_target: int | None,
    ubatch: int | None,
) -> None:
    if ctx is not None:
        props.append(("ctx-size", str(ctx)))
    if fit_target is not None:
        props.append(("fit-target", str(fit_target)))
    if ubatch is not None:
        props.append(("ubatch-size", str(ubatch)))
        props.append(("batch-size", str(_server_batch_size(ubatch))))


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_ini_selections(path: str = MODELS_TOML) -> list[IniSelection]:
    if not os.path.exists(path):
        raise InvalidSelectionError(f"models.toml not found: {path}")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    raw_models = raw.get("models", [])
    if not isinstance(raw_models, list):
        raise InvalidSelectionError("models.toml field 'models' must be an array")
    selections: list[IniSelection] = []
    raw_models_list = cast(list[object], raw_models)
    for idx, value in enumerate(raw_models_list, start=1):
        if not isinstance(value, dict):
            raise InvalidSelectionError(f"models[{idx}] must be a table")
        selections.extend(_parse_model_ini_selections(idx, cast(dict[str, object], value)))
    return selections


def _parse_model_ini_selections(idx: int, model_entry: dict[str, object]) -> list[IniSelection]:
    raw_ini = model_entry.get("ini")
    if raw_ini is None:
        return []
    if not isinstance(raw_ini, list):
        raise InvalidSelectionError(f"models[{idx}].ini must be an array")
    raw_ini_list = cast(list[object], raw_ini)
    repo = _required_str(model_entry, idx, "repo", prefix="models")
    quant = _required_str(model_entry, idx, "quant", prefix="models")
    group = _required_str(model_entry, idx, "group", prefix="models")
    model, _quant, provider = result_key_from_parts(repo, quant)
    selections: list[IniSelection] = []
    for ini_idx, value in enumerate(raw_ini_list, start=1):
        if not isinstance(value, dict):
            raise InvalidSelectionError(f"models[{idx}].ini[{ini_idx}] must be a table")
        selections.append(_parse_ini_selection(idx, ini_idx, cast(dict[str, object], value), group, model, repo, provider, quant))
    return selections


def _parse_ini_selection(
    model_idx: int,
    ini_idx: int,
    value: dict[str, object],
    group: str,
    model: str,
    repo: str,
    provider: str,
    quant: str,
) -> IniSelection:
    allowed = {"mode", "ubatch"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InvalidSelectionError(
            f"models[{model_idx}].ini[{ini_idx}] has unknown fields: {', '.join(unknown)}"
        )
    mode = _required_str(value, ini_idx, "mode", prefix=f"models[{model_idx}].ini")
    if mode not in ("text", "vision"):
        raise InvalidSelectionError(f"models[{model_idx}].ini[{ini_idx}].mode must be 'text' or 'vision'")
    ubatch = _required_int(value, ini_idx, "ubatch", prefix=f"models[{model_idx}].ini")
    return IniSelection(group, model, repo, provider, quant, mode, ubatch)


def _required_str(value: dict[str, object], idx: int, key: str, *, prefix: str = "configs") -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or raw == "":
        raise InvalidSelectionError(f"{prefix}[{idx}].{key} must be a non-empty string")
    return raw


def _required_int(value: dict[str, object], idx: int, key: str, *, prefix: str = "configs") -> int:
    raw = value.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise InvalidSelectionError(f"{prefix}[{idx}].{key} must be an integer")
    return raw


def load_bench_configs(results_file: str = RESULTS_FILE) -> list[BenchConfig]:
    models = {result_key_from_parts(repo, quant): (group, repo) for repo, quant, group in load_models()}
    configs: list[BenchConfig] = []
    seen: set[tuple[str, str, str, Mode, int]] = set()
    duplicates: list[str] = []
    if not os.path.exists(results_file):
        return configs
    with open(results_file, newline="") as f:
        for row in csv.DictReader(f):
            normalized = {key: value or "" for key, value in row.items()}
            mode = normalized.get("mode")
            if mode not in ("text", "vision"):
                continue
            provider = normalized.get("provider", "")
            model = canonical_result_model(normalized.get("model", ""), provider)
            quant = normalized.get("quant", "")
            model_record = models.get((model, quant, provider))
            ctx = parse_ctx(normalized.get("ctx"))
            ubatch = _parse_int(normalized.get("ubatch"))
            fit_target = _parse_int(normalized.get("fit_target"))
            pp_tps = _parse_float(normalized.get(PP_COL))
            tg_tps = _parse_float(normalized.get(TG_COL))
            if model_record is None or ctx is None or ubatch is None or pp_tps is None or tg_tps is None:
                continue
            group, repo = model_record
            key = (model, quant, provider, mode, ubatch)
            if key in seen:
                duplicates.append(f"{model} {quant} {provider} {mode} ubatch={ubatch}")
                continue
            seen.add(key)
            configs.append(BenchConfig(group, model, repo, provider, quant, mode, ubatch, ctx, fit_target, pp_tps, tg_tps))
    if duplicates:
        raise InvalidSelectionError("Duplicate benchmark rows for selected key space:\n" + "\n".join(duplicates))
    return sorted(configs, key=_bench_config_sort_key)


def resolve_manual_selections(
    selections: Sequence[IniSelection], bench_configs: Sequence[BenchConfig]
) -> list[BenchConfig]:
    selected: list[BenchConfig] = []
    missing: list[str] = []
    scan_cache = None
    seen: set[tuple[str, str, str, str, Mode, int, int]] = set()
    for selection in selections:
        matches = [config for config in bench_configs if _matches_selection(config, selection)]
        if matches:
            config = matches[0]
        else:
            if scan_cache is None:
                scan_cache = load_scan_cache()
            scan_entry = get_scan_entry(
                scan_cache,
                render_model_tag(selection.repo, selection.quant),
                selection.mode == "vision",
                selection.ubatch,
            )
            if scan_entry is None:
                missing.append(_format_selection(selection))
                continue
            config = BenchConfig(
                selection.group,
                selection.model,
                selection.repo,
                selection.provider,
                selection.quant,
                selection.mode,
                selection.ubatch,
                scan_entry["ctx"],
                scan_entry["fit_target"],
                0.0,
                0.0,
            )
        key = (
            config.group,
            config.model,
            config.provider,
            config.quant,
            config.mode,
            config.ubatch,
            config.ctx,
        )
        if key not in seen:
            selected.append(config)
            seen.add(key)
    if missing:
        raise InvalidSelectionError("No benchmark result for selected configs:\n" + "\n".join(missing))
    return sorted(selected, key=_bench_config_sort_key)


def add_free_vision_configs(
    selected: Sequence[BenchConfig], bench_configs: Sequence[BenchConfig]
) -> list[BenchConfig]:
    result = list(selected)
    seen = {_config_identity(config) for config in result}
    vision_by_text_key = {
        _free_vision_key(config): config
        for config in bench_configs
        if config.mode == "vision"
    }
    scan_cache = None
    for config in selected:
        if config.mode != "text":
            continue
        vision_config = vision_by_text_key.get(_free_vision_key(config))
        if vision_config is None:
            if scan_cache is None:
                scan_cache = load_scan_cache()
            scan_entry = get_scan_entry(
                scan_cache,
                render_model_tag(config.repo, config.quant),
                True,
                config.ubatch,
            )
            if scan_entry is None or scan_entry["ctx"] != config.ctx:
                continue
            vision_config = BenchConfig(
                config.group,
                config.model,
                config.repo,
                config.provider,
                config.quant,
                "vision",
                config.ubatch,
                scan_entry["ctx"],
                scan_entry["fit_target"],
                0.0,
                0.0,
            )
        identity = _config_identity(vision_config)
        if identity not in seen:
            result.append(vision_config)
            seen.add(identity)
    return sorted(result, key=_bench_config_sort_key)


def _config_identity(config: BenchConfig) -> tuple[str, str, str, str, Mode, int, int]:
    return (
        config.group,
        config.model,
        config.provider,
        config.quant,
        config.mode,
        config.ubatch,
        config.ctx,
    )


def _free_vision_key(config: BenchConfig) -> tuple[str, str, str, str, int, int]:
    return (
        config.group,
        config.model,
        config.provider,
        config.quant,
        config.ubatch,
        config.ctx,
    )


def _matches_selection(config: BenchConfig, selection: IniSelection) -> bool:
    return (
        config.group == selection.group
        and config.model == selection.model
        and config.provider == selection.provider
        and config.quant == selection.quant
        and config.mode == selection.mode
        and config.ubatch == selection.ubatch
    )


def _format_selection(selection: IniSelection) -> str:
    return (
        f"group={selection.group}, provider={selection.provider}, quant={selection.quant}, "
        f"mode={selection.mode}, ubatch={selection.ubatch}"
    )


def format_sampler_settings(group: str, skip_keys: set[str] | None = None) -> list[tuple[str, str]]:
    cfg = SAMPLER_CONFIG.get(group, {})
    props: list[tuple[str, str]] = []
    for k, v in cfg.items():
        if skip_keys and k in skip_keys:
            continue
        props.append((k, v))
    return props


def build_ini_sections(
    configs: Sequence[BenchConfig],
    gguf_exists_fn: Callable[[str], object | None] = find_local_gguf_path,
) -> list[IniSection]:
    missing: list[str] = []
    raw: list[IniSection] = []
    used_labels: set[str] = set()
    for config in sorted(configs, key=_bench_config_sort_key):
        full_tag = render_model_tag(config.repo, config.quant)
        if gguf_exists_fn(full_tag) is None:
            missing.append(f"{full_tag} not found on disk: {_config_description(config)}")
            continue
        props: list[tuple[str, str]] = []
        props.append(("hf", full_tag))
        _append_fit_props(props, config.ctx, config.fit_target, config.ubatch)
        if config.mode == "vision":
            props.append(("mmproj-auto", "on"))
            props.append(("mmproj-offload", "on"))
        skip_keys = {"ubatch-size"}
        props.extend(format_sampler_settings(config.group, skip_keys=skip_keys))
        label = _config_label(config, used_labels)
        used_labels.add(label)
        raw.append({"name": label, "props": props, "comment": _config_description(config)})
    if missing:
        raise MissingSelectedModelError("Missing selected GGUF files:\n" + "\n".join(missing))
    return _merge_vision_sections(raw)


def _bench_config_sort_key(config: BenchConfig) -> tuple[str, str, str, int, int, str, int, int]:
    return (
        config.group,
        config.model,
        config.provider,
        0 if config.mode == "text" else 1,
        quant_sort_key(config.quant),
        config.quant,
        config.ubatch,
        config.ctx,
    )


def _config_label(config: BenchConfig, used_labels: set[str]) -> str:
    base = "-".join([
        config.group,
        slug(config.provider),
        slug(config.quant),
        slug(config.mode),
        f"ub{config.ubatch}",
    ])
    if base not in used_labels:
        return base
    suffix = 2
    while f"{base}-{suffix}" in used_labels:
        suffix += 1
    return f"{base}-{suffix}"


def slug(value: str) -> str:
    parts: list[str] = []
    last_was_dash = True
    for char in value.lower():
        if char.isascii() and char.isalnum():
            parts.append(char)
            last_was_dash = False
        elif not last_was_dash:
            parts.append("-")
            last_was_dash = True
    return "".join(parts).strip("-") or "config"


def _config_description(config: BenchConfig) -> str:
    speed = "unbenchmarked" if config.pp_tps == 0.0 and config.tg_tps == 0.0 else f"pp {config.pp_tps:.0f}, tg {config.tg_tps:.0f}"
    return (
        f"{config.group} {config.provider} {config.quant}. {config.mode}, "
        f"ctx {config.ctx}, ubatch {config.ubatch}, {speed}."
    )


def _merge_vision_sections(sections: list[IniSection]) -> list[IniSection]:
    by_model_key: dict[tuple[str, str, str, str], list[int]] = {}
    for idx, sec in enumerate(sections):
        props = dict(sec["props"])
        hf = props.get("hf", "")
        ctx = props.get("ctx-size", "")
        ubatch = props.get("ubatch-size", "")
        mode = "vision" if "mmproj-auto" in props else "text"
        by_model_key.setdefault((hf, ctx, ubatch, mode), []).append(idx)

    merged_indices: set[int] = set()
    result: list[IniSection] = []
    for idx, sec in enumerate(sections):
        if idx in merged_indices:
            continue
        props = dict(sec["props"])
        mode = "vision" if "mmproj-auto" in props else "text"
        if mode == "text":
            hf = props.get("hf", "")
            ctx = props.get("ctx-size", "")
            ubatch = props.get("ubatch-size", "")
            vision_indices = by_model_key.get((hf, ctx, ubatch, "vision"), [])
            vision_idx = next((i for i in vision_indices if i > idx and i not in merged_indices), None)
            if vision_idx is not None:
                vision_props = dict(sections[vision_idx]["props"])
                vision_fit_target = vision_props.get("fit-target")
                merged = _apply_vision_merge(sec, vision_fit_target)
                merged_indices.add(vision_idx)
                result.append(merged)
                continue
        result.append(sec)
    return result


def _apply_vision_merge(text_section: IniSection, vision_fit_target: str | None) -> IniSection:
    props: list[tuple[str, str]] = []
    text_props = dict(text_section["props"])
    for k, v in text_section["props"]:
        if k == "fit-target" and vision_fit_target is not None:
            props.append((k, vision_fit_target))
        else:
            props.append((k, v))
    if "mmproj-auto" not in text_props:
        props.append(("mmproj-auto", "on"))
    if "mmproj-offload" not in text_props:
        props.append(("mmproj-offload", "on"))
    if vision_fit_target is not None and "fit-target" not in text_props:
        props.append(("fit-target", vision_fit_target))
    comment = text_section["comment"]
    if comment:
        comment += " Includes matching vision settings."
    return {"name": text_section["name"], "props": props, "comment": comment}


def render_ini(sections: Sequence[IniSection]) -> str:
    lines: list[str] = []
    lines.append("version = 1")
    lines.append("")
    lines.append("[*]")
    lines.append("fit = on")
    lines.append("fit-ctx = 5000")
    lines.append("flash-attn = on")
    lines.append(f"parallel = {SERVER_PARALLEL}")
    lines.append("no-mmproj = on")
    lines.append("no-mmap = on")
    lines.append(f"batch-size = {_server_batch_size(512)}")
    for sec in sections:
        lines.append("")
        if sec["comment"]:
            lines.append(f"; {sec['comment']}")
        lines.append(f"[{sec['name']}]")
        for k, v in sec["props"]:
            lines.append(f"{k} = {v}")
    lines.append("")
    return "\n".join(lines)


def generate_ini(
    configs: Sequence[BenchConfig],
    output: str,
    dry_run: bool,
) -> None:
    content = render_ini(
        build_ini_sections(configs, gguf_exists_fn=find_local_gguf_path)
    )
    if dry_run:
        print(content, end="")
    else:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output, "w") as f:
            f.write(content)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate models.ini from manual models.toml ini selections"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout")
    parser.add_argument("--output", default=MODELS_FILE, help="Output file path")
    args = parser.parse_args()

    selections = load_ini_selections()
    bench_configs = load_bench_configs()
    configs = add_free_vision_configs(resolve_manual_selections(selections, bench_configs), bench_configs)
    generate_ini(configs, args.output, args.dry_run)


if __name__ == "__main__":
    main()

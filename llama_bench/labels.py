from __future__ import annotations

from collections.abc import Mapping, Sequence

from llama_bench.consolidation import ConfigKey, ReportEntry
from llama_bench.results import format_ctx

PROFILE_LABELS: Mapping[str, str] = {
    "agentic-coding": "agentic",
    "slow-max-intelligence-chat": "smart",
    "regular-quick-chat": "chat",
    "data-extraction": "extract",
    "writing-and-polish": "writing",
    "long-docs": "long",
    "vision-deep": "vision",
    "vision-fast": "vision-fast",
    "small-fast-tasks": "fast",
}
PROFILE_TITLES: Mapping[str, str] = {
    "agentic-coding": "Agentic coding",
    "slow-max-intelligence-chat": "Slow max-intelligence chat",
    "regular-quick-chat": "Regular quick chat",
    "data-extraction": "Data extraction",
    "writing-and-polish": "Writing and polish",
    "long-docs": "Long documents",
    "vision-deep": "Deep vision",
    "vision-fast": "Fast vision",
    "small-fast-tasks": "Small fast tasks",
}
CHOICE_LABELS: Mapping[str, str] = {
    "recommended": "",
    "faster": "fast",
    "fastest": "fastest",
    "higher-quality": "quality",
    "best-quality": "best-quality",
    "more-context": "long",
    "highest-context": "maxctx",
}
CHOICE_DESCRIPTIONS: Mapping[str, str] = {
    "recommended": "default",
    "faster": "faster alternative",
    "fastest": "fastest alternative",
    "higher-quality": "higher-quality alternative",
    "best-quality": "best-quality alternative",
    "more-context": "longer-context alternative",
    "highest-context": "maximum-context alternative",
}


def label_for_config(key: ConfigKey, uses: Sequence[ReportEntry], used_labels: set[str]) -> str:
    group = key[0]
    primary = uses[0]
    base = f"{group}-{label_stem(primary)}"
    if base not in used_labels:
        return base

    for suffix in label_suffixes(key, uses):
        candidate = f"{base}-{suffix}"
        if candidate not in used_labels:
            return candidate
    suffix = 2
    while f"{base}-{suffix}" in used_labels:
        suffix += 1
    return f"{base}-{suffix}"


def label_stem(entry: ReportEntry) -> str:
    profile_label = PROFILE_LABELS.get(entry.selection.profile, slug(entry.selection.profile))
    choice_label = CHOICE_LABELS.get(entry.choice, slug(entry.choice))
    if entry.choice == "recommended" or not choice_label:
        return profile_label
    return f"{profile_label}-{choice_label}"


def label_suffixes(key: ConfigKey, uses: Sequence[ReportEntry]) -> list[str]:
    _group, _provider, quant, mode, ubatch, ctx, _pp_tps, _tg_tps = key
    suffixes: list[str] = []
    if mode == "vision" and "vision" not in label_stem(uses[0]):
        suffixes.append("vision")
    suffixes.extend(
        choice_label
        for entry in uses
        if (choice_label := CHOICE_LABELS.get(entry.choice, ""))
        and choice_label not in suffixes
    )
    suffixes.extend([slug(quant), format_ctx(ctx).lower(), f"ub{ubatch}"])
    return suffixes


def description_for_config(key: ConfigKey, uses: Sequence[ReportEntry]) -> str:
    _group, _provider, _quant, mode, _ubatch, ctx, pp_tps, tg_tps = key
    primary = uses[0]
    profile = PROFILE_TITLES.get(primary.selection.profile, primary.selection.profile)
    choice = CHOICE_DESCRIPTIONS.get(primary.choice, primary.choice)
    description = f"{profile} {choice}"
    seen_profiles = {primary.selection.profile}
    other_uses: list[ReportEntry] = []
    for entry in uses[1:]:
        if entry.selection.profile in seen_profiles:
            continue
        seen_profiles.add(entry.selection.profile)
        other_uses.append(entry)
    if other_uses:
        joined = ", ".join(
            (
                f"{PROFILE_TITLES.get(entry.selection.profile, entry.selection.profile)} "
                f"{CHOICE_DESCRIPTIONS.get(entry.choice, entry.choice)}"
            )
            for entry in other_uses[:3]
        )
        if len(other_uses) > 3:
            joined += f", +{len(other_uses) - 3} more"
        description += f"; also {joined}"
    return f"{description}. {mode}, ctx {format_ctx(ctx)}, pp {pp_tps:.0f}, tg {tg_tps:.0f}."


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

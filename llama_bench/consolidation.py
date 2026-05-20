from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llama_bench.selection import ProfileSelection, ScoredCandidate

PROFILE_PRIORITIES = {
    "agentic-coding": 1,
    "slow-max-intelligence-chat": 2,
    "regular-quick-chat": 3,
    "data-extraction": 4,
    "writing-and-polish": 5,
    "long-docs": 6,
    "vision-deep": 7,
    "vision-fast": 8,
    "small-fast-tasks": 9,
}
CHOICE_PRIORITIES = {
    "recommended": 0,
    "faster": 1,
    "higher-quality": 2,
    "more-context": 3,
    "fastest": 4,
    "best-quality": 5,
    "highest-context": 6,
}

ConfigKey = tuple[str, str, str, str, int, int, float, float]


@dataclass(frozen=True)
class ReportEntry:
    selection: ProfileSelection
    choice: str
    scored: ScoredCandidate


@dataclass(frozen=True)
class LabelledConfig:
    label: str
    description: str
    key: ConfigKey
    entries: tuple[ReportEntry, ...]


def consolidate_entries(
    entries: Sequence[ReportEntry], *, max_configs_per_group: int
) -> list[ReportEntry]:
    by_group: dict[str, list[ReportEntry]] = {}
    for entry in entries:
        by_group.setdefault(entry.selection.group, []).append(entry)

    consolidated: list[ReportEntry] = []
    for group in sorted(by_group):
        kept: list[ScoredCandidate] = []
        for entry in sorted(by_group[group], key=entry_priority):
            replacement = find_consolidation_target(entry, kept)
            if replacement is None:
                if entry.choice != "recommended" and len(kept) >= max_configs_per_group:
                    continue
                kept.append(entry.scored)
                replacement = entry.scored
            elif len(kept) < max_configs_per_group and should_keep_distinct(entry, replacement):
                kept.append(entry.scored)
                replacement = entry.scored
            consolidated.append(ReportEntry(entry.selection, entry.choice, replacement))
    return consolidated


def entry_priority(entry: ReportEntry) -> tuple[int, int, float]:
    return (
        0 if entry.choice == "recommended" else 1,
        PROFILE_PRIORITIES.get(entry.selection.profile, 99) * 10
        + CHOICE_PRIORITIES.get(entry.choice, 99),
        -entry.scored.score,
    )


def find_consolidation_target(
    entry: ReportEntry, kept: Sequence[ScoredCandidate]
) -> ScoredCandidate | None:
    candidates = [scored for scored in kept if can_reuse_config(entry, scored)]
    if not candidates:
        return None
    return min(candidates, key=lambda scored: reuse_loss(entry.scored, scored))


def should_keep_distinct(entry: ReportEntry, replacement: ScoredCandidate) -> bool:
    if entry.choice != "recommended":
        return False
    return reuse_loss(entry.scored, replacement) > 0.2


def can_reuse_config(entry: ReportEntry, replacement: ScoredCandidate) -> bool:
    original = entry.scored
    original_candidate = original.candidate
    replacement_candidate = replacement.candidate
    if original_candidate.mode != replacement_candidate.mode:
        return False
    if not satisfies_profile_constraints(entry, replacement):
        return False

    quality_tolerance, ctx_ratio, speed_ratio = reuse_tolerances(entry)
    return (
        replacement.quality.score >= original.quality.score - quality_tolerance
        and replacement_candidate.ctx >= original_candidate.ctx * ctx_ratio
        and replacement_candidate.pp_tps >= original_candidate.pp_tps * speed_ratio
        and replacement_candidate.tg_tps >= original_candidate.tg_tps * speed_ratio
    )


def satisfies_profile_constraints(entry: ReportEntry, replacement: ScoredCandidate) -> bool:
    profile = entry.selection.profile_config
    if profile is None:
        return True
    candidate = replacement.candidate
    return (
        candidate.mode in profile.modes
        and candidate.ctx >= profile.min_ctx
        and candidate.pp_tps >= profile.min_pp_tps
        and candidate.tg_tps >= profile.min_tg_tps
    )


def reuse_tolerances(entry: ReportEntry) -> tuple[float, float, float]:
    quality_tolerance = 0.08
    ctx_ratio = 0.80
    speed_ratio = 0.80
    if entry.choice == "recommended":
        quality_tolerance = 0.06
        ctx_ratio = 0.85
        speed_ratio = 0.85
    elif entry.choice in {"higher-quality", "best-quality"}:
        quality_tolerance = 0.03
    elif entry.choice in {"more-context", "highest-context"}:
        ctx_ratio = 0.90
    elif entry.choice in {"faster", "fastest"}:
        speed_ratio = 0.90
    if entry.selection.profile == "agentic-coding" and entry.choice == "recommended":
        quality_tolerance = 0.04
        ctx_ratio = 0.90
        speed_ratio = 0.90
    return quality_tolerance, ctx_ratio, speed_ratio


def reuse_loss(original: ScoredCandidate, replacement: ScoredCandidate) -> float:
    original_candidate = original.candidate
    replacement_candidate = replacement.candidate
    return (
        max(0.0, original.quality.score - replacement.quality.score)
        + relative_deficit(original_candidate.ctx, replacement_candidate.ctx)
        + relative_deficit(original_candidate.pp_tps, replacement_candidate.pp_tps)
        + relative_deficit(original_candidate.tg_tps, replacement_candidate.tg_tps)
    )


def relative_deficit(original: float | int, replacement: float | int) -> float:
    if original <= 0:
        return 0.0
    return max(0.0, (float(original) - float(replacement)) / float(original))


def config_key(group: str, scored: ScoredCandidate) -> ConfigKey:
    candidate = scored.candidate
    return (
        group,
        candidate.provider,
        candidate.quant,
        candidate.mode,
        candidate.ubatch,
        candidate.ctx,
        candidate.pp_tps,
        candidate.tg_tps,
    )

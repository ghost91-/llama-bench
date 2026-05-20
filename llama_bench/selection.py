import csv
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log
from typing import Literal, TypeAlias

from llama_bench.capabilities import CapabilityName, ModelCapabilities, load_capabilities
from llama_bench.quality import (
    Quality,
    QualitySource,
    estimate_quality,
    kld_quality,
    normalise_qualities,
    quant_proxy_quality,
)
from llama_bench.quality import (
    estimate_close_provider_kld,
    estimate_cross_provider_kld,
    estimate_kld_for_provider,
    providers_are_close,
    provider_klds_by_equivalent_quant,
    quant_kld_curve,
)
from llama_bench.quant_order import QUANT_ORDER, UNKNOWN_QUANT_ORDER
from llama_bench.results import PP_COL, RESULTS_FILE, TG_COL, model_groups, parse_ctx

_estimate_close_provider_kld = estimate_close_provider_kld
_estimate_cross_provider_kld = estimate_cross_provider_kld
_estimate_kld_for_provider = estimate_kld_for_provider
_providers_are_close = providers_are_close
_provider_klds_by_equivalent_quant = provider_klds_by_equivalent_quant
_quant_kld_curve = quant_kld_curve

__all__ = [
    "AlternativeName",
    "Candidate",
    "Mode",
    "Profile",
    "ProfileSelection",
    "PROFILES",
    "Quality",
    "QualitySource",
    "ScoredCandidate",
    "estimate_quality",
    "kld_quality",
    "load_candidates",
    "quant_proxy_quality",
    "score_candidates",
    "select_for_group",
    "select_profiles",
]

Mode: TypeAlias = Literal["text", "vision"]
AlternativeName: TypeAlias = Literal[
    "faster",
    "higher-quality",
    "more-context",
    "fastest",
    "best-quality",
    "highest-context",
]

KLD_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kld-results.csv")


@dataclass(frozen=True)
class Profile:
    name: str
    min_ctx: int
    modes: tuple[Mode, ...]
    min_capabilities: Mapping[CapabilityName, int]
    weights: Mapping[str, float]
    alternatives: tuple[AlternativeName, ...]
    min_pp_tps: float = 0.0
    min_tg_tps: float = 0.0
    target_tg_tps: float = 0.0


@dataclass(frozen=True)
class Candidate:
    group: str
    model: str
    quant: str
    provider: str
    mode: Mode
    ctx: int
    ubatch: int
    pp_tps: float
    tg_tps: float
    params: int | None
    size_gib: float | None
    kld: float | None

    @property
    def key(self) -> tuple[str, str, str, Mode, int, int]:
        return (self.model, self.quant, self.provider, self.mode, self.ubatch, self.ctx)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    quality: Quality
    score: float


@dataclass(frozen=True)
class ProfileSelection:
    group: str
    profile: str
    recommendation: ScoredCandidate | None
    alternatives: Mapping[AlternativeName, ScoredCandidate]
    skipped_reason: str | None
    profile_config: Profile | None = None


PROFILES: dict[str, Profile] = {
    "agentic-coding": Profile(
        name="agentic-coding",
        min_ctx=96_000,
        modes=("text",),
        min_capabilities={"coding": 4, "reasoning": 4, "tools": 3},
        weights={"quality": 0.35, "ctx": 0.15, "pp": 0.35, "tg": 0.15},
        alternatives=("faster", "higher-quality", "more-context", "fastest", "best-quality", "highest-context"),
        target_tg_tps=30.0,
    ),
    "regular-quick-chat": Profile(
        name="regular-quick-chat",
        min_ctx=24_000,
        modes=("text",),
        min_capabilities={"reasoning": 2, "writing": 2},
        weights={"quality": 0.25, "ctx": 0.10, "pp": 0.20, "tg": 0.45},
        alternatives=("faster", "higher-quality", "more-context", "best-quality", "highest-context"),
    ),
    "slow-max-intelligence-chat": Profile(
        name="slow-max-intelligence-chat",
        min_ctx=64_000,
        modes=("text",),
        min_capabilities={"reasoning": 4},
        weights={"quality": 0.55, "ctx": 0.25, "pp": 0.10, "tg": 0.10},
        alternatives=("faster", "more-context", "best-quality", "highest-context"),
    ),
    "small-fast-tasks": Profile(
        name="small-fast-tasks",
        min_ctx=8_000,
        modes=("text",),
        min_capabilities={"reasoning": 1},
        weights={"quality": 0.25, "ctx": 0.05, "pp": 0.25, "tg": 0.45},
        alternatives=("higher-quality", "more-context", "best-quality", "highest-context"),
        min_pp_tps=2500.0,
        min_tg_tps=70.0,
    ),
    "vision-deep": Profile(
        name="vision-deep",
        min_ctx=30_000,
        modes=("vision",),
        min_capabilities={"vision": 3, "reasoning": 3},
        weights={"quality": 0.26, "ctx": 0.30, "pp": 0.25, "tg": 0.19},
        alternatives=("faster", "higher-quality", "more-context", "best-quality", "highest-context"),
        min_tg_tps=15.0,
    ),
    "vision-fast": Profile(
        name="vision-fast",
        min_ctx=8_000,
        modes=("vision",),
        min_capabilities={"vision": 2},
        weights={"quality": 0.35, "ctx": 0.05, "pp": 0.20, "tg": 0.40},
        alternatives=("higher-quality", "more-context", "best-quality", "highest-context"),
        min_pp_tps=500.0,
        min_tg_tps=70.0,
    ),
    "long-docs": Profile(
        name="long-docs",
        min_ctx=128_000,
        modes=("text",),
        min_capabilities={"reasoning": 3, "writing": 3},
        weights={"quality": 0.40, "ctx": 0.35, "pp": 0.20, "tg": 0.05},
        alternatives=("faster", "higher-quality", "best-quality", "highest-context"),
    ),
    "data-extraction": Profile(
        name="data-extraction",
        min_ctx=48_000,
        modes=("text",),
        min_capabilities={"reasoning": 3, "tools": 3},
        weights={"quality": 0.35, "ctx": 0.25, "pp": 0.25, "tg": 0.15},
        alternatives=("faster", "higher-quality", "more-context", "best-quality", "highest-context"),
    ),
    "writing-and-polish": Profile(
        name="writing-and-polish",
        min_ctx=24_000,
        modes=("text",),
        min_capabilities={"writing": 4},
        weights={"quality": 0.45, "ctx": 0.15, "pp": 0.15, "tg": 0.25},
        alternatives=("faster", "higher-quality", "more-context", "best-quality", "highest-context"),
    ),
}


def load_candidates(results_file: str = RESULTS_FILE, kld_file: str = KLD_FILE) -> list[Candidate]:
    groups = model_groups()
    kld_by_key = _load_kld(kld_file)
    candidates: list[Candidate] = []
    if not os.path.exists(results_file):
        return candidates
    with open(results_file, newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {key: value or "" for key, value in raw_row.items()}
            mode = row.get("mode")
            if mode not in ("text", "vision"):
                continue
            key = (row.get("model", ""), row.get("quant", ""), row.get("provider", ""))
            group = groups.get(key)
            if group is None:
                continue
            candidate = _candidate_from_row(row, group, mode, kld_by_key.get(key))
            if candidate is not None:
                candidates.append(candidate)
    return sorted(candidates, key=_candidate_sort_key)


def select_profiles(
    candidates: Sequence[Candidate],
    *,
    profiles: Iterable[Profile] = PROFILES.values(),
    capabilities: Mapping[str, ModelCapabilities] | None = None,
) -> list[ProfileSelection]:
    caps = load_capabilities() if capabilities is None else capabilities
    by_group: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.group, []).append(candidate)

    selections: list[ProfileSelection] = []
    for profile in profiles:
        for group in sorted(by_group):
            selections.append(select_for_group(by_group[group], profile, caps.get(group)))
    return selections


def select_for_group(
    candidates: Sequence[Candidate], profile: Profile, capabilities: ModelCapabilities | None
) -> ProfileSelection:
    group = candidates[0].group if candidates else ""
    if not candidates:
        return ProfileSelection(group, profile.name, None, {}, "no benchmark results", profile)
    if capabilities is None:
        return ProfileSelection(group, profile.name, None, {}, "missing capability prior", profile)

    missing_caps = [
        f"{name} {capabilities[name]} < {minimum}"
        for name, minimum in profile.min_capabilities.items()
        if capabilities[name] < minimum
    ]
    if missing_caps:
        return ProfileSelection(group, profile.name, None, {}, "; ".join(missing_caps), profile)

    passing = [
        candidate
        for candidate in candidates
        if candidate.mode in profile.modes and candidate.ctx >= profile.min_ctx
    ]
    if not passing:
        max_ctx = max((candidate.ctx for candidate in candidates if candidate.mode in profile.modes), default=0)
        return ProfileSelection(
            group,
            profile.name,
            None,
            {},
            f"no {','.join(profile.modes)} config reaches {profile.min_ctx}; max ctx is {max_ctx}",
            profile,
        )
    passing = [
        candidate
        for candidate in passing
        if candidate.pp_tps >= profile.min_pp_tps and candidate.tg_tps >= profile.min_tg_tps
    ]
    if not passing:
        speed_candidates = [candidate for candidate in candidates if candidate.mode in profile.modes]
        max_pp = max((candidate.pp_tps for candidate in speed_candidates), default=0.0)
        max_tg = max((candidate.tg_tps for candidate in speed_candidates), default=0.0)
        return ProfileSelection(
            group,
            profile.name,
            None,
            {},
            f"no {','.join(profile.modes)} config reaches "
            f"pp>={profile.min_pp_tps:.0f} and tg>={profile.min_tg_tps:.0f}; "
            f"max pp is {max_pp:.1f}, max tg is {max_tg:.1f}",
            profile,
        )

    scored = score_candidates(passing, profile, quality_candidates=candidates)
    recommendation = max(scored, key=lambda row: row.score)
    alternatives = _alternatives(scored, recommendation, profile.alternatives)
    return ProfileSelection(group, profile.name, recommendation, alternatives, None, profile)


def score_candidates(
    candidates: Sequence[Candidate], profile: Profile, *, quality_candidates: Sequence[Candidate] | None = None
) -> list[ScoredCandidate]:
    max_ctx = max(candidate.ctx for candidate in candidates)
    max_pp = max(candidate.pp_tps for candidate in candidates)
    max_tg = max(candidate.tg_tps for candidate in candidates)
    qualities = normalise_qualities(candidates, quality_candidates or candidates, lambda c: c.key)
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        quality = qualities[candidate.key]
        score = (
            profile.weights["quality"] * quality.score
            + profile.weights["ctx"] * _context_score(candidate.ctx, profile.min_ctx, max_ctx)
            + profile.weights["pp"] * _ratio(candidate.pp_tps, max_pp)
            + profile.weights["tg"] * _ratio(candidate.tg_tps, max_tg)
        ) * _soft_target_scale(candidate.tg_tps, profile.target_tg_tps)
        scored.append(ScoredCandidate(candidate, quality, score))
    return scored


def _soft_target_scale(value: float, target: float) -> float:
    if target <= 0.0 or value >= target:
        return 1.0
    return max(0.0, value / target)


def _alternatives(
    scored: Sequence[ScoredCandidate],
    recommendation: ScoredCandidate,
    names: Sequence[AlternativeName],
) -> dict[AlternativeName, ScoredCandidate]:
    alternatives: dict[AlternativeName, ScoredCandidate] = {}
    for name in names:
        selected = _alternative(scored, recommendation, name)
        if selected is not None and selected.candidate.key != recommendation.candidate.key:
            alternatives[name] = selected
    return alternatives


def _alternative(
    scored: Sequence[ScoredCandidate], recommendation: ScoredCandidate, name: AlternativeName
) -> ScoredCandidate | None:
    if name == "faster":
        return _faster_alternative(scored, recommendation)
    if name == "higher-quality":
        return _higher_quality_alternative(scored, recommendation)
    if name == "more-context":
        return _more_context_alternative(scored, recommendation)
    if name == "fastest":
        return max(scored, key=lambda row: (row.candidate.tg_tps, row.candidate.pp_tps))
    if name == "best-quality":
        return max(scored, key=lambda row: (row.quality.score, row.score))
    if name == "highest-context":
        return max(scored, key=lambda row: (row.candidate.ctx, row.score))


def _faster_alternative(
    scored: Sequence[ScoredCandidate], recommendation: ScoredCandidate
) -> ScoredCandidate | None:
    candidates = [
        row
        for row in scored
        if row.candidate.key != recommendation.candidate.key
        and (
            row.candidate.tg_tps > recommendation.candidate.tg_tps
            or row.candidate.pp_tps > recommendation.candidate.pp_tps
        )
    ]
    return max(candidates, key=lambda row: row.score, default=None)


def _higher_quality_alternative(
    scored: Sequence[ScoredCandidate], recommendation: ScoredCandidate
) -> ScoredCandidate | None:
    candidates = [
        row
        for row in scored
        if row.candidate.key != recommendation.candidate.key
        and row.quality.score > recommendation.quality.score
    ]
    return max(candidates, key=lambda row: row.score, default=None)


def _more_context_alternative(
    scored: Sequence[ScoredCandidate], recommendation: ScoredCandidate
) -> ScoredCandidate | None:
    candidates = [
        row
        for row in scored
        if row.candidate.key != recommendation.candidate.key
        if row.candidate.ctx > recommendation.candidate.ctx
    ]
    return max(candidates, key=lambda row: row.score, default=None)


def _ratio(value: float | int, maximum: float | int) -> float:
    return float(value) / float(maximum) if maximum else 0.0


def _context_score(ctx: int, min_ctx: int, max_ctx: int) -> float:
    if max_ctx <= min_ctx:
        return 1.0
    return log(ctx / min_ctx) / log(max_ctx / min_ctx)


def _candidate_from_row(
    row: Mapping[str, str], group: str, mode: Mode, kld: float | None
) -> Candidate | None:
    ctx = parse_ctx(row.get("ctx"))
    ubatch = _parse_int(row.get("ubatch"))
    pp_tps = _parse_float(row.get(PP_COL))
    tg_tps = _parse_float(row.get(TG_COL))
    if ctx is None or ubatch is None or pp_tps is None or tg_tps is None:
        return None
    return Candidate(
        group=group,
        model=row.get("model", ""),
        quant=row.get("quant", ""),
        provider=row.get("provider", ""),
        mode=mode,
        ctx=ctx,
        ubatch=ubatch,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        params=_parse_params(row.get("params")),
        size_gib=_parse_float(row.get("size_gib")),
        kld=kld,
    )


def _load_kld(path: str) -> dict[tuple[str, str, str], float]:
    rows: dict[tuple[str, str, str], float] = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "")
            quant = row.get("quant", "")
            provider = row.get("provider", "")
            kld = _parse_float(row.get("kld"))
            if kld is not None:
                rows[(model, quant, provider)] = kld
    return rows


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_params(value: str | None) -> int | None:
    if value is None or value == "" or value == "?":
        return None
    value = value.strip().upper()
    multiplier = 1
    if value.endswith("T"):
        multiplier = 1_000_000_000_000
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]
    return int(float(value) * multiplier)


def _candidate_sort_key(candidate: Candidate) -> tuple[str, str, int, str, str, int]:
    return (
        candidate.group,
        candidate.model,
        QUANT_ORDER.get(candidate.quant, UNKNOWN_QUANT_ORDER),
        candidate.provider,
        candidate.mode,
        candidate.ubatch,
    )

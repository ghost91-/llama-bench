from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, log, log2
from statistics import median
from typing import TYPE_CHECKING, Literal, TypeAlias

from llama_bench.quant_order import QUANT_POSITION, canonical_quant

if TYPE_CHECKING:
    from llama_bench.selection import Candidate

QualitySource: TypeAlias = Literal[
    "measured-kld", "estimated-kld", "close-provider-kld", "cross-provider-kld", "quant-proxy"
]

MIN_CLOSE_PROVIDER_OVERLAP = 2
MAX_CLOSE_PROVIDER_AVG_LOG_DELTA = 0.05
MAX_CLOSE_PROVIDER_LOG_DELTA = 0.10


@dataclass(frozen=True)
class Quality:
    score: float
    source: QualitySource
    kld: float | None


def estimate_quality(candidate: Candidate, group_candidates: Sequence[Candidate]) -> Quality:
    if candidate.kld is not None:
        return Quality(kld_quality(candidate.kld), "measured-kld", candidate.kld)

    provider_candidates = [row for row in group_candidates if row.provider == candidate.provider]
    same_provider_estimated = estimate_kld_for_provider(
        candidate.quant, provider_candidates, allow_extrapolate=False
    )
    if same_provider_estimated is not None:
        return Quality(kld_quality(same_provider_estimated), "estimated-kld", same_provider_estimated)

    close_provider_estimated = estimate_close_provider_kld(candidate, group_candidates)
    if close_provider_estimated is not None:
        return Quality(kld_quality(close_provider_estimated), "close-provider-kld", close_provider_estimated)

    cross_provider_interpolated = estimate_cross_provider_kld(
        candidate, group_candidates, allow_extrapolate=False
    )
    if cross_provider_interpolated is not None:
        return Quality(kld_quality(cross_provider_interpolated), "cross-provider-kld", cross_provider_interpolated)

    same_provider_extrapolated = estimate_kld_for_provider(
        candidate.quant, provider_candidates, allow_extrapolate=True
    )
    if same_provider_extrapolated is not None:
        return Quality(kld_quality(same_provider_extrapolated), "estimated-kld", same_provider_extrapolated)

    cross_provider_estimated = estimate_cross_provider_kld(
        candidate, group_candidates, allow_extrapolate=True
    )
    if cross_provider_estimated is not None:
        return Quality(kld_quality(cross_provider_estimated), "cross-provider-kld", cross_provider_estimated)

    return Quality(quant_proxy_quality(candidate), "quant-proxy", None)


CandidateKey = tuple[str, str, str, str, int, int]


def normalise_qualities(
    candidates: Sequence[Candidate],
    quality_candidates: Sequence[Candidate],
    candidate_key: Callable[[Candidate], CandidateKey],
) -> dict[CandidateKey, Quality]:
    qualities = {candidate_key(candidate): estimate_quality(candidate, quality_candidates) for candidate in candidates}
    kld_values = [quality.kld for quality in qualities.values() if quality.kld is not None]
    if kld_values:
        best = min(log(kld) for kld in kld_values)
        worst = max(log(kld) for kld in kld_values)
        return {
            key: Quality(_normalise_log_kld(quality.kld, best, worst), quality.source, quality.kld)
            for key, quality in qualities.items()
        }
    return qualities


def estimate_kld_for_provider(
    quant: str, provider_candidates: Sequence[Candidate], *, allow_extrapolate: bool
) -> float | None:
    target_key = _equivalent_quant_key(quant)
    same_quant = [
        row.kld
        for row in provider_candidates
        if _equivalent_quant_key(row.quant) == target_key and row.kld is not None
    ]
    if same_quant:
        return sum(same_quant) / len(same_quant)

    target_position = QUANT_POSITION.get(canonical_quant(quant))
    if target_position is None:
        measured = [row.kld for row in provider_candidates if row.kld is not None]
        return median(measured) if measured else None

    curve = quant_kld_curve(provider_candidates)
    if not curve:
        return None
    if len(curve) == 1:
        return curve[0][1] if allow_extrapolate else None
    log_kld = interpolate_log_kld(target_position, curve, allow_extrapolate=allow_extrapolate)
    return exp(log_kld) if log_kld is not None else None


def estimate_cross_provider_kld(
    candidate: Candidate, group_candidates: Sequence[Candidate], *, allow_extrapolate: bool
) -> float | None:
    estimates: list[float] = []
    providers = sorted({row.provider for row in group_candidates if row.provider != candidate.provider})
    for provider in providers:
        provider_candidates = [row for row in group_candidates if row.provider == provider]
        estimate = estimate_kld_for_provider(
            candidate.quant, provider_candidates, allow_extrapolate=allow_extrapolate
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def estimate_close_provider_kld(
    candidate: Candidate, group_candidates: Sequence[Candidate]
) -> float | None:
    estimates: list[float] = []
    target_key = _equivalent_quant_key(candidate.quant)
    providers = sorted({row.provider for row in group_candidates if row.provider != candidate.provider})
    for provider in providers:
        if not providers_are_close(candidate.provider, provider, group_candidates):
            continue
        provider_candidates = [row for row in group_candidates if row.provider == provider]
        same_quant = [
            row.kld
            for row in provider_candidates
            if _equivalent_quant_key(row.quant) == target_key and row.kld is not None
        ]
        if same_quant:
            estimates.append(sum(same_quant) / len(same_quant))
    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def providers_are_close(left: str, right: str, group_candidates: Sequence[Candidate]) -> bool:
    left_klds = provider_klds_by_equivalent_quant(left, group_candidates)
    right_klds = provider_klds_by_equivalent_quant(right, group_candidates)
    overlapping = sorted(set(left_klds) & set(right_klds))
    if len(overlapping) < MIN_CLOSE_PROVIDER_OVERLAP:
        return False
    log_deltas = [abs(log(left_klds[quant] / right_klds[quant])) for quant in overlapping]
    return (
        sum(log_deltas) / len(log_deltas) <= MAX_CLOSE_PROVIDER_AVG_LOG_DELTA
        and max(log_deltas) <= MAX_CLOSE_PROVIDER_LOG_DELTA
    )


def provider_klds_by_equivalent_quant(
    provider: str, group_candidates: Sequence[Candidate]
) -> dict[str, float]:
    by_quant: dict[str, list[float]] = {}
    for row in group_candidates:
        if row.provider == provider and row.kld is not None:
            by_quant.setdefault(_equivalent_quant_key(row.quant), []).append(row.kld)
    return {quant: median(values) for quant, values in by_quant.items()}


def quant_kld_curve(group_candidates: Sequence[Candidate]) -> list[tuple[float, float]]:
    by_position: dict[float, list[float]] = {}
    for row in group_candidates:
        position = QUANT_POSITION.get(canonical_quant(row.quant))
        if position is not None and row.kld is not None and row.kld > 0:
            by_position.setdefault(position, []).append(row.kld)
    points = [(position, median(values)) for position, values in by_position.items()]
    points.sort()
    return _monotone_nonincreasing(points)


def interpolate_log_kld(
    target_position: float, curve: Sequence[tuple[float, float]], *, allow_extrapolate: bool = True
) -> float | None:
    if target_position <= curve[0][0]:
        if not allow_extrapolate and target_position < curve[0][0]:
            return None
        left, right = curve[0], curve[1]
        left_position, right_position = left[0], right[0]
        target_position = max(target_position, left_position - (right_position - left_position))
    elif target_position >= curve[-1][0]:
        if not allow_extrapolate and target_position > curve[-1][0]:
            return None
        left, right = curve[-2], curve[-1]
        left_position, right_position = left[0], right[0]
        target_position = min(target_position, right_position + (right_position - left_position))
    else:
        bracket = (curve[0], curve[1])
        for idx, candidate_right in enumerate(curve[1:], start=1):
            candidate_left = curve[idx - 1]
            if candidate_left[0] <= target_position <= candidate_right[0]:
                bracket = (candidate_left, candidate_right)
                break
        left, right = bracket
    left_position, left_kld = left
    right_position, right_kld = right
    if left_position == right_position:
        return log(left_kld)
    weight = (target_position - left_position) / (right_position - left_position)
    return log(left_kld) + weight * (log(right_kld) - log(left_kld))


def kld_quality(kld: float) -> float:
    return max(0.0, min(1.0, exp(-30.0 * kld)))


def quant_proxy_quality(candidate: Candidate) -> float:
    quant_position = QUANT_POSITION.get(canonical_quant(candidate.quant), QUANT_POSITION["Q3_K_M"])
    params = candidate.params or 8_000_000_000
    params_b = params / 1_000_000_000
    midpoint = 4.0 + 0.7 * max(0.0, log2(8.0 / params_b))
    return max(0.05, min(1.0, 1.0 / (1.0 + exp(-1.35 * (quant_position - midpoint)))))


def _equivalent_quant_key(quant: str) -> str:
    return canonical_quant(quant)


def _normalise_log_kld(kld: float | None, best: float, worst: float) -> float:
    if kld is None:
        return 0.0
    if best == worst:
        return 1.0
    return (worst - log(kld)) / (worst - best)


def _monotone_nonincreasing(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    smoothed: list[tuple[float, float]] = []
    best_so_far = float("inf")
    for order, kld in points:
        best_so_far = min(best_so_far, kld)
        smoothed.append((order, best_so_far))
    return smoothed

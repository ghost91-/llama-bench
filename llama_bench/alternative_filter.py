from llama_bench.selection import AlternativeName, ScoredCandidate

MIN_HIGHER_QUALITY_GAIN = 0.10
SIMILAR_REL_TOLERANCE = 0.05
SIMILAR_QUALITY_TOLERANCE = 0.03


def drop_near_duplicate(
    alternatives: dict[AlternativeName, ScoredCandidate], *, near: AlternativeName, extreme: AlternativeName
) -> None:
    near_scored = alternatives.get(near)
    extreme_scored = alternatives.get(extreme)
    if near_scored is not None and extreme_scored is not None and similar_choice(near_scored, extreme_scored):
        del alternatives[near]


def drop_redundant_q8_k_xl(
    alternatives: dict[AlternativeName, ScoredCandidate], recommendation: ScoredCandidate
) -> None:
    has_q8_0_quality_option = is_q8_0(recommendation.candidate.quant) or any(
        name in {"higher-quality", "best-quality"} and is_q8_0(scored.candidate.quant)
        for name, scored in alternatives.items()
    )
    has_high_quality_default = recommendation.quality.score >= 0.90
    if not has_q8_0_quality_option and not has_high_quality_default:
        return

    for name, scored in list(alternatives.items()):
        if name in {"higher-quality", "best-quality"} and is_q8_k_xl(scored.candidate.quant):
            del alternatives[name]


def is_q8_0(quant: str) -> bool:
    return quant == "Q8_0"


def is_q8_k_xl(quant: str) -> bool:
    return quant in {"Q8_K_XL", "UD-Q8_K_XL"}


def is_material_alternative(
    name: str, recommendation: ScoredCandidate, alternative: ScoredCandidate
) -> bool:
    if name in {"higher-quality", "best-quality"}:
        return alternative.quality.score >= recommendation.quality.score + MIN_HIGHER_QUALITY_GAIN
    return True


def similar_choice(left: ScoredCandidate, right: ScoredCandidate) -> bool:
    left_candidate = left.candidate
    right_candidate = right.candidate
    return (
        left_candidate.mode == right_candidate.mode
        and left_candidate.ctx == right_candidate.ctx
        and relative_close(left_candidate.pp_tps, right_candidate.pp_tps)
        and relative_close(left_candidate.tg_tps, right_candidate.tg_tps)
        and abs(left.quality.score - right.quality.score) <= SIMILAR_QUALITY_TOLERANCE
    )


def relative_close(left: float, right: float) -> bool:
    baseline = max(abs(left), abs(right))
    return baseline == 0.0 or abs(left - right) / baseline <= SIMILAR_REL_TOLERANCE

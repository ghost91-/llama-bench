#!/usr/bin/env python3
import argparse
from collections.abc import Iterable, Sequence

from llama_bench.alternative_filter import (
    MIN_HIGHER_QUALITY_GAIN as MIN_HIGHER_QUALITY_GAIN,
    SIMILAR_QUALITY_TOLERANCE as SIMILAR_QUALITY_TOLERANCE,
    SIMILAR_REL_TOLERANCE as SIMILAR_REL_TOLERANCE,
    drop_near_duplicate,
    drop_redundant_q8_k_xl,
    is_material_alternative,
    is_q8_0,
    is_q8_k_xl,
    relative_close,
    similar_choice,
)
from llama_bench.consolidation import (
    ConfigKey,
    LabelledConfig,
    ReportEntry,
    consolidate_entries,
    config_key,
    entry_priority,
    relative_deficit,
    reuse_loss,
    reuse_tolerances,
)
from llama_bench.labels import (
    description_for_config,
    label_for_config,
    label_stem,
    slug,
)
from llama_bench.results import format_ctx
from llama_bench.selection import (
    PROFILES,
    ProfileSelection,
    ScoredCandidate,
    load_candidates,
    select_profiles,
)

_entry_priority = entry_priority
_config_key = config_key
_consolidate_entries = consolidate_entries
_slug = slug
_label_for_config = label_for_config
_label_stem = label_stem
_description_for_config = description_for_config
_relative_deficit = relative_deficit
_reuse_loss = reuse_loss
_reuse_tolerances = reuse_tolerances
_drop_near_duplicate = drop_near_duplicate
_drop_redundant_q8_k_xl = drop_redundant_q8_k_xl
_is_material_alternative = is_material_alternative
_similar_choice = similar_choice
_relative_close = relative_close
_is_q8_0 = is_q8_0
_is_q8_k_xl = is_q8_k_xl

TableRow = tuple[str, str, str, str, str, str, str, str, str, str, str, str]
ReverseRow = tuple[str, str, str, str, str, str, str, str, str, str, str, str, str]


def render_selection(selection: ProfileSelection) -> list[str]:
    lines: list[str] = []
    lines.append(selection.group)
    if selection.recommendation is None:
        lines.append(f"  skipped: {selection.skipped_reason or 'no recommendation'}")
        return lines

    lines.append(f"  recommended: {_format_scored(selection.recommendation)}")
    if selection.alternatives:
        lines.append("  alternatives:")
        for name, candidate in selection.alternatives.items():
            lines.append(f"    {name}: {_format_scored(candidate)}")
    return lines


def _format_scored(scored: ScoredCandidate) -> str:
    candidate = scored.candidate
    quality = scored.quality
    kld = f" kld={quality.kld:.4g}" if quality.kld is not None else ""
    return (
        f"{candidate.provider} {candidate.quant} {candidate.mode} ub={candidate.ubatch} "
        f"ctx={format_ctx(candidate.ctx)} pp={candidate.pp_tps:.1f} tg={candidate.tg_tps:.1f} "
        f"quality={quality.score:.2f}/{quality.source}{kld} score={scored.score:.3f}"
    )


def render_table(selections: Sequence[ProfileSelection]) -> str:
    header: TableRow = (
        "profile",
        "group",
        "status",
        "provider",
        "quant",
        "mode",
        "ubatch",
        "ctx",
        "pp",
        "tg",
        "quality",
        "score / reason",
    )
    rows: list[TableRow] = [header]
    for selection in selections:
        rows.append(_table_row(selection))
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    lines = [_render_table_row(header, widths), _render_separator(widths)]
    lines.extend(_render_table_row(row, widths) for row in rows[1:])
    return "\n".join(lines)


def render_markdown(selections: Sequence[ProfileSelection]) -> str:
    header: TableRow = (
        "Profile",
        "Group",
        "Choice",
        "Provider",
        "Quant",
        "Mode",
        "Ubatch",
        "Ctx",
        "PP",
        "TG",
        "Quality",
        "Score",
    )
    rows: list[TableRow] = [header]
    for entry in _iter_report_entries(selections):
        rows.append(_markdown_row(entry.selection, entry.choice, entry.scored))
    lines = [_render_markdown_row(header), _render_markdown_separator(len(header))]
    lines.extend(_render_markdown_row(row) for row in rows[1:])
    return "\n".join(lines)


def render_reverse_markdown(
    selections: Sequence[ProfileSelection], *, consolidate: bool = False, max_configs_per_group: int = 5
) -> str:
    header: ReverseRow = (
        "Label",
        "Description",
        "Group",
        "Provider",
        "Quant",
        "Mode",
        "Ubatch",
        "Ctx",
        "PP",
        "TG",
        "Quality",
        "Uses",
        "Count",
    )
    configs = build_labelled_configs(
        selections, consolidate=consolidate, max_configs_per_group=max_configs_per_group
    )

    rows: list[ReverseRow] = [header]
    for config in configs:
        key = config.key
        uses = config.entries
        group, provider, quant, mode, ubatch, ctx, pp_tps, tg_tps = key
        rows.append(
            (
                config.label,
                config.description,
                group,
                provider,
                quant,
                mode,
                str(ubatch),
                format_ctx(ctx),
                f"{pp_tps:.1f}",
                f"{tg_tps:.1f}",
                _quality_summary([entry.scored for entry in uses]),
                "<br>".join(f"{entry.selection.profile}: {entry.choice}" for entry in uses),
                str(len(uses)),
            )
        )
    lines = [_render_markdown_row(header), _render_markdown_separator(len(header))]
    lines.extend(_render_markdown_row(row) for row in rows[1:])
    return "\n".join(lines)


def build_labelled_configs(
    selections: Sequence[ProfileSelection], *, consolidate: bool = False, max_configs_per_group: int = 5
) -> list[LabelledConfig]:
    entries = list(_iter_report_entries(selections))
    if consolidate:
        entries = consolidate_entries(entries, max_configs_per_group=max_configs_per_group)

    grouped: dict[ConfigKey, list[ReportEntry]] = {}
    for entry in entries:
        grouped.setdefault(config_key(entry.selection.group, entry.scored), []).append(entry)

    items = sorted(grouped.items(), key=_reverse_sort_key)
    used_labels: set[str] = set()
    configs: list[LabelledConfig] = []
    for key, uses in items:
        sorted_uses = tuple(sorted(uses, key=entry_priority))
        label = label_for_config(key, sorted_uses, used_labels)
        used_labels.add(label)
        configs.append(
            LabelledConfig(
                label=label,
                description=description_for_config(key, sorted_uses),
                key=key,
                entries=sorted_uses,
            )
        )
    return configs


def _iter_report_entries(
    selections: Sequence[ProfileSelection],
) -> Iterable[ReportEntry]:
    for selection in selections:
        if selection.recommendation is None:
            continue
        yield ReportEntry(selection, "recommended", selection.recommendation)
        seen = {selection.recommendation.candidate.key}
        for name, scored in _report_alternatives(selection):
            key = scored.candidate.key
            if key in seen:
                continue
            seen.add(key)
            yield ReportEntry(selection, name, scored)


def _reverse_sort_key(
    item: tuple[ConfigKey, list[ReportEntry]],
) -> tuple[str, int, str, str, str]:
    key, uses = item
    group, provider, quant, mode, ubatch, _ctx, _pp_tps, _tg_tps = key
    return (group, -len(uses), provider, quant, f"{mode}:{ubatch}")


def _report_alternatives(
    selection: ProfileSelection,
) -> list[tuple[str, ScoredCandidate]]:
    if selection.recommendation is None:
        return []
    alternatives = dict(selection.alternatives)
    drop_near_duplicate(alternatives, near="faster", extreme="fastest")
    drop_near_duplicate(alternatives, near="higher-quality", extreme="best-quality")
    drop_near_duplicate(alternatives, near="more-context", extreme="highest-context")
    drop_redundant_q8_k_xl(alternatives, selection.recommendation)
    return [
        (name, scored)
        for name, scored in alternatives.items()
        if is_material_alternative(name, selection.recommendation, scored)
    ]


def _markdown_row(selection: ProfileSelection, choice: str, scored: ScoredCandidate) -> TableRow:
    candidate = scored.candidate
    return (
        selection.profile,
        selection.group,
        choice,
        candidate.provider,
        candidate.quant,
        candidate.mode,
        str(candidate.ubatch),
        format_ctx(candidate.ctx),
        f"{candidate.pp_tps:.1f}",
        f"{candidate.tg_tps:.1f}",
        _quality_text(scored),
        f"{scored.score:.3f}",
    )


def _quality_text(scored: ScoredCandidate) -> str:
    quality = scored.quality
    text = f"{quality.score:.2f}/{quality.source}"
    if quality.kld is not None:
        text += f"/{quality.kld:.4g}"
    return text


def _table_row(selection: ProfileSelection) -> TableRow:
    if selection.recommendation is None:
        return (
            selection.profile,
            selection.group,
            "skip",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            selection.skipped_reason or "no recommendation",
        )
    scored = selection.recommendation
    candidate = scored.candidate
    quality = scored.quality
    quality_text = f"{quality.score:.2f}/{quality.source}"
    if quality.kld is not None:
        quality_text += f"/{quality.kld:.4g}"
    return (
        selection.profile,
        selection.group,
        "ok",
        candidate.provider,
        candidate.quant,
        candidate.mode,
        str(candidate.ubatch),
        format_ctx(candidate.ctx),
        f"{candidate.pp_tps:.1f}",
        f"{candidate.tg_tps:.1f}",
        quality_text,
        f"{scored.score:.3f}",
    )


def _render_table_row(row: TableRow, widths: Sequence[int]) -> str:
    return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))


def _render_separator(widths: Sequence[int]) -> str:
    return " | ".join("-" * width for width in widths)


def _render_markdown_row(row: Sequence[str]) -> str:
    return "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |"


def _render_markdown_separator(columns: int) -> str:
    return "| " + " | ".join("---" for _ in range(columns)) + " |"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _quality_summary(scored_rows: Sequence[ScoredCandidate]) -> str:
    quality_texts = sorted({_quality_text(scored) for scored in scored_rows})
    if len(quality_texts) == 1:
        return quality_texts[0]
    scores = [scored.quality.score for scored in scored_rows]
    source = scored_rows[0].quality.source
    kld_values = {scored.quality.kld for scored in scored_rows}
    suffix = f"/{next(iter(kld_values)):.4g}" if len(kld_values) == 1 and None not in kld_values else ""
    return f"{min(scores):.2f}-{max(scores):.2f}/{source}{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report recommended benchmark configs by profile.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES),
        help="profile to report; repeatable; defaults to all profiles",
    )
    parser.add_argument("--group", action="append", help="model group to report; repeatable")
    parser.add_argument("--reverse", action="store_true", help="group output by model/configuration")
    parser.add_argument("--consolidate", action="store_true", help="reuse close-enough configs in reverse output")
    parser.add_argument("--max-configs-per-group", type=int, default=5, help="soft consolidation target per model group")
    parser.add_argument("--results", default=None, help="fit-bench-results.csv path")
    parser.add_argument("--kld", default=None, help="kld-results.csv path")
    args = parser.parse_args(argv)

    if args.results is not None and args.kld is not None:
        candidates = load_candidates(args.results, args.kld)
    elif args.results is not None:
        candidates = load_candidates(args.results)
    elif args.kld is not None:
        candidates = load_candidates(kld_file=args.kld)
    else:
        candidates = load_candidates()
    profiles = [PROFILES[name] for name in args.profile] if args.profile else list(PROFILES.values())
    selections = select_profiles(candidates, profiles=profiles)
    if args.group:
        groups = set(args.group)
        selections = [selection for selection in selections if selection.group in groups]

    if args.reverse:
        print(
            render_reverse_markdown(
                selections,
                consolidate=args.consolidate,
                max_configs_per_group=args.max_configs_per_group,
            )
        )
    else:
        print(render_markdown(selections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

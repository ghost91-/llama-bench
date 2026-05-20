from pathlib import Path

import select_configs
from llama_bench.selection import load_candidates, select_profiles


ROOT = Path(__file__).resolve().parent.parent


def test_forward_report_matches_snapshot() -> None:
    selections = select_profiles(load_candidates())

    assert select_configs.render_markdown(selections) == _snapshot("selection-report.md")


def test_reverse_report_matches_snapshot() -> None:
    selections = select_profiles(load_candidates())


    assert select_configs.render_reverse_markdown(selections) == _snapshot(
        "selection-report-reverse.md"
    )


def test_consolidated_report_matches_snapshot() -> None:
    selections = select_profiles(load_candidates())

    assert select_configs.render_reverse_markdown(selections, consolidate=True) == _snapshot(
        "selection-report-consolidated.md"
    )


def _snapshot(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").rstrip("\n")

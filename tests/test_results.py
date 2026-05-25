import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pytest import MonkeyPatch

import llama_bench.results as results


def test_parse_and_format_helpers() -> None:
    assert results.parse_ctx(None) is None
    assert results.parse_ctx("") is None
    assert results.parse_ctx("   ") is None
    assert results.parse_ctx("150k") == 150000
    assert results.parse_ctx("5000") == 5000
    assert results.format_ctx(5000) == "5k"
    assert results.format_ctx(5500) == "5500"
    assert results.format_ngl(-1) == "all"
    assert results.format_params(7_000_000_000) == "7B"
    assert results.format_params(1_500_000_000) == "1.5B"
    assert results.format_mmproj(64) == "64M"


def test_parse_ctx_rejects_legacy_and_fractional_values() -> None:
    with pytest.raises(ValueError):
        results.parse_ctx(" ? ")
    with pytest.raises(ValueError):
        results.parse_ctx("1.5k")


def test_result_csv_fieldnames_include_fit_configuration() -> None:
    assert "fit_target" in results.CSV_FIELDNAMES
    assert "moe" in results.CSV_FIELDNAMES


def test_append_result_row_merges_matching_rows(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))

    first_row = {
        "model": "Foo",
        "quant": "Q4_K_M",
        "provider": "unsloth",
        "mode": "text",
        "ubatch": "512",
        "fit_target": "128",
        "ctx": "5000",
        results.PP_COL: "10.5",
        "bench_ts": "2026-01-01T00:00:00+00:00",
    }
    second_row = {
        "model": "Foo",
        "quant": "Q4_K_M",
        "provider": "unsloth",
        "mode": "text",
        "ubatch": "512",
        "fit_target": "192",
        "ctx": "10000",
        results.TG_COL: "4.2",
    }

    results.append_result_row(first_row)
    results.append_result_row(second_row)

    with results_file.open(newline="") as f:
        rows = [{key: value or "" for key, value in row.items()} for row in csv.DictReader(f)]

    assert len(rows) == 1
    assert rows[0]["fit_target"] == "192"
    assert rows[0]["ctx"] == "10000"
    assert rows[0][results.PP_COL] == "10.5"
    assert rows[0][results.TG_COL] == "4.2"
    assert results.get_bench_ts("unsloth/Foo:Q4_K_M", ubatch=512) == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )


def test_append_result_row_does_not_overwrite_with_empty_values(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))

    results.append_result_row(
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "5000",
            results.PP_COL: "10.5",
        }
    )
    results.append_result_row(
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "",
            results.PP_COL: None,
        }
    )

    with results_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["ctx"] == "5000"
    assert rows[0][results.PP_COL] == "10.5"


def test_append_result_row_keeps_different_mode_and_ubatch_separate(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))
    base = {
        "model": "Foo",
        "quant": "Q4_K_M",
        "provider": "unsloth",
        "ctx": "5000",
    }

    results.append_result_row({**base, "mode": "text", "ubatch": "512"})
    results.append_result_row({**base, "mode": "vision", "ubatch": "512"})
    results.append_result_row({**base, "mode": "text", "ubatch": "1024"})

    with results_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert [(row["mode"], row["ubatch"]) for row in rows] == [
        ("text", "512"),
        ("vision", "512"),
        ("text", "1024"),
    ]


def test_append_result_row_drops_existing_and_new_unknown_columns(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*results.CSV_FIELDNAMES, "legacy"])
        writer.writeheader()
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "text",
                "ubatch": "512",
                "legacy": "old",
            }
        )

    results.append_result_row(
        {
            "model": "Foo",
            "quant": "Q4_K_M",
            "provider": "unsloth",
            "mode": "text",
            "ubatch": "512",
            "ctx": "5000",
            "new_col": "new",
        }
    )

    with results_file.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "legacy" not in reader.fieldnames
    assert "new_col" not in reader.fieldnames
    assert rows[0]["ctx"] == "5000"


def test_load_models_and_tags(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    models_file = tmp_path / "models.toml"
    models_file.write_text(
        """
[[models]]
repo = "unsloth/Foo-GGUF"
quant = "Q4_K_M"
group = "foo"

[[models]]
repo = "bartowski/Bar-GGUF"
quant = "Q5_K_M"
group = "bar"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(results, "MODELS_TOML", str(models_file))

    assert results.load_models() == [
        ("unsloth/Foo-GGUF", "Q4_K_M", "foo"),
        ("bartowski/Bar-GGUF", "Q5_K_M", "bar"),
    ]
    assert results.load_tags() == ["unsloth/Foo-GGUF:Q4_K_M", "bartowski/Bar-GGUF:Q5_K_M"]


def test_unknown_quant_sorts_after_all_known_quants() -> None:
    from llama_bench.quant_order import UNKNOWN_QUANT_ORDER, quant_sort_key

    assert quant_sort_key("Q8_K_XL") < UNKNOWN_QUANT_ORDER
    assert quant_sort_key("IQ3_S-3.00bpw") == quant_sort_key("IQ3_S")
    assert quant_sort_key("TOTALLY_UNKNOWN") == UNKNOWN_QUANT_ORDER


def test_sort_results_file_places_unknown_quant_last(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))
    rows = [
        {"model": "B", "quant": "Q8_0", "provider": "zz", "mode": "vision", "params": "?", "ubatch": "512"},
        {"model": "A", "quant": "Q5_K_M", "provider": "bartowski", "mode": "text", "params": "7B", "ubatch": "1024"},
        {"model": "A", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "7B", "ubatch": "512"},
        {"model": "C", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "500M", "ubatch": "512"},
    ]
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "quant", "provider", "mode", "params", "ubatch"])
        writer.writeheader()
        writer.writerows(rows)

    results.sort_results_file()

    with results_file.open(newline="") as f:
        sorted_rows = list(csv.DictReader(f))
    assert [(row["model"], row["quant"], row["provider"], row["mode"]) for row in sorted_rows] == [
        ("C", "Q4_K_M", "unsloth", "text"),
        ("A", "Q4_K_M", "unsloth", "text"),
        ("A", "Q5_K_M", "bartowski", "text"),
        ("B", "Q8_0", "zz", "vision"),
    ]


def test_sort_results_file_orders_param_units_mode_and_ubatch_ties(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))
    rows = [
        {"model": "D", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "?", "ubatch": "512"},
        {"model": "C", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "1T", "ubatch": "512"},
        {"model": "A", "quant": "Q4_K_M", "provider": "unsloth", "mode": "vision", "params": "7B", "ubatch": "512"},
        {"model": "A", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "7B", "ubatch": "1024"},
        {"model": "A", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "7B", "ubatch": "512"},
        {"model": "B", "quant": "Q4_K_M", "provider": "unsloth", "mode": "text", "params": "500M", "ubatch": "512"},
    ]
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "quant", "provider", "mode", "params", "ubatch"])
        writer.writeheader()
        writer.writerows(rows)

    results.sort_results_file()

    with results_file.open(newline="") as f:
        sorted_rows = list(csv.DictReader(f))
    assert [(row["model"], row["mode"], row["ubatch"], row["params"]) for row in sorted_rows] == [
        ("B", "text", "512", "500M"),
        ("A", "text", "512", "7B"),
        ("A", "text", "1024", "7B"),
        ("A", "vision", "512", "7B"),
        ("C", "text", "512", "1T"),
        ("D", "text", "512", "?"),
    ]


def test_get_bench_ts_filters_mode_and_handles_missing_or_invalid_timestamps(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_file = tmp_path / "fit-bench-results.csv"
    monkeypatch.setattr(results, "RESULTS_FILE", str(results_file))

    assert results.get_bench_ts("unsloth/Foo-GGUF:Q4_K_M") is None
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "vision",
                "ubatch": "512",
                "bench_ts": "bad",
            }
        )
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "text",
                "ubatch": "1024",
                "bench_ts": "2026-02-03T04:05:06+00:00",
            }
        )
        writer.writerow(
            {
                "model": "Foo",
                "quant": "Q4_K_M",
                "provider": "unsloth",
                "mode": "text",
                "ubatch": "2048",
                "bench_ts": "2026-02-03T04:05:06+02:00",
            }
        )

    assert results.get_bench_ts("unsloth/Foo-GGUF:Q4_K_M", mode="vision", ubatch=512) is None
    assert results.get_bench_ts("unsloth/Foo-GGUF:Q4_K_M", ubatch=512) is None
    assert results.get_bench_ts("unsloth/Foo-GGUF:Q4_K_M", ubatch=1024) == datetime(
        2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc
    )
    assert results.get_bench_ts("unsloth/Foo-GGUF:Q4_K_M", ubatch=2048) == datetime(
        2026, 2, 3, 2, 5, 6, tzinfo=timezone.utc
    )

# llama-bench

Benchmark llama.cpp GGUF models to find optimal context sizes for a given GPU.

## Commands

Uses `uv run` (no manual venv). Lint: `uv run ruff check .`. No tests, no CI.

Log files: Keep all `.log` files in the `logs/` directory.

Setup: `uv sync` — requires `llama-fit-params`, `llama-bench`, `llama-server` on PATH.

Local llama.cpp sources are available at `~/Development/other/llama.cpp`; use that tree first when checking upstream flags, behaviour, or implementation details.

Model list: `models.toml` is the single source of truth for all target models (repo, quant, group).

## Gotchas

- Model tags: `repo:quant` format (e.g. `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`)
- `ngl == -1` means "all layers on GPU" (llama.cpp convention, displayed as `all`)
- MoE models: `llama-fit-params` can produce non-monotonic ngl — scan must run to completion
- `fit-bench-results.csv` rows merge in-place: vision cols update independently from text cols
- CSV may contain rows for models no longer in `models.toml` (e.g. Nemotron-3-Super-120B-A12B) — these are kept as historical data, not orphans
- Sort order for results defined in `results.py:QUANT_ORDER` and `PROVIDER_ORDER`
- Ruff line-length 99, Python >=3.11

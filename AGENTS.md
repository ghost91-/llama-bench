# llama-bench

Benchmark llama.cpp GGUF models to find optimal context sizes for a given GPU.

## Commands

Uses `uv run` (no manual venv).

- **Lint:** `uv run ruff check .`
- **Type check:** `uv run basedpyright`
- **Tests:** `uv run pytest`
- **Setup:** `uv sync` — requires `llama-fit-params`, `llama-bench`, `llama-server` on PATH
- **Run:** `uv run fit_bench.py [tags...] [flags]`
- **Download:** `uv run download_models.py`
- **Clean cache:** `uv run cleanup_model_cache.py`
- **KLD extract:** `uv run kld_extract.py`
- **Plot metrics:** `uv run plot_metrics.py`
- **Generate INI:** `uv run generate_models_ini.py`

Local llama.cpp sources at `~/Development/other/llama.cpp`; check that tree first for upstream flags, behaviour, or implementation details.

Model list: `models.toml` is the single source of truth for all target models (repo, quant, group).

## Context

Targets an RTX 4070 Laptop (8 GB VRAM) + 64 GB RAM. `fit_bench.py` reserves 256 MiB VRAM for non-model use (hybrid dGPU setup). With `--vision`, the fit-target becomes `256 + mmproj size`.

## Workflows

- No tags = all models from `models.toml`, filtered by `--provider`/`--group`
- `--scan`: scan only, write cache, don't benchmark
- `--rescan AGE`: re-scan if `scan_ts` older than AGE (e.g. `24h`, `7d`, `30m`). Default: use cache.
- `--rebench AGE`: re-bench if `bench_ts` older than AGE. Default: always bench.
- `--scan` + `--rebench` → error (incompatible)

## Domain reference

### Conventions

- Model tags: `repo:quant` format (e.g. `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`)
- `ngl == -1` means all layers on GPU (llama.cpp convention, displayed as `all`)
- `models.toml` is the single source of truth for target models (repo, quant, group)

### Data files

- `scan-cache.json`: scan results keyed by `repo:quant` tag. Top-level fields: `mmproj`, `moe`, `max_ctx`, `max_ctx_ts`, `caps`, `text`, `vision`. `caps` contains `vision` and `reasoning`; `reasoning` is either `false` or an object with `switchable` and `efforts`. Per-ubatch scan entries contain `fit_target`, `ctx`, `ngl`, `offload`, `ot`, `scan_ts`. Pruned to `models.toml` entries on every write.
- `fit-bench-results.csv`: benchmark results keyed by `(model, quant, provider, mode, ubatch)`. Current columns include `mode`, `ubatch`, `offload`, `pp4096_tps`, `pp4096_stddev_tps`, `tg128_tps`, `tg128_stddev_tps`, `reasoning`, `switchable`, `efforts`, `bench_ts`.
- `kld-results.csv`: consolidated KLD data.
- Runtime logs go in `logs/` (for example `logs/scan-text.log`, `logs/bench-vision.log`).

Never prune or clean up `fit-bench-results.csv` or `kld-results.csv`; the user manages those result datasets manually. Scripts may ignore rows that are not in `models.toml`, but agents must not delete or rewrite historical result rows as cleanup.

### Gotchas

- MoE models: `llama-fit-params` can produce non-monotonic ngl — scan must run to completion
- Active cache/output schema uses booleans and empty values; do not reintroduce legacy `yes` / `no` / `-` forms
- Cleanup/migration compatibility code is temporary: once data files are rewritten cleanly, remove the compatibility path instead of keeping it around
- Reasoning effort strings use `|`, not `/`
- Keep logging concise and non-redundant; `fit_bench.py` output has already been trimmed intentionally

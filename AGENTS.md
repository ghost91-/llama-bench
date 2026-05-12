# llama-bench

Benchmark llama.cpp GGUF models to find optimal context sizes for a given GPU.

## Commands

Uses `uv run` (no manual venv). No tests, no CI.

- **Lint:** `uv run ruff check .`
- **Setup:** `uv sync` — requires `llama-fit-params`, `llama-bench`, `llama-server` on PATH
- **Run:** `uv run fit_bench.py [tags...] [flags]`
- **Download:** `uv run download_models.py`
- **Clean cache:** `uv run cleanup_model_cache.py`
- **KLD extract:** `uv run kld_extract.py`
- **Plot KLD:** `uv run plot_kld.py`
- **Generate INI:** `uv run generate_models_ini.py`

Local llama.cpp sources at `~/Development/other/llama.cpp`; check that tree first for upstream flags, behaviour, or implementation details.

Model list: `models.toml` is the single source of truth for all target models (repo, quant, group).

## Context

Targets an RTX 4070 Laptop (8 GB VRAM) + 64 GB RAM. `fit_bench.py` reserves 128 MiB VRAM for non-model use (hybrid dGPU setup). With `--vision`, the fit-target becomes `128 + mmproj size`.

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

- `scan-cache.json`: scan results keyed by `repo:quant` tag, with `text`/`vision` sub-objects per mode. `has_vision` = capability flag (`"yes"`/`"no"`), `mmproj` = size string. Pruned to `models.toml` entries on every write.
- `fit-bench-results.csv`: bench results with runtime config conditions. Vision cols update independently from text cols.
- `kld-results.csv`: consolidated KLD data.

### Gotchas

- MoE models: `llama-fit-params` can produce non-monotonic ngl — scan must run to completion

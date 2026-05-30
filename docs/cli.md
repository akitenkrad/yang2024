**English** | [日本語](cli.ja.md)

# CLI

The Rust binary is `oasis`. Build with `cargo build --release`, then `cargo run --release -- <subcommand> [flags]`.

## LLM environment variables

The LLM layer is **Ollama first → OpenAI fallback** (`socsim-llm`). Configure via env vars; nothing is hardcoded:

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model |
| `OPENAI_API_KEY` | (unset) | enables the OpenAI fallback |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |

A warm prompt cache replays identical responses (`temperature=0` + fixed seed → pseudo-determinism). LLM is only called for **opinion leaders**; with `--n-leaders 0` no LLM call happens at all (peripheral cheap policy only — useful offline).

## `run`

Run a single configuration.

| Flag | Default | Meaning |
|------|---------|---------|
| `--platform` | `x` | `x` (interest recommender) or `reddit` (hot-score) |
| `--n-agents` | `200` | number of agents N |
| `--n-leaders` | `20` | top-degree nodes that call the LLM (0 = none) |
| `--timesteps` | `30` | timesteps T (1 tick ≈ 3 minutes) |
| `--activation-rate` | `0.3` | activation subsampling rate ∈ [0,1] |
| `--llm-budget` | `2000` | max LLM calls per run (then falls back to cheap policy) |
| `--ba-m` | `4` | BA edges per new node |
| `--recsys` | platform default | `interest` / `hot-score` / `none` (ablation) |
| `--k-in` / `--k-out` | `5` / `5` | in/out-network feed sizes |
| `--convergence-patience` | `3` | stop after this many consecutive zero-action steps |
| `--seed` | random | core RNG seed (deterministic core) |
| `--temperature` | `0.0` | LLM temperature |
| `--llm-seed` | `0` | LLM backend seed |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache |
| `--output-dir` | `results` | output base directory |

```bash
cargo run --release -- run --platform x --n-agents 200 --n-leaders 20 --timesteps 30 \
    --activation-rate 0.3 --llm-budget 2000 --seed 42

# RecSys ablation (information diffusion should be impaired)
cargo run --release -- run --recsys none --n-agents 200 --seed 42
```

## `sweep`

Sweep agent count × activation rate; aggregate final metrics into `sweep_summary.csv`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--platform` | `x` | platform |
| `--n-agents-values` | `200,1000` | comma-separated agent counts |
| `--activation-rate-min/max/step` | `0.1` / `0.5` / `0.2` | activation rate grid |
| `--n-leaders` | `20` | leaders (capped at N) |
| `--timesteps` | `30` | timesteps |
| `--recsys` | platform default | recommender |
| `--runs` | `3` | independent trials per condition |
| `--seed` | `42` | base seed (each trial is derived independently) |
| `--cache-path` | `.llm_cache/cache.json` | shared cache (raises hit rate) |
| `--output-dir` | `results` | output base directory |

```bash
cargo run --release -- sweep --n-agents-values 200,1000,5000 \
    --activation-rate-min 0.1 --activation-rate-max 0.5 --activation-rate-step 0.2 \
    --runs 5 --seed 42
```

## `reproduce`

Reproduces OASIS's headline emergent phenomena in one shot — **information diffusion** (cascade reach, max cascade size, breadth over the follow graph), **group polarization** (polarization index `P`), and **crowd / herd effects** (down-treat following rate) — contrasted across a **RecSys ablation** (interest / hot-score / none). It runs every recommender condition for `--runs` independent trials, averages the metrics, scores them against the paper's qualitative findings as PASS/off anchors, and writes `reproduce_summary.json` plus per-condition `metrics_<recsys>.csv`. The Python `oasis-tools reproduce` reads these and draws `recsys_diffusion.png`, `polarization_crowd.png`, and `cascade_timeseries.png`.

The deterministic socsim core (BA network, activation, recommender, info propagation, metrics) already runs without an LLM; only the leader action selection is the LLM part. Pass `--mock` to drive that with a deterministic scripted client (a conformist-amplifier caricature: a leader reposts the top recommended post, or posts when its feed is empty), so `reproduce` is fully offline / sandbox-verifiable. The mock is bit-deterministic given a seed.

| Flag | Default | Meaning |
|------|---------|---------|
| `--platform` | `x` | platform (decides recsys default) |
| `--n-agents` | `200` | agent count `N` |
| `--n-leaders` | `30` | opinion leaders (high-degree nodes that call the LLM/mock) |
| `--timesteps` | `24` | timesteps `T` |
| `--activation-rate` | `0.8` | activation subsampling rate |
| `--recsys-values` | `interest,hot-score,none` | recommenders to contrast |
| `--runs` | `3` | independent trials per condition (seed-derived) |
| `--seed` | `42` | base seed |
| `--mock` | off | drive with the deterministic scripted client (no live LLM) |
| `--quick` | off | shrink `N` / `runs` / `T` for a smoke run |
| `--cache-path` | `.llm_cache/cache.json` | shared prompt cache (live only) |
| `--output-dir` | `results` | output base directory |

```bash
# offline one-shot reproduction (no live LLM)
cargo run --release -- reproduce --mock

# lightweight smoke run
cargo run --release -- reproduce --mock --quick

# render the report and figures from the latest reproduce run
uv run oasis-tools reproduce --run --mock
```

The RecSys-ablation anchor uses **max cascade size** rather than reach: with many activating agents, propagation reach (unique authors) saturates regardless of the recommender, whereas the recommender's effect shows in *how far a single post cascades*. Hot-score (global popularity, surfaces the same hottest post to everyone) drives larger cascades than `none` (follow-network latest only), which is the "recommender shapes diffusion" finding.

---
*This file was generated by Claude Code.*

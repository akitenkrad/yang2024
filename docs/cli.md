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
| `--output-dir` | `results` | runvault results root |

Output goes to a runvault run directory. The run directory *is* the output directory, so no timestamped subdirectory and no `latest` symlink are created. Ask `runvault` for the most recent finished run:

```bash
runvault path --experiment oasis --latest --subcommand run
```

```
results/
└── oasis/                                          ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← the last run that finished
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<time>_<cfg8>_<exec4>
    │   ├── run.json                                 ← metadata (git commit / env / LLM / paper)
    │   ├── config.json                              ← envelope; the conditions sit under ["parameters"]
    │   ├── metrics.csv                              ← long form (step / step_unit / scope / name / value)
    │   ├── events.jsonl                             ← one cascade per line (x.yang2024.cascade)
    │   ├── status.json                              ← outcome and duration
    │   └── manifest.csv                             ← hashes of artifacts/ and logs/
    └── figures/                                     ← what the plotting scripts write (outside the run)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── metrics_timeseries.png
```

`metrics.csv` is long form, one value per row. The eight per-step metrics (`polarization_index` / `opinion_std` / `active_user_count` / `propagation_reach` / `cascade_size_max` / `cascade_max_breadth` / `n_posts` / `herd_disagree_rate`) carry a `step` with `step_unit=step`; `converged` (0.0 / 1.0) / `final_step` / `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` describe the whole run with one number each and carry no step. The LLM model / provider / temperature live in the `llm` block of `run.json` (no `llm_meta.json` is written).

The former `cascades.csv` became `x.yang2024.cascade` lines in `events.jsonl`. That table is one row per cascade with no time axis, so it cannot go in `metrics.csv` — every row would claim the same primary key (`name`, `step=∅`, `scope`).

```bash
cargo run --release -- run --platform x --n-agents 200 --n-leaders 20 --timesteps 30 \
    --activation-rate 0.3 --llm-budget 2000 --seed 42

# RecSys ablation (information diffusion should be impaired)
cargo run --release -- run --recsys none --n-agents 200 --seed 42
```

## `sweep`

Sweep agent count × activation rate. A sweep is recorded as one parent run plus one child run per condition. The children take the subcommand name `sweep-point`, sit beside the parent in the experiment directory rather than under it, and point at the parent through `lineage.parent_run_uid`. No per-trial summary CSV is written — the same values are `terminal` lines in the children's `events.jsonl`.

```
results/
└── oasis/
    ├── sweep_20260405_160827_48d033b7_ee20/         ← parent; its parameters are the grid
    │   ├── run.json                                  ← carries lineage.sweep_id; rng.master_seed is null
    │   └── config.json
    ├── sweep-point_20260405_160828_174916dd_955d/   ← child = one condition's trials
    │   ├── config.json                               ← that condition (n_agents / activation_rate)
    │   ├── metrics.csv                               ← the condition's aggregate (n_units / n_converged / mean_final_*)
    │   └── events.jsonl                              ← one trial = one terminal line
    └── ...
```

`runvault path --experiment oasis --latest --subcommand sweep` prints the parent. `oasis-tools visualize-sweep` takes that parent, collects the children's `terminal` lines and rebuilds the familiar one-row-per-trial table.

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
| `--output-dir` | `results` | runvault results root |

```bash
cargo run --release -- sweep --n-agents-values 200,1000,5000 \
    --activation-rate-min 0.1 --activation-rate-max 0.5 --activation-rate-step 0.2 \
    --runs 5 --seed 42
```

## `reproduce`

Reproduces OASIS's headline emergent phenomena in one shot — **information diffusion** (cascade reach, max cascade size, breadth over the follow graph), **group polarization** (polarization index `P`), and **crowd / herd effects** (down-treat following rate) — contrasted across a **RecSys ablation** (interest / hot-score / none). It runs every recommender condition for `--runs` independent trials, averages the metrics, and scores them against the paper's qualitative findings as PASS/off anchors. The three recommender conditions share one run, so the representative run's per-step series and the per-condition trial means are named `<recsys>_<metric>` in `metrics.csv` (e.g. `hot-score_cascade_size_max`, `interest_mean_polarization_index`). The anchor verdicts are categories rather than numbers, so they go to `events.jsonl` as `x.yang2024.anchor` (the observed values themselves are also run-scope metrics). The bands they are checked against are anchors this replication chose rather than values the paper reports, so they do not go into `reference.csv`, which demands a source. The Python `oasis-tools reproduce` reads these and draws `recsys_diffusion.png`, `polarization_crowd.png`, and `cascade_timeseries.png`.

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
| `--output-dir` | `results` | runvault results root |

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

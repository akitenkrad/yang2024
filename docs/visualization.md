**English** | [日本語](visualization.ja.md)

# Visualization

The Python tools live in `tools/` as the `oasis-tools` package (module `oasis_tools`). Install with `uv sync` at the workspace root, then run `uv run oasis-tools <subcommand>`. How a run directory is read lives in the `runvault` package (`runvault.read`) — rather than scanning `results/` and guessing at the newest directory, the tools ask `runvault path`.

Omitting the directory argument resolves the run through `runvault path --latest`, so `runvault` has to be on PATH or the `RUNVAULT` environment variable has to point at the binary. Figures are written outside the run directory (`results/oasis/figures/<run_slug>/`) — `manifest.csv` is settled when the run ends, so a figure added to `artifacts/` afterwards would carry no hash.

## `visualize`

Reads the run directory's `metrics.csv` (pivoted internally) and the cascade lines of `events.jsonl` (`x.yang2024.cascade`), and writes:

- `metrics_timeseries.png` — four panels: **polarization index P** (group polarization, Finding 2), **active-user count** (Time-Engine check), **propagation reach** (information diffusion via the recommender), and **cascade size / breadth** (Finding 1).
- `cascade_tree.png` — the largest cascades drawn as root→repost star trees (networkx). Suppress with `--no-graph`.

```bash
uv run oasis-tools visualize
uv run oasis-tools visualize --results_dir "$(runvault path --experiment oasis --latest --subcommand run)" --output_dir out
```

## `visualize-sweep`

Rebuilds the one-row-per-trial table from the sweep parent's children (`oasis_tools.sweep_summary`) and writes heatmaps and line plots over the agent-count × activation-rate grid (the scale-effect view):

- `sweep_polarization_heatmap.png` — final polarization P.
- `sweep_reach_heatmap.png` — final propagation reach.
- `sweep_metrics_vs_n.png` — P / opinion diversity / reach vs N, one line per activation rate.

```bash
uv run oasis-tools visualize-sweep
uv run oasis-tools visualize-sweep --sweep_dir "$(runvault path --experiment oasis --latest --subcommand sweep)"
```

## `show-experiment-settings`

Prints the `parameters` of `config.json` (a run and a sweep parent are told apart by the presence of `n_agents_values`), the `llm` block of `run.json`, and the LLM call breakdown from `metrics.csv`. A pre-migration flat `config.json` / `sweep_config.json` / `llm_meta.json` is still read. `--json` emits machine-readable JSON.

```bash
uv run oasis-tools show-experiment-settings
```

## `reproduce`

Reads the run directory `oasis reproduce` wrote — the RecSys-ablation matrix from the run-scope metrics of `metrics.csv` (`<recsys>_mean_*`), the PASS/off verdicts from `events.jsonl` (`x.yang2024.anchor`) — prints the matrix and the anchor table, and draws three figures into `results/oasis/figures/<run_slug>/`:

- `recsys_diffusion.png` — final propagation reach, max cascade size, and breadth per recommender (information diffusion).
- `polarization_crowd.png` — final polarization index `P`, polarization gain, and herd-following rate per recommender.
- `cascade_timeseries.png` — the representative run's max-cascade-size and reach over time, per recommender.

`--run` first invokes the Rust binary; add `--mock` (and optionally `--quick`) to stay offline. `--json` dumps the summary.

```bash
uv run oasis-tools reproduce --run --mock          # reproduce + report + figures, offline
uv run oasis-tools reproduce                        # visualize the most recent reproduce run
```

## Interpreting the outputs

Because the local model differs from the paper's, read the figures **qualitatively**: a cascade that spreads over multiple stages (not a single broadcast), a polarization index that trends upward, propagation reach that drops under `--recsys none`, and growth of polarization / diversity with larger N.

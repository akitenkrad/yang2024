**English** | [日本語](architecture.ja.md)

# Architecture

## Repository structure

```
yang2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (yang2024-workspace)
├── simulation/                 # Rust crate oasis-simulation (bin oasis)
│   ├── src/
│   │   ├── main.rs             # CLI (clap: run / sweep / reproduce)
│   │   ├── config.rs           # Config, Platform, RecSysKind, LlmSettings
│   │   ├── world.rs            # OasisWorld (WorldState), AgentProfile, Post, embed()
│   │   ├── recsys.rs           # deterministic recommender (interest / hot-score / none)
│   │   ├── prompts.rs          # CoT action prompt
│   │   ├── parse.rs            # ACTION/TARGET/CONTENT parsing
│   │   ├── mechanisms.rs       # the 6 mechanisms (one per phase)
│   │   ├── llm.rs              # Ollama→OpenAI fallback + cache (socsim-llm)
│   │   ├── reproduce_mock.rs   # offline scripted client for reproduce/run --mock
│   │   ├── simulation.rs       # init_world + run / run_mock driver + output writers
│   │   └── metrics.rs          # polarization, cascade, reach, herd
│   ├── examples/mock_smoke.rs  # offline (no live LLM) smoke
│   └── tests/integration_test.rs
├── tools/src/oasis_tools/      # Python package oasis-tools
└── docs/
```

## The dynamic follow graph

Agents are **nodes on a dynamic social graph**, not spatial agents, so the spatial primitive (`socsim-grid`) is unused. The graph is `socsim_net::SocialNetwork` initialised with `SocialNetwork::barabasi_albert(ids, m, rng)` (the paper's large-scale users follow a BA scale-free degree distribution). A `follow` action adds a new edge at runtime, so the graph is dynamic. A post by `B` is a feed candidate for `B`'s neighbours (`neighbors(B)`).

## Two-layer determinism

- **Deterministic socsim core** — BA network generation, Time-Engine activation (24-dim hourly activity × `--activation-rate`), the recommender, info propagation and metrics. Two RNG streams are derived from the root seed: `derive_seed(root, &[0])` for world init (network, profiles, activity probabilities, initial opinions) and `derive_seed(root, &[1])` for the engine (`RandomActivationScheduler` + the activation subsampling draw). Bit-reproducible given a seed.
- **Non-deterministic LLM layer** — confined to `AgentActionMechanism` (the `Decision` phase). `CachingClient<Box<dyn LlmClient>>` with a production `FallbackClient<OllamaClient, OpenAiClient>` (Ollama first → OpenAI fallback). Pseudo-determinised by the prompt→response cache, `temperature=0` and a fixed seed. Tests inject `socsim_llm::mock::ScriptedClient`.

## The six mechanisms (one per phase)

| Mechanism | Phase | Role |
|-----------|-------|------|
| `ActivationMechanism` | PreStep | Time Engine: pick the active set via 24-dim activity × activation rate; copy the leader set into scratch |
| `FeedRecommendationMechanism` | Environment | RecSys: build each active agent's feed (deterministic) |
| `AgentActionMechanism` | Decision | **LLM only here**: leaders call the LLM (CoT); peripheral agents use a cheap stochastic policy; respects `--llm-budget` |
| `InfoPropagationMechanism` | Interaction | apply actions to the post store + social graph (new posts, reposts, likes, follow edges); opinions drift on repost/like |
| `MetricsMechanism` | Reward | record per-step metrics to the recorder |
| `PostStepMechanism` | PostStep | update agent memory; request_stop after `--convergence-patience` consecutive zero-action steps |

## Metrics

- `polarization_index` — `P = (1/N)·Σ(o_a − ō)²` (group polarization; should rise over time).
- `opinion_std` — opinion diversity proxy.
- `active_user_count` — agents that acted this step (Time-Engine check).
- `propagation_reach` — unique nodes reached (RecSys ablation: drops sharply with `--recsys none`).
- `cascade_size_max` / `cascade_max_breadth` — information-diffusion cascade size and breadth from the post store's `root` links.
- `herd_disagree_rate` — herd-following proxy.

## Output

[runvault](https://github.com/akitenkrad/rs-runvault) owns where output goes and how it is named. One subcommand invocation is one run, and the run directory *is* the output directory, so there is no timestamped subdirectory and no `latest` symlink.

`results/oasis/<subcommand>_<time>_<cfg8>_<exec4>/` holds `run.json` (git commit / env / LLM / paper), `config.json` (an envelope; the conditions sit under `parameters`), `metrics.csv` (long form `run_uid, step, step_unit, scope, name, value`), `events.jsonl`, `status.json` and `manifest.csv`. The cascade table is `x.yang2024.cascade` lines in `events.jsonl` (one row per cascade with no time axis, so it cannot go in `metrics.csv`), and the LLM model / provider / temperature live in the `llm` block of `run.json`. A sweep is a parent run plus one child (`sweep-point`) per condition, each trial's final values being a `terminal` event in the child. See the [CLI reference](cli.md).

## References

- Yang, Z., Zhang, Z., Zheng, Z., et al. (2024). *OASIS: Open Agent Social Interaction Simulations with One Million Agents.* arXiv:2411.11581.
- Barabási, A.-L., & Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science.
- [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) (`socsim-core` / `socsim-engine` / `socsim-net` / `socsim-llm`).

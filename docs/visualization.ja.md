[English](visualization.md) | **日本語**

# 可視化

Python ツールは `tools/` の `oasis-tools` パッケージ (モジュール `oasis_tools`)．workspace ルートで `uv sync` 後，`uv run oasis-tools <サブコマンド>` で実行する．run ディレクトリの読み方は `runvault` パッケージ (`runvault.read`) に預けてある — `results/` を走査して新しそうなディレクトリを当てにいくのではなく，`runvault path` に聞く．

ディレクトリ引数を省略すると `runvault path --latest` が解決するので，`runvault` が PATH にあるか，環境変数 `RUNVAULT` がバイナリを指している必要がある．図は run ディレクトリの外 (`results/oasis/figures/<run_slug>/`) に出る — `manifest.csv` は run が終わった時点で確定するので，後から `artifacts/` に足した図にはハッシュが付かない．

## `visualize`

run ディレクトリの `metrics.csv` (内部で wide にピボット) と `events.jsonl` のカスケード行 (`x.yang2024.cascade`) を読み，以下を出力する:

- `metrics_timeseries.png` — 4 パネル: **極化指数 P** (グループ極化, Finding 2)，**active-user 数** (Time-Engine 検証)，**伝播到達数** (推薦器を介した情報拡散)，**カスケード規模 / 幅** (Finding 1)．
- `cascade_tree.png` — 規模上位カスケードを root→repost の星形ツリーで描画 (networkx)．`--no-graph` で抑止．

```bash
uv run oasis-tools visualize
uv run oasis-tools visualize --results_dir "$(runvault path --experiment oasis --latest --subcommand run)" --output_dir out
```

## `visualize-sweep`

sweep 親の子 run から «1 行 1 試行» の表を組み直し (`oasis_tools.sweep_summary`)，エージェント数 × 活性化率のグリッドについてヒートマップと折れ線 (スケール効果ビュー) を出力する:

- `sweep_polarization_heatmap.png` — 最終極化指数 P．
- `sweep_reach_heatmap.png` — 最終伝播到達数．
- `sweep_metrics_vs_n.png` — P / 意見多様性 / 到達数 vs N (活性化率ごとに 1 本の折れ線)．

```bash
uv run oasis-tools visualize-sweep
uv run oasis-tools visualize-sweep --sweep_dir "$(runvault path --experiment oasis --latest --subcommand sweep)"
```

## `show-experiment-settings`

`config.json` の `parameters` (run か sweep 親かは `n_agents_values` の有無で判別する) と，`run.json` の `llm` ブロック + `metrics.csv` の LLM 呼び出し内訳を整形表示する．移行前の flat な `config.json` / `sweep_config.json` / `llm_meta.json` も読める．`--json` で機械可読 JSON を出力する．

```bash
uv run oasis-tools show-experiment-settings
```

## `reproduce`

`oasis reproduce` の run ディレクトリを読み — RecSys アブレーション行列は `metrics.csv` の run スコープ指標 (`<推薦器>_mean_*`)，PASS/off の判定は `events.jsonl` の `x.yang2024.anchor` — 行列とアンカー表を表示し，`results/oasis/figures/<run_slug>/` に 3 つの図を描く:

- `recsys_diffusion.png` — 推薦器別の最終 伝播到達・最大カスケード規模・幅 (情報拡散)．
- `polarization_crowd.png` — 推薦器別の最終 極化指数 `P`・極化増分・群衆追随率．
- `cascade_timeseries.png` — 代表 run の最大カスケード規模・伝播到達の時系列 (推薦器別)．

`--run` を付けると先に Rust バイナリを呼ぶ．`--mock` (および任意で `--quick`) を付ければオフラインで完結する．`--json` でサマリを出力する．

```bash
uv run oasis-tools reproduce --run --mock          # 再現 + レポート + 図 (オフライン)
uv run oasis-tools reproduce                        # 既存の最新 reproduce run を可視化
```

## 出力の読み方

ローカルモデルは論文と異なるため，図は**定性的**に読む: 単一ブロードキャストでなく多段に広がるカスケード，増大傾向の極化指数，`--recsys none` で低下する伝播到達数，N 増に伴う極化 / 多様性の増大．

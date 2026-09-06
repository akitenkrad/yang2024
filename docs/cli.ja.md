[English](cli.md) | **日本語**

# CLI

Rust バイナリは `oasis`．`cargo build --release` 後，`cargo run --release -- <サブコマンド> [フラグ]`．

## LLM 環境変数

LLM レイヤは **Ollama 第一 → OpenAI フォールバック** (`socsim-llm`)．環境変数で設定し，ソースにハードコードしない:

| 変数 | 既定 | 意味 |
|------|------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama エンドポイント |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama モデル |
| `OPENAI_API_KEY` | (未設定) | OpenAI フォールバックを有効化 |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI モデル |

ウォームキャッシュは同一応答を再生する (`temperature=0` + 固定 seed → 擬似決定論)．LLM が呼ばれるのは**オピニオンリーダー**のみで，`--n-leaders 0` では LLM は一切呼ばれない (周辺の簡易ポリシーのみ; オフラインに有用)．

## `run`

単一設定で実行する．

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--platform` | `x` | `x` (興味推薦) または `reddit` (ホットスコア) |
| `--n-agents` | `200` | エージェント数 N |
| `--n-leaders` | `20` | LLM を呼ぶ高次数ノード (0 = なし) |
| `--timesteps` | `30` | タイムステップ T (1 tick ≈ 3 分) |
| `--activation-rate` | `0.3` | 活性化サブサンプリング率 ∈ [0,1] |
| `--llm-budget` | `2000` | 1 実行あたり最大 LLM 呼び出し数 (超過で簡易ポリシー) |
| `--ba-m` | `4` | BA の新規ノードあたり結合数 |
| `--recsys` | プラットフォーム既定 | `interest` / `hot-score` / `none` (アブレーション) |
| `--k-in` / `--k-out` | `5` / `5` | in/out-network フィード件数 |
| `--convergence-patience` | `3` | 連続ゼロアクションがこの数に達したら停止 |
| `--seed` | ランダム | コア RNG seed (決定論的コア) |
| `--temperature` | `0.0` | LLM 温度 |
| `--llm-seed` | `0` | LLM バックエンド seed |
| `--cache-path` | `.llm_cache/cache.json` | プロンプト→応答キャッシュ |
| `--output-dir` | `results` | runvault の results ルート |

出力は runvault の run ディレクトリへ．run ディレクトリが出力先そのものなので，タイムスタンプ付きサブディレクトリも `latest` symlink も作らない．直近の完了 run のパスは `runvault` に聞く:

```bash
runvault path --experiment oasis --latest --subcommand run
```

```
results/
└── oasis/                                          ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← 最後に完了した run
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<時刻>_<cfg8>_<exec4>
    │   ├── run.json                                 ← メタデータ (git commit / 環境 / LLM / 論文情報)
    │   ├── config.json                              ← 封筒．実験条件は ["parameters"] の下
    │   ├── metrics.csv                              ← long 形式 (step / step_unit / scope / name / value)
    │   ├── events.jsonl                             ← カスケード 1 本 = 1 行 (x.yang2024.cascade)
    │   ├── status.json                              ← 終了状態と所要時間
    │   └── manifest.csv                             ← artifacts/ と logs/ のハッシュ
    └── figures/                                     ← 可視化スクリプトの出力 (run の外)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── metrics_timeseries.png
```

`metrics.csv` は 1 行 1 値の long 形式．ステップごとの 8 指標 (`polarization_index` / `opinion_std` / `active_user_count` / `propagation_reach` / `cascade_size_max` / `cascade_max_breadth` / `n_posts` / `herd_disagree_rate`) は `step_unit=step` の `step` を持ち，run 全体を 1 つの値で表す `converged` (0.0 / 1.0) / `final_step` / `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` は `scope=run` で `step` を持たない．LLM のモデル・provider・温度は `run.json` の `llm` ブロックにある (`llm_meta.json` は書かれない)．

旧 `cascades.csv` は `events.jsonl` の `x.yang2024.cascade` 行になった．カスケード表は時間軸を持たない «1 本 1 行» なので `metrics.csv` には置けない — 全行が同じ主キー (`name`, `step=∅`, `scope`) を名乗ってしまう．

```bash
cargo run --release -- run --platform x --n-agents 200 --n-leaders 20 --timesteps 30 \
    --activation-rate 0.3 --llm-budget 2000 --seed 42

# RecSys アブレーション (情報拡散が阻害されるはず)
cargo run --release -- run --recsys none --n-agents 200 --seed 42
```

## `sweep`

エージェント数 × 活性化率を走査する．親 run 1 本と，条件 1 点ごとの子 run に分けて記録される．子はサブコマンド名 `sweep-point` を名乗り，親の下ではなく experiment ディレクトリの兄弟として並び，`lineage.parent_run_uid` で親を指す．1 行 1 試行のサマリ CSV は書かない (同じ値は子の `events.jsonl` の `terminal` 行にある)．

```
results/
└── oasis/
    ├── sweep_20260405_160827_48d033b7_ee20/         ← 親．parameters が格子の定義
    │   ├── run.json                                  ← lineage.sweep_id を持つ．rng.master_seed は null
    │   └── config.json
    ├── sweep-point_20260405_160828_174916dd_955d/   ← 子 = 1 条件の試行群
    │   ├── config.json                               ← その条件 (n_agents / activation_rate)
    │   ├── metrics.csv                               ← 条件の集約 (n_units / n_converged / mean_final_*)
    │   └── events.jsonl                              ← 試行 1 本 = terminal 行 1 本
    └── ...
```

親のパスは `runvault path --experiment oasis --latest --subcommand sweep` で取れる．`oasis-tools visualize-sweep` はこの親を受け取り，子 run の `terminal` 行を集めて従来のサマリ表 (1 行 1 試行) を組み直す．

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--platform` | `x` | プラットフォーム |
| `--n-agents-values` | `200,1000` | カンマ区切りのエージェント数 |
| `--activation-rate-min/max/step` | `0.1` / `0.5` / `0.2` | 活性化率グリッド |
| `--n-leaders` | `20` | リーダー数 (N で上限) |
| `--timesteps` | `30` | タイムステップ |
| `--recsys` | プラットフォーム既定 | 推薦器 |
| `--runs` | `3` | 各条件の独立試行数 |
| `--seed` | `42` | 基点 seed (各試行は独立に derive) |
| `--cache-path` | `.llm_cache/cache.json` | 共有キャッシュ (ヒット率向上) |
| `--output-dir` | `results` | runvault の results ルート |

```bash
cargo run --release -- sweep --n-agents-values 200,1000,5000 \
    --activation-rate-min 0.1 --activation-rate-max 0.5 --activation-rate-step 0.2 \
    --runs 5 --seed 42
```

## `reproduce`

OASIS の見出し的な創発現象を一括再現する — **情報拡散** (フォローグラフ上のカスケード到達数・最大カスケード規模・幅)，**グループ極化** (極化指数 `P`)，**群衆 / 群れ効果** (down-treat 群追随率) — を **RecSys アブレーション** (interest / hot-score / none) で対比する．各推薦器条件を `--runs` 回独立試行して平均し，論文の定性的知見と突き合わせて PASS/off アンカーとして採点する．3 つの推薦器条件が 1 本の run に同居するので，条件ごとの代表 run のステップ別系列と試行平均は `<推薦器>_<指標名>` (例 `hot-score_cascade_size_max` / `interest_mean_polarization_index`) という名前で `metrics.csv` に入る．アンカーの判定は数ではなくカテゴリなので `events.jsonl` の `x.yang2024.anchor` へ書く (観測値そのものは run スコープ指標にもある)．照合先の帯は論文が報告した数値ではなくこの再現実装が置いたアンカーなので，出典を要求する `reference.csv` には書かない．Python の `oasis-tools reproduce` がこれらを読み，`recsys_diffusion.png`・`polarization_crowd.png`・`cascade_timeseries.png` を描く．

決定論的 socsim コア (BA 網・活性化・推薦器・情報伝播・指標) は LLM 無しで動く．LLM の部分は leader の行動選択のみである．`--mock` を付けると，その部分を決定論的 scripted クライアント («同調的増幅器» の戯画: leader は推薦フィード先頭をリポストし，フィードが空なら新規投稿する) で駆動するため，`reproduce` は完全にオフライン / サンドボックスで検証できる．mock は seed を固定すれば bit 決定論的である．

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--platform` | `x` | プラットフォーム (recsys 既定を決める) |
| `--n-agents` | `200` | エージェント数 `N` |
| `--n-leaders` | `30` | オピニオンリーダー (LLM/mock を呼ぶ高次数ノード) |
| `--timesteps` | `24` | タイムステップ `T` |
| `--activation-rate` | `0.8` | 活性化サブサンプリング率 |
| `--recsys-values` | `interest,hot-score,none` | 対比する推薦器 |
| `--runs` | `3` | 各条件の独立試行数 (seed 派生) |
| `--seed` | `42` | 基点 seed |
| `--mock` | off | 決定論的 scripted クライアントで駆動する (ライブ LLM 不要) |
| `--quick` | off | `N` / `runs` / `T` を縮小したスモーク |
| `--cache-path` | `.llm_cache/cache.json` | 共有プロンプトキャッシュ (live のみ) |
| `--output-dir` | `results` | runvault の results ルート |

```bash
# オフライン一括再現 (ライブ LLM 不要)
cargo run --release -- reproduce --mock

# 軽量スモーク
cargo run --release -- reproduce --mock --quick

# 最新の reproduce 結果からレポートと図を生成する
uv run oasis-tools reproduce --run --mock
```

RecSys アブレーションのアンカーは到達数ではなく **最大カスケード規模** を用いる: 活性化エージェントが多いと伝播到達 (ユニーク著者数) は推薦器に依らず飽和するが，推薦器の効果は «1 投稿がどこまでカスケードするか» に現れる．hot-score (グローバル人気で全員に最ホット投稿を見せる) は `none` (フォロー先の最新のみ) より大きなカスケードを生み，これが «推薦器が拡散を形作る» 知見である．

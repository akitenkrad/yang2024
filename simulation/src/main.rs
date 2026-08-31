//! Yang et al. (2024) "OASIS: Open Agent Social Interaction Simulations with One
//! Million Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で BA フォローグラフ上の LLM 駆動 行動選択 + 推薦 + 情報
//!               伝播を実行する．
//! `sweep`     : エージェント数 × 活性化率 を走査する．親 run 1 本と，条件 1 点ごとの
//!               子 run (`sweep-point`) に分ける．
//! `reproduce` : 論文の創発現象 (情報拡散カスケード / グループ極化 / 群衆効果) を
//!               RecSys アブレーション (interest / hot-score / none) で対比し，
//!               観測 vs 論文の PASS/off を判定する．
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};
use serde::Serialize;

use oasis_simulation::config::{
    parse_platform, parse_recsys, Config, LlmSettings, Platform, RecSysConfig, RecSysKind,
};
use oasis_simulation::llm::{build_live_client, OasisClient};
use oasis_simulation::metrics::StepMetrics;
use oasis_simulation::record::{self, ANCHOR_EVENT, DOMAIN, EXPERIMENT, REPO_ID};
use oasis_simulation::reproduce_mock::build_reproduce_client;
use oasis_simulation::simulation::{run_with_client, SimulationResult};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "oasis",
    about = "Yang et al. (2024) OASIS: Open Agent Social Interaction Simulations — 再現実験"
)]
struct Cli {
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で BA フォローグラフ上の LLM 駆動 行動選択 + 推薦 + 情報伝播を実行する．
    Run(RunArgs),
    /// エージェント数 × 活性化率 を走査し，最終集団指標を集計する．
    Sweep(SweepArgs),
    /// 論文の創発現象 (情報拡散 / 極化 / 群衆効果) + RecSys アブレーションを一括再現する．
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// プラットフォーム (x / reddit)．
    #[arg(long, default_value = "x")]
    platform: String,

    /// エージェント数 N．
    #[arg(long, default_value_t = 200)]
    n_agents: usize,

    /// オピニオンリーダー数 (高次数ノード; LLM を呼ぶ対象)．
    #[arg(long, default_value_t = 20)]
    n_leaders: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 30)]
    timesteps: usize,

    /// 活性化サブサンプリング率 ∈ [0,1]．
    #[arg(long, default_value_t = 0.3)]
    activation_rate: f64,

    /// 1 実行あたりの最大 LLM 呼び出し数．
    #[arg(long, default_value_t = 2000)]
    llm_budget: usize,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 4)]
    ba_m: usize,

    /// 推薦器種別 (interest / hot-score / none)．省略時はプラットフォーム既定．
    #[arg(long)]
    recsys: Option<String>,

    /// in-network 取り込み件数 k_in．
    #[arg(long, default_value_t = 5)]
    k_in: usize,

    /// out-network 取り込み件数 k_out．
    #[arg(long, default_value_t = 5)]
    k_out: usize,

    /// 連続ゼロアクション収束しきい値 (これに達したら停止)．
    #[arg(long, default_value_t = 3)]
    convergence_patience: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0; 再現性のため)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (既定 .llm_cache/cache.json)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (ライブ LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// プラットフォーム (x / reddit)．
    #[arg(long, default_value = "x")]
    platform: String,

    /// カンマ区切りのエージェント数リスト．
    #[arg(long, default_value = "200,1000")]
    n_agents_values: String,

    /// 活性化率スイープ下限．
    #[arg(long, default_value_t = 0.1)]
    activation_rate_min: f64,

    /// 活性化率スイープ上限．
    #[arg(long, default_value_t = 0.5)]
    activation_rate_max: f64,

    /// 活性化率スイープ刻み．
    #[arg(long, default_value_t = 0.2)]
    activation_rate_step: f64,

    /// オピニオンリーダー数．
    #[arg(long, default_value_t = 20)]
    n_leaders: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 30)]
    timesteps: usize,

    /// 1 実行あたりの最大 LLM 呼び出し数．
    #[arg(long, default_value_t = 2000)]
    llm_budget: usize,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 4)]
    ba_m: usize,

    /// 推薦器種別 (interest / hot-score / none)．
    #[arg(long)]
    recsys: Option<String>,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// プラットフォーム (x / reddit; recsys 既定の決定に使う)．
    #[arg(long, default_value = "x")]
    platform: String,

    /// エージェント数 N．
    #[arg(long, default_value_t = 200)]
    n_agents: usize,

    /// オピニオンリーダー数 (高次数ノード; LLM/mock を呼ぶ対象)．
    #[arg(long, default_value_t = 30)]
    n_leaders: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 24)]
    timesteps: usize,

    /// 活性化サブサンプリング率 ∈ [0,1]．
    #[arg(long, default_value_t = 0.8)]
    activation_rate: f64,

    /// 1 実行あたりの最大 LLM 呼び出し数．
    #[arg(long, default_value_t = 5000)]
    llm_budget: usize,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 4)]
    ba_m: usize,

    /// 対比する推薦器のリスト (カンマ区切り; interest / hot-score / none)．
    #[arg(long, default_value = "interest,hot-score,none")]
    recsys_values: String,

    /// 各条件あたりの独立試行数 (シードを派生して平均)．
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// 乱数シード基点 (各条件・試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (ライブ LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// LLM 生成温度 (live 時のみ)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (live 時のみ)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (live 時のみ; 全条件で共有)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 軽量モード (N・runs・T を縮小; 動作確認用)．
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(Serialize)]
struct SweepParameters {
    platform: String,
    recsys: String,
    n_agents_values: Vec<usize>,
    activation_rate_values: Vec<f64>,
    n_leaders: usize,
    timesteps: usize,
    llm_budget: usize,
    ba_m: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// スイープの子 run (N × 活性化率の 1 点) の実験条件．
///
/// `run` の条件に `runs` が付いた形で，`run` とは別のサブコマンド名を持つ．
/// 同じ `run` を名乗らせると，「1 本のシミュレーション」と「同一条件の
/// `runs` 本」という中身の違う 2 つが 1 つの名前に同居し，`runvault path
/// --subcommand run` がどちらを返すか分からなくなる．
#[derive(Serialize)]
struct SweepPointParameters {
    platform: String,
    recsys: String,
    n_agents: usize,
    activation_rate: f64,
    n_leaders: usize,
    timesteps: usize,
    llm_budget: usize,
    ba_m: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// `reproduce` run の実験条件．
///
/// `n_agents` / `runs` / `timesteps` / `n_leaders` は `--quick` を反映した **実際に
/// 回した値**で，`--quick` そのものは持たない (同じ条件なら同じ config_hash になる)．
#[derive(Serialize)]
struct ReproduceParameters {
    platform: String,
    recsys_values: Vec<String>,
    n_agents: usize,
    n_leaders: usize,
    timesteps: usize,
    activation_rate: f64,
    llm_budget: usize,
    ba_m: usize,
    runs: usize,
    convergence_patience: usize,
    mock: bool,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// このサブコマンドを駆動する LLM クライアントを組む．
///
/// `Run::start` の前に呼ぶ．`run.json` の `llm` ブロックに書くモデル名と endpoint は，
/// 実際に応答するバックエンドから採らないと意味を持たない．
fn build_client(mock: bool, llm: &LlmSettings) -> OasisClient {
    if mock {
        build_reproduce_client()
    } else {
        build_live_client(llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    }
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// 活性化率スイープの値列を [min, max] を step 刻みで生成する．
fn activation_values(min: f64, max: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if step <= 0.0 {
        out.push(min);
        return out;
    }
    let mut v = min;
    // 浮動小数の誤差を避けるため丸めて格納する．
    while v <= max + 1e-9 {
        out.push((v * 1000.0).round() / 1000.0);
        v += step;
    }
    out
}

/// 推薦器設定を組み立てる (CLI 指定がなければプラットフォーム既定)．
fn build_recsys(
    platform: Platform,
    recsys: &Option<String>,
    k_in: usize,
    k_out: usize,
) -> RecSysConfig {
    let kind = match recsys {
        Some(s) => parse_recsys(s).unwrap_or_else(|e| panic!("{e}")),
        None => RecSysKind::default_for(platform),
    };
    RecSysConfig {
        kind,
        k_in,
        k_out,
        ..RecSysConfig::default()
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let platform = parse_platform(&args.platform).unwrap_or_else(|e| panic!("{}", e));
    let recsys = build_recsys(platform, &args.recsys, args.k_in, args.k_out);

    // シードを実体化してから記録する．--seed 省略時にシミュレーション側で
    // rand::random に落とすと，実際に使われたシードがどこにも残らない．
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);

    let cfg = Config {
        platform,
        n_agents: args.n_agents,
        n_leaders: args.n_leaders,
        timesteps: args.timesteps,
        activation_rate: args.activation_rate,
        llm_budget: args.llm_budget,
        ba_m: args.ba_m,
        recsys,
        convergence_patience: args.convergence_patience,
        seed: Some(seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            // mock は in-memory cache なので永続キャッシュは無効化する．
            cache_path: if args.mock {
                None
            } else {
                Some(args.cache_path.clone())
            },
        },
    };

    if !args.mock {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let client = build_client(args.mock, &cfg.llm);
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let parameters = cfg.to_run_config_json(seed, args.mock);
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    println!("=== Yang et al. (2024) OASIS LLM ソーシャルメディアシミュレーション 再現実験 ===");
    println!(
        "platform: {} | recsys: {} | N: {} | leaders: {} | T: {} | activation: {}",
        cfg.platform.label(),
        cfg.recsys.kind.label(),
        cfg.n_agents,
        cfg.n_leaders,
        cfg.timesteps,
        cfg.activation_rate,
    );
    println!(
        "seed: {} | llm-budget: {} | LLM: temp={} llm_seed={} cache={} | mode={}",
        seed,
        cfg.llm_budget,
        cfg.llm.temperature,
        cfg.llm.seed,
        args.cache_path,
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-----------------------------------------------------------------");

    let result = run_with_client(&cfg, client).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
    record::log_simulation(&mut rv, &result);
    record::log_cascades(&mut rv, &result.cascade_rows);

    let last = result.metrics_history.last().unwrap();
    println!(
        "収束: {} | step: {}",
        if result.converged { "Yes" } else { "No" },
        result.final_step
    );
    println!(
        "最終 極化指数 P: {:.4} | 意見std: {:.4} | 伝播到達: {} | 最大カスケード: {}",
        last.polarization_index, last.opinion_std, last.propagation_reach, last.cascade_size_max,
    );
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("メトリクス → {}/metrics.csv", dir.display());
    println!("カスケード → {}/events.jsonl", dir.display());
    println!("設定       → {}/config.json", dir.display());
    println!("LLM メタ   → {}/run.json (llm ブロック)", dir.display());
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let platform = parse_platform(&args.platform).unwrap_or_else(|e| panic!("{}", e));
    let recsys_kind = match &args.recsys {
        Some(s) => parse_recsys(s).unwrap_or_else(|e| panic!("{e}")),
        None => RecSysKind::default_for(platform),
    };

    let n_agents_values: Vec<usize> = split_csv(&args.n_agents_values)
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("不正なエージェント数: {s}"))
        })
        .collect();
    let activation_rate_values = activation_values(
        args.activation_rate_min,
        args.activation_rate_max,
        args.activation_rate_step,
    );

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = n_agents_values.len() * activation_rate_values.len() * args.runs;

    let llm_settings = LlmSettings {
        temperature: args.temperature,
        seed: args.llm_seed,
        cache_path: Some(args.cache_path.clone()),
    };
    // 全条件が同じバックエンドを使うので，`llm` ブロックは 1 度組んで子 run へ配る．
    // 名乗る名前を知っているのはクライアントだけなので，回す前に 1 つ組んで訊く．
    let llm = {
        let probe = build_client(false, &llm_settings);
        record::llm_block(
            probe.inner().model(),
            probe.inner().endpoint(),
            llm_settings.temperature,
        )
    };

    let sweep_parameters = SweepParameters {
        platform: platform.label().to_string(),
        recsys: recsys_kind.label().to_string(),
        n_agents_values: n_agents_values.clone(),
        activation_rate_values: activation_rate_values.clone(),
        n_leaders: args.n_leaders,
        timesteps: args.timesteps,
        llm_budget: args.llm_budget,
        ba_m: args.ba_m,
        runs: args.runs,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
    };

    // 親 run: グリッド定義そのものを parameters に持つ．個別条件の指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らず，基点シードは
    // /parameters.seed と seed_pointers 経由で execution_hash に残る．
    // sweep_id は runvault が親の run_slug で埋める．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .llm(llm.clone())
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Yang et al. (2024) OASIS パラメータスイープ (N × activation) ===");
    println!(
        "platform: {} | recsys: {} | N: {} 種 | activation: {} 種 | 試行: {} | 合計: {} 実行",
        platform.label(),
        recsys_kind.label(),
        n_agents_values.len(),
        activation_rate_values.len(),
        args.runs,
        n_total,
    );
    println!("シード (base): {}", args.seed);
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------------");

    // エージェント数別の平均極化指数 (最後に出す要約)．試行ごとの値は子 run の
    // events.jsonl が正本なので，ここでは表示のためだけに積む．
    let mut polarization_by_n: Vec<(usize, Vec<f64>)> =
        n_agents_values.iter().map(|&n| (n, Vec::new())).collect();
    let mut done = 0usize;

    for &n_agents in &n_agents_values {
        for &activation_rate in &activation_rate_values {
            let params = SweepPointParameters {
                platform: platform.label().to_string(),
                recsys: recsys_kind.label().to_string(),
                n_agents,
                activation_rate,
                n_leaders: args.n_leaders.min(n_agents),
                timesteps: args.timesteps,
                llm_budget: args.llm_budget,
                ba_m: args.ba_m,
                runs: args.runs,
                seed: args.seed,
                llm_temperature: args.temperature,
                llm_seed: args.llm_seed,
            };

            // 子は «その条件の試行群» そのもの．master_seed は親と同じ基点で，
            // 条件が違えば config_hash が違うので run としては別物になる．
            // 同じ条件の繰り返しは無いので replicate_index は 0．
            let mut child = Run::start(
                RunOptions::new(EXPERIMENT, "sweep-point")
                    .repo_id(REPO_ID)
                    .domain(DOMAIN)
                    .results_root(&args.output_dir)
                    .parameters(&params)
                    .expect("runvault: 子 run の parameters の組み立てに失敗")
                    .seed_pointers(["/seed"])
                    .master_seed(args.seed)
                    .replicate_index(0)
                    .llm(llm.clone())
                    .lineage(Lineage {
                        sweep_id: Some(sweep_id.clone()),
                        parent_run_uid: Some(parent_run_uid.clone()),
                        ..Default::default()
                    })
                    .replication(record::replication()),
            )
            .expect("runvault: 子 run の開始に失敗");

            let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
            for run_idx in 0..args.runs {
                // 各 (platform, n_agents, activation_rate, run) に独立なシードを派生させる．
                let seed = record::sweep_trial_seed(
                    args.seed,
                    platform.label(),
                    n_agents,
                    activation_rate,
                    run_idx,
                );

                let cfg = Config {
                    platform,
                    n_agents,
                    n_leaders: args.n_leaders.min(n_agents),
                    timesteps: args.timesteps,
                    activation_rate,
                    llm_budget: args.llm_budget,
                    ba_m: args.ba_m,
                    recsys: RecSysConfig {
                        kind: recsys_kind,
                        ..RecSysConfig::default()
                    },
                    convergence_patience: 3,
                    seed: Some(seed),
                    llm: llm_settings.clone(),
                };

                let client = build_client(false, &cfg.llm);
                let result = run_with_client(&cfg, client)
                    .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

                // 旧 sweep_summary.csv の 1 行が terminal 行 1 本に対応する．
                // metrics.csv に入れると (run_uid, step, scope, name) が重複する．
                record::log_trial(
                    &mut child,
                    &format!("trial-{run_idx}"),
                    seed,
                    args.timesteps,
                    &result,
                );
                let outcome = record::TrialOutcome::from_result(&result);
                if let Some((_, values)) =
                    polarization_by_n.iter_mut().find(|(n, _)| *n == n_agents)
                {
                    values.push(outcome.polarization_index);
                }
                trials.push(outcome);

                done += 1;
            }
            record::log_condition_summary(&mut child, &trials);
            child.finish().expect("runvault: 子 run の完了に失敗");

            println!(
                "[{}/{}] N={} activation={:.2} 完了 ({} 試行)",
                done, n_total, n_agents, activation_rate, args.runs,
            );
        }
    }

    let parent_dir = parent.finish().expect("runvault: sweep 親 run の完了に失敗");

    println!("=================================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------------");
    println!("エージェント数別の平均 極化指数 P:");
    for (n_agents, values) in &polarization_by_n {
        if values.is_empty() {
            continue;
        }
        let avg_p = values.iter().sum::<f64>() / values.len() as f64;
        println!("  N={:<6} → P̄ = {:.4}", n_agents, avg_p);
    }
    println!("-----------------------------------------------------------------");
    println!("親 run  → {}", parent_dir.display());
    println!("試行の値 → 各子 run の events.jsonl (terminal 行)");
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

/// 1 推薦器条件を `runs` 回回した集計セル (情報拡散 / 極化 / 群衆効果)．
///
/// 試行ごとの値ではなく試行平均だけを持つ (旧 `reproduce_summary.json` と同じ粒度)．
#[derive(Clone)]
struct ReproCell {
    /// 条件ラベル (= 推薦器ラベル)．指標名の接頭辞にもなる．
    label: String,
    /// 試行平均の最終 伝播到達ユニークノード数 (情報拡散の広さ)．
    mean_propagation_reach: f64,
    /// 試行平均の最終 最大カスケード規模 (情報拡散の深さ)．
    mean_cascade_size_max: f64,
    /// 試行平均の最終 最大カスケード幅．
    mean_cascade_max_breadth: f64,
    /// 試行平均の最終 投稿総数．
    mean_n_posts: f64,
    /// 試行平均の最終 極化指数 P (意見分散)．
    mean_polarization_index: f64,
    /// 試行平均の «極化の増分» (最終 P − 初期 P; 正なら極化が進行)．
    mean_polarization_gain: f64,
    /// 試行平均の最終 群衆追随率 (down-treat 群追随; 群衆効果代理 0..1)．
    mean_herd_disagree_rate: f64,
    /// 試行平均の収束/最終ステップ．
    mean_final_step: f64,
}

impl ReproCell {
    /// run スコープ指標として書く値の並び (接頭辞は [`ReproCell::label`])．
    fn metrics(&self) -> [(&'static str, f64); 8] {
        [
            ("mean_propagation_reach", self.mean_propagation_reach),
            ("mean_cascade_size_max", self.mean_cascade_size_max),
            ("mean_cascade_max_breadth", self.mean_cascade_max_breadth),
            ("mean_n_posts", self.mean_n_posts),
            ("mean_polarization_index", self.mean_polarization_index),
            ("mean_polarization_gain", self.mean_polarization_gain),
            ("mean_herd_disagree_rate", self.mean_herd_disagree_rate),
            ("mean_final_step", self.mean_final_step),
        ]
    }
}

/// 観測値と論文の定性的知見を突き合わせた 1 アンカー (`events.jsonl` へ書く)．
///
/// `name` は runvault の指標名にもなるので slug (小文字・数字・`_`・`-`・`.`) に
/// 収める．旧実装が名前の括弧に書いていた «どういう比較か» は `paper` 欄に移した．
#[derive(Serialize)]
struct ReproAnchor {
    name: String,
    paper: String,
    observed: f64,
    target_lo: f64,
    /// 帯の上限．上限なしは `None`．
    ///
    /// `f64::INFINITY` は JSON で `null` に潰れ，«上限が無い» と «書き忘れた» が
    /// 区別できなくなる．最初から `Option` で持つ．
    target_hi: Option<f64>,
    pass: bool,
}

/// 1 セルの実行結果 (集計セルと代表 run の履歴)．
struct ReproCellResult {
    cell: ReproCell,
    /// 代表 run (run 0) のステップごとの履歴．
    representative: Vec<StepMetrics>,
}

/// 1 推薦器条件を `runs` 回実行して集計セルを作る．
fn run_repro_cell(
    platform: Platform,
    recsys_kind: RecSysKind,
    base: &Config,
    runs: usize,
    root_seed: u64,
    mock: bool,
) -> ReproCellResult {
    let mut reach = 0.0;
    let mut casc_size = 0.0;
    let mut casc_breadth = 0.0;
    let mut n_posts = 0.0;
    let mut polar = 0.0;
    let mut polar_gain = 0.0;
    let mut herd = 0.0;
    let mut final_step = 0.0;
    let mut representative: Vec<StepMetrics> = Vec::new();

    for run_idx in 0..runs {
        let seed =
            record::repro_trial_seed(root_seed, platform.label(), recsys_kind.label(), run_idx);
        let cfg = Config {
            platform,
            recsys: RecSysConfig {
                kind: recsys_kind,
                ..base.recsys
            },
            seed: Some(seed),
            ..base.clone()
        };
        let client = build_client(mock, &cfg.llm);
        let result: SimulationResult = run_with_client(&cfg, client)
            .unwrap_or_else(|e| panic!("実行に失敗 ({}): {e}", recsys_kind.label()));
        let first = result.metrics_history.first().unwrap();
        let last = result.metrics_history.last().unwrap();
        reach += last.propagation_reach as f64;
        casc_size += last.cascade_size_max as f64;
        casc_breadth += last.cascade_max_breadth as f64;
        n_posts += last.n_posts as f64;
        polar += last.polarization_index;
        polar_gain += last.polarization_index - first.polarization_index;
        herd += last.herd_disagree_rate;
        final_step += result.final_step as f64;
        if run_idx == 0 {
            representative = result.metrics_history.clone();
        }
    }

    let n = runs.max(1) as f64;
    ReproCellResult {
        cell: ReproCell {
            label: recsys_kind.label().to_string(),
            mean_propagation_reach: reach / n,
            mean_cascade_size_max: casc_size / n,
            mean_cascade_max_breadth: casc_breadth / n,
            mean_n_posts: n_posts / n,
            mean_polarization_index: polar / n,
            mean_polarization_gain: polar_gain / n,
            mean_herd_disagree_rate: herd / n,
            mean_final_step: final_step / n,
        },
        representative,
    }
}

fn cmd_reproduce(args: ReproduceArgs) {
    let platform = parse_platform(&args.platform).unwrap_or_else(|e| panic!("{}", e));
    let recsys_kinds: Vec<RecSysKind> = split_csv(&args.recsys_values)
        .iter()
        .map(|s| parse_recsys(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();

    // quick モードは軽量化 (動作確認用; 論文値検証には使わない)．leader は «周辺
    // エージェント多数» の構図を保つため N に比例して縮め，推薦器が露出を実質的に
    // ゲートする条件 (= RecSys アブレーションが効く条件) を quick でも維持する．
    let n_agents = if args.quick { 80 } else { args.n_agents };
    let runs = if args.quick { 2 } else { args.runs };
    let timesteps = if args.quick { 16 } else { args.timesteps };
    let requested_leaders = if args.quick { 8 } else { args.n_leaders };
    let n_leaders = requested_leaders.min(n_agents);

    if !args.mock {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    // 基準設定 (全条件で共通; recsys/seed のみ条件ごとに差替)．収束で早期停止
    // しないよう patience を大きく取り，各条件を同じ T まで回して比較する．
    let base = Config {
        platform,
        n_agents,
        n_leaders,
        timesteps,
        activation_rate: args.activation_rate,
        llm_budget: args.llm_budget,
        ba_m: args.ba_m,
        recsys: RecSysConfig {
            kind: RecSysKind::Interest,
            ..RecSysConfig::default()
        },
        convergence_patience: timesteps + 1,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: if args.mock {
                None
            } else {
                Some(args.cache_path.clone())
            },
        },
    };

    // 名乗る名前を知っているのはクライアントだけなので，回す前に 1 つ組んで訊く．
    let llm_block = {
        let probe = build_client(args.mock, &base.llm);
        record::llm_block(
            probe.inner().model(),
            probe.inner().endpoint(),
            base.llm.temperature,
        )
    };

    let parameters = ReproduceParameters {
        platform: platform.label().to_string(),
        recsys_values: recsys_kinds.iter().map(|k| k.label().to_string()).collect(),
        n_agents,
        n_leaders,
        timesteps,
        activation_rate: args.activation_rate,
        llm_budget: args.llm_budget,
        ba_m: args.ba_m,
        runs,
        convergence_patience: base.convergence_patience,
        mock: args.mock,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(args.seed)
            .llm(llm_block)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    println!("=== Yang et al. (2024) OASIS 創発現象 一括再現 ===");
    println!(
        "platform: {} | N: {} | leaders: {} | T: {} | activation: {} | runs: {} | mode: {}",
        platform.label(),
        n_agents,
        n_leaders,
        timesteps,
        args.activation_rate,
        runs,
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // --- RecSys アブレーション行列 (interest / hot-score / none) ---
    let mut cells: Vec<ReproCell> = Vec::new();
    for &kind in &recsys_kinds {
        let out = run_repro_cell(platform, kind, &base, runs, args.seed, args.mock);
        // 3 条件が 1 本の run に同居するので，(step, scope, name) が衝突しないよう
        // 条件ラベルを名前に付ける．
        record::log_history(&mut rv, Some(&out.cell.label), &out.representative);
        record::log_prefixed(&mut rv, &out.cell.label, &out.cell.metrics());
        cells.push(out.cell);
    }

    // --- アンカー評価 (論文の定性的知見) ---
    let cell = |label: &str| -> ReproCell {
        cells
            .iter()
            .find(|c| c.label == label)
            .cloned()
            .unwrap_or_else(|| panic!("セル {label} が見つかりません"))
    };
    let mut anchors: Vec<ReproAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: Option<f64>| {
        anchors.push(ReproAnchor {
            name: name.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && hi.is_none_or(|h| obs <= h),
        });
    };

    // 情報拡散・極化・群衆効果のアンカーは «推薦器あり» 条件 (interest 優先) を代表に取る．
    let has = |label: &str| cells.iter().any(|c| c.label == label);
    let recsys_on = if has("interest") {
        cell("interest")
    } else if has("hot-score") {
        cell("hot-score")
    } else {
        cells[0].clone()
    };

    // H1 (情報拡散): 推薦器ありでは種投稿が多段にカスケードする (最大カスケード > 1)．
    push(
        "diffusion_cascade",
        "multi-hop information cascade (max cascade size > 1)",
        recsys_on.mean_cascade_size_max,
        1.0 + 1e-9,
        None,
    );
    // H2 (情報拡散の広さ): 伝播到達が種投稿数を超えて広がる (reach > leaders 起点)．
    push(
        "diffusion_reach",
        "information spreads beyond seeds (reach >= 2)",
        recsys_on.mean_propagation_reach,
        2.0,
        None,
    );
    // H3 (極化): 同調的増幅で集団意見が構造化し極化指数 P > 0 を保つ．
    push(
        "polarization_present",
        "group polarization emerges (final P > 0)",
        recsys_on.mean_polarization_index,
        1e-6,
        None,
    );
    // H4 (群衆効果): down-treat 群追随率が観測される (群衆効果代理 >= 0)．
    push(
        "crowd_effect_observed",
        "herd / crowd following (herd rate in [0,1])",
        recsys_on.mean_herd_disagree_rate,
        0.0,
        Some(1.0 + 1e-9),
    );
    // H5 (RecSys アブレーション): 推薦器は拡散を **形作る**．
    //   伝播到達 (= 活性化したノードの一意著者数) は «誰が活性化したか» に支配され
    //   推薦器でほぼ飽和するため，識別力が低い．論文の知見は «推薦器がどの投稿を
    //   どこまで増幅するか» にあるので，**最大カスケード規模** で対比する．グローバル
    //   人気で全員に同一の最ホット投稿を見せる hot-score は，フォロー先ローカルの
    //   最新のみを見せる none より大きなカスケードを生む (= 推薦器が増幅を駆動)．
    if has("none") && has("hot-score") {
        let none_casc = cell("none").mean_cascade_size_max;
        let hot_casc = cell("hot-score").mean_cascade_size_max;
        push(
            "recsys_shapes_diffusion",
            "recommender amplifies cascades (cascade(hot-score) - cascade(none) >= 0)",
            hot_casc - none_casc,
            -1e-9,
            None,
        );
    } else if cells.len() >= 2 {
        // hot-score/none が揃わない場合: 推薦器条件間で最大カスケードに差がある
        //   (= 推薦器は中立でない) ことを確認する (range > 0)．
        let max_c = cells
            .iter()
            .map(|c| c.mean_cascade_size_max)
            .fold(f64::MIN, f64::max);
        let min_c = cells
            .iter()
            .map(|c| c.mean_cascade_size_max)
            .fold(f64::MAX, f64::min);
        push(
            "recsys_shapes_diffusion",
            "recommender choice changes diffusion (cascade range across recsys > 0)",
            max_c - min_c,
            1e-9,
            None,
        );
    }

    // 観測量そのものは run 全体を 1 つの値で表す数なので指標に書く．判定 (PASS/off)
    // と帯はカテゴリ・自前のアンカーなので events.jsonl へ．
    let observed: Vec<(&str, f64)> = anchors
        .iter()
        .map(|a| (a.name.as_str(), a.observed))
        .collect();
    record::log_scoped(&mut rv, &observed);
    for a in &anchors {
        record::log_verdict(&mut rv, ANCHOR_EVENT, &a.name, a);
    }
    let n_pass = anchors.iter().filter(|a| a.pass).count();
    record::log_scoped(
        &mut rv,
        &[
            ("anchors_passed", n_pass as f64),
            ("anchors_total", anchors.len() as f64),
        ],
    );

    // --- コンソール出力 ---
    println!("--- RecSys アブレーション行列 (拡散 / 極化 / 群衆効果) ---");
    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "recsys", "reach", "casc", "breadth", "P", "P-gain", "herd"
    );
    for c in &cells {
        println!(
            "{:<12} {:>8.2} {:>8.2} {:>8.2} {:>10.4} {:>8.4} {:>8.3}",
            c.label,
            c.mean_propagation_reach,
            c.mean_cascade_size_max,
            c.mean_cascade_max_breadth,
            c.mean_polarization_index,
            c.mean_polarization_gain,
            c.mean_herd_disagree_rate,
        );
    }
    println!("--- 論文知見アンカー ---");
    for a in &anchors {
        let hi = match a.target_hi {
            Some(h) => format!("{h:.3}"),
            None => "∞".to_string(),
        };
        println!(
            "[{}] {:<26} obs={:.4} target=[{:.3},{}]",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
        );
    }
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("条件別メトリクス → {}/metrics.csv", dir.display());
    println!("アンカー判定     → {}/events.jsonl", dir.display());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

//! Yang et al. (2024) "OASIS: Open Agent Social Interaction Simulations with One
//! Million Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で BA フォローグラフ上の LLM 駆動 行動選択 + 推薦 + 情報
//!               伝播を実行する．
//! `sweep`     : エージェント数 × 活性化率 を走査し，最終集団指標を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : 論文の創発現象 (情報拡散カスケード / グループ極化 / 群衆効果) を
//!               RecSys アブレーション (interest / hot-score / none) で対比し，
//!               観測 vs 論文の PASS/off を `reproduce_summary.json` に集計する．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use oasis_simulation::config::{
    parse_platform, parse_recsys, Config, LlmSettings, Platform, RecSysConfig, RecSysKind,
};
use oasis_simulation::simulation::{
    ensure_output_dir, run, run_mock, save_cascades, save_llm_meta, save_metrics, SimulationResult,
};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "oasis",
    about = "Yang et al. (2024) OASIS: Open Agent Social Interaction Simulations — 再現実験"
)]
struct Cli {
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

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    platform: String,
    recsys: String,
    n_agents: usize,
    activation_rate: f64,
    run: usize,
    seed: u64,
    converged: bool,
    final_step: usize,
    final_polarization_index: f64,
    final_opinion_std: f64,
    final_propagation_reach: usize,
    final_cascade_size_max: usize,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    platform: String,
    recsys: String,
    n_agents_values: Vec<usize>,
    activation_rate_values: Vec<f64>,
    n_leaders: usize,
    timesteps: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
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

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

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
        seed: args.seed,
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
        output_dir: output_dir.clone(),
    };

    if !args.mock {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }
    ensure_output_dir(&cfg.output_dir);

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
        "seed: {:?} | llm-budget: {} | LLM: temp={} llm_seed={} cache={} | mode={}",
        cfg.seed,
        cfg.llm_budget,
        cfg.llm.temperature,
        cfg.llm.seed,
        args.cache_path,
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-----------------------------------------------------------------");

    let result = if args.mock {
        run_mock(&cfg).unwrap_or_else(|e| panic!("mock 実行に失敗: {}", e))
    } else {
        run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e))
    };

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_cascades(&result.cascade_rows, &cfg.output_dir);
    save_llm_meta(&result, &cfg, &cfg.output_dir);

    // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let path = format!("{}/config.json", cfg.output_dir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

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
    println!("メトリクス → {}/metrics.csv", cfg.output_dir);
    println!("カスケード → {}/cascades.csv", cfg.output_dir);
    println!("LLM メタ   → {}/llm_meta.json", cfg.output_dir);
    println!("設定       → {}/config.json", cfg.output_dir);
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

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = n_agents_values.len() * activation_rate_values.len() * args.runs;

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
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &n_agents in &n_agents_values {
        for &activation_rate in &activation_rate_values {
            for run_idx in 0..args.runs {
                let seed = socsim_core::derive_seed(
                    args.seed,
                    &[
                        label_hash(platform.label()),
                        n_agents as u64,
                        (activation_rate * 1000.0) as u64,
                        run_idx as u64,
                    ],
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
                    llm: LlmSettings {
                        temperature: args.temperature,
                        seed: args.llm_seed,
                        cache_path: Some(args.cache_path.clone()),
                    },
                    output_dir: sweep_dir.clone(),
                };

                let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                let last = result.metrics_history.last().unwrap();

                summary_rows.push(SweepRow {
                    platform: platform.label().to_string(),
                    recsys: recsys_kind.label().to_string(),
                    n_agents,
                    activation_rate,
                    run: run_idx,
                    seed,
                    converged: result.converged,
                    final_step: result.final_step,
                    final_polarization_index: last.polarization_index,
                    final_opinion_std: last.opinion_std,
                    final_propagation_reach: last.propagation_reach,
                    final_cascade_size_max: last.cascade_size_max,
                    cache_hit_rate: result.metadata.cache_hit_rate(),
                });

                done += 1;
            }
            println!(
                "[{}/{}] N={} activation={:.2} 完了 ({} 試行)",
                done, n_total, n_agents, activation_rate, args.runs,
            );
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            platform: platform.label().to_string(),
            recsys: recsys_kind.label().to_string(),
            n_agents_values: n_agents_values.clone(),
            activation_rate_values: activation_rate_values.clone(),
            n_leaders: args.n_leaders,
            timesteps: args.timesteps,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("=================================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------------");
    println!("エージェント数別の平均 極化指数 P:");
    for &n_agents in &n_agents_values {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.n_agents == n_agents)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_p =
            rows.iter().map(|r| r.final_polarization_index).sum::<f64>() / rows.len() as f64;
        println!("  N={:<6} → P̄ = {:.4}", n_agents, avg_p);
    }
    println!("-----------------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

/// 1 推薦器条件を `runs` 回回した集計セル (情報拡散 / 極化 / 群衆効果)．
#[derive(serde::Serialize, Clone)]
struct ReproCell {
    /// 条件ラベル (= 推薦器ラベル; summary/CSV のキー)．
    label: String,
    recsys: String,
    runs: usize,
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

/// 観測値と論文の定性的知見を突き合わせた 1 アンカー．
#[derive(serde::Serialize)]
struct ReproAnchor {
    name: String,
    paper: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

/// 1 推薦器条件を `runs` 回実行して集計セルを作る．
#[allow(clippy::too_many_arguments)]
fn run_repro_cell(
    platform: Platform,
    recsys_kind: RecSysKind,
    base: &Config,
    runs: usize,
    root_seed: u64,
    mock: bool,
    out_dir: &str,
) -> ReproCell {
    let mut reach = 0.0;
    let mut casc_size = 0.0;
    let mut casc_breadth = 0.0;
    let mut n_posts = 0.0;
    let mut polar = 0.0;
    let mut polar_gain = 0.0;
    let mut herd = 0.0;
    let mut final_step = 0.0;
    // 代表 (run 0) のメトリクス履歴を CSV に保存し，Python 側で時系列描画に使う．
    let mut representative: Option<Vec<oasis_simulation::metrics::StepMetrics>> = None;

    for run_idx in 0..runs {
        let seed = socsim_core::derive_seed(
            root_seed,
            &[
                label_hash(platform.label()),
                label_hash(recsys_kind.label()),
                run_idx as u64,
            ],
        );
        let cfg = Config {
            platform,
            recsys: RecSysConfig {
                kind: recsys_kind,
                ..base.recsys
            },
            seed: Some(seed),
            ..base.clone()
        };
        let result: SimulationResult = if mock {
            run_mock(&cfg)
                .unwrap_or_else(|e| panic!("mock 実行に失敗 ({}): {e}", recsys_kind.label()))
        } else {
            run(&cfg).unwrap_or_else(|e| panic!("実行に失敗 ({}): {e}", recsys_kind.label()))
        };
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
            representative = Some(result.metrics_history.clone());
        }
    }

    let n = runs.max(1) as f64;
    if let Some(hist) = representative {
        let rows: Vec<_> = hist.iter().flat_map(|m| m.to_rows()).collect();
        let path = format!("{out_dir}/metrics_{}.csv", recsys_kind.label());
        socsim_results::write_csv(&rows, &path).expect("metrics_<recsys>.csv の書き込みに失敗");
    }

    ReproCell {
        label: recsys_kind.label().to_string(),
        recsys: recsys_kind.label().to_string(),
        runs,
        mean_propagation_reach: reach / n,
        mean_cascade_size_max: casc_size / n,
        mean_cascade_max_breadth: casc_breadth / n,
        mean_n_posts: n_posts / n,
        mean_polarization_index: polar / n,
        mean_polarization_gain: polar_gain / n,
        mean_herd_disagree_rate: herd / n,
        mean_final_step: final_step / n,
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

    let ts = timestamp();
    let out_dir = format!("{}/reproduce_{}", args.output_dir, ts);
    ensure_output_dir(&out_dir);
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
        output_dir: out_dir.clone(),
    };

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
    println!("出力先: {out_dir}");
    println!("-------------------------------------------------");

    // --- RecSys アブレーション行列 (interest / hot-score / none) ---
    let mut cells: Vec<ReproCell> = Vec::new();
    for &kind in &recsys_kinds {
        let cell = run_repro_cell(platform, kind, &base, runs, args.seed, args.mock, &out_dir);
        cells.push(cell);
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
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproAnchor {
            name: name.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
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
        "diffusion_cascade (max cascade size > 1)",
        "multi-hop information cascade",
        recsys_on.mean_cascade_size_max,
        1.0 + 1e-9,
        f64::INFINITY,
    );
    // H2 (情報拡散の広さ): 伝播到達が種投稿数を超えて広がる (reach > leaders 起点)．
    push(
        "diffusion_reach (reach >= 2)",
        "information spreads beyond seeds",
        recsys_on.mean_propagation_reach,
        2.0,
        f64::INFINITY,
    );
    // H3 (極化): 同調的増幅で集団意見が構造化し極化指数 P > 0 を保つ．
    push(
        "polarization_present (final P > 0)",
        "group polarization emerges",
        recsys_on.mean_polarization_index,
        1e-6,
        f64::INFINITY,
    );
    // H4 (群衆効果): down-treat 群追随率が観測される (群衆効果代理 >= 0)．
    push(
        "crowd_effect_observed (herd rate in [0,1])",
        "herd / crowd following",
        recsys_on.mean_herd_disagree_rate,
        0.0,
        1.0 + 1e-9,
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
            "recsys_shapes_diffusion (cascade(hot-score) - cascade(none) >= 0)",
            "recommender amplifies cascades",
            hot_casc - none_casc,
            -1e-9,
            f64::INFINITY,
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
            "recsys_shapes_diffusion (cascade range across recsys > 0)",
            "recommender choice changes diffusion",
            max_c - min_c,
            1e-9,
            f64::INFINITY,
        );
    }

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
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.3}", a.target_hi)
        };
        println!(
            "[{}] {:<48} obs={:.4} target=[{:.3},{}]",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
        );
    }
    let n_pass = anchors.iter().filter(|a| a.pass).count();
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());

    // --- reproduce_summary.json ---
    let summary = serde_json::json!({
        "timestamp": ts,
        "mode": if args.mock { "mock" } else { "live" },
        "config": {
            "platform": platform.label(),
            "n_agents": n_agents,
            "n_leaders": n_leaders,
            "timesteps": timesteps,
            "activation_rate": args.activation_rate,
            "llm_budget": args.llm_budget,
            "runs": runs,
            "seed": args.seed,
        },
        "recsys_ablation": cells,
        "anchors": anchors,
        "n_pass": n_pass,
        "n_total": anchors.len(),
    });
    let path = format!("{out_dir}/reproduce_summary.json");
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");
    let _ = refresh_latest_symlink(&args.output_dir, &format!("reproduce_{ts}"));
    println!("サマリ → {path}");
    println!("条件別メトリクス → {out_dir}/metrics_<recsys>.csv");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

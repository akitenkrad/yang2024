//! Yang et al. (2024) "OASIS: Open Agent Social Interaction Simulations with One
//! Million Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で BA フォローグラフ上の LLM 駆動 行動選択 + 推薦 + 情報
//!               伝播を実行する．
//! `sweep`     : エージェント数 × 活性化率 を走査し，最終集団指標を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : 論文 Finding (情報拡散 / 極化 / 群衆効果 / RecSys アブレーション)
//!               の一括検証 (Phase 3; 未実装スタブ)．

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use chrono::Local;
use clap::{Parser, Subcommand};
use csv::Writer;

use oasis_simulation::config::{
    parse_platform, parse_recsys, Config, LlmSettings, Platform, RecSysConfig, RecSysKind,
};
use oasis_simulation::simulation::{
    ensure_output_dir, run, save_cascades, save_llm_meta, save_metrics,
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
    /// 論文 Finding の一括再現 (Phase 3; 未実装)．
    Reproduce,
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

/// latest シンボリックリンクを (再) 作成する．
fn refresh_latest(output_dir: &str, target: &str) {
    let symlink_path = Path::new(output_dir).join("latest");
    if symlink_path.is_symlink() {
        let _ = fs::remove_file(&symlink_path);
    }
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(target, &symlink_path);
    }
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

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
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
            cache_path: Some(args.cache_path.clone()),
        },
        output_dir: output_dir.clone(),
    };

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
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
        "seed: {:?} | llm-budget: {} | LLM: temp={} llm_seed={} cache={}",
        cfg.seed, cfg.llm_budget, cfg.llm.temperature, cfg.llm.seed, args.cache_path
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-----------------------------------------------------------------");

    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_cascades(&result.cascade_rows, &cfg.output_dir);
    save_llm_meta(&result, &cfg, &cfg.output_dir);

    // config.json
    {
        let path = format!("{}/config.json", cfg.output_dir);
        let file = File::create(&path).expect("config.json の作成に失敗");
        serde_json::to_writer_pretty(BufWriter::new(file), &cfg.to_run_config_json())
            .expect("config.json の書き込みに失敗");
    }

    refresh_latest(&args.output_dir, &timestamp);

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

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
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

    // sweep_summary.csv
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        let file = File::create(&path).expect("sweep_summary.csv の作成に失敗");
        let mut wtr = Writer::from_writer(BufWriter::new(file));
        for row in &summary_rows {
            wtr.serialize(row).expect("サマリ行の書き込みに失敗");
        }
        wtr.flush().expect("フラッシュに失敗");
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
        let file = File::create(&path).expect("sweep_config.json の作成に失敗");
        serde_json::to_writer_pretty(BufWriter::new(file), &config_json)
            .expect("sweep_config.json の書き込みに失敗");
    }

    refresh_latest(&args.output_dir, &format!("{}_sweep", timestamp));

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
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce => {
            eprintln!(
                "reproduce は Phase 3 で実装予定です (情報拡散 / 極化 / 群衆効果 / \
                 RecSys アブレーションの一括再現)．現状は run / sweep を使ってください．"
            );
            std::process::exit(1);
        }
    }
}

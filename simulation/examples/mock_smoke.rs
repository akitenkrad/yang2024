//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / cascades.csv / llm_meta.json / config.json) と
//! Python 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient`
//! で決定論的に leader の行動を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use socsim_results::{refresh_latest_symlink, timestamp, write_json};

use oasis_simulation::config::{Config, LlmSettings, Platform, RecSysConfig, RecSysKind};
use oasis_simulation::llm::wrap_client;
use oasis_simulation::simulation::{
    ensure_output_dir, run_with_client, save_cascades, save_llm_meta, save_metrics,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = timestamp();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        platform: Platform::X,
        n_agents: 40,
        n_leaders: 8,
        timesteps: 12,
        activation_rate: 0.5,
        llm_budget: 1000,
        ba_m: 3,
        recsys: RecSysConfig {
            kind: RecSysKind::Interest,
            ..RecSysConfig::default()
        },
        convergence_patience: 100, // 収束で早期停止させない
        seed: Some(42),
        llm: LlmSettings::default(),
        output_dir: output_dir.clone(),
    };

    // leader 擬似挙動: フィードがあれば先頭をリポスト，無ければ新規投稿する．
    // これにより情報カスケードが多段に伸び，極化ドリフトも進む．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        if prompt.contains("author=") {
            "THOUGHT: worth amplifying.\nACTION: repost\nTARGET: 0\nCONTENT: -".to_string()
        } else {
            "THOUGHT: I will weigh in.\nACTION: post\nTARGET: -\nCONTENT: My stance on the topic."
                .to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_cascades(&result.cascade_rows, &cfg.output_dir);
    save_llm_meta(&result, &cfg, &cfg.output_dir);

    // config.json (socsim_results::write_json に委譲)．
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    write_json(&cfg.to_run_config_json(), &cfg_path).unwrap();

    // latest symlink (socsim_results に委譲)．
    let _ = refresh_latest_symlink(&base, &timestamp);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
    println!(
        "final P={:.4} opinion_std={:.4} reach={} cascade_max={} posts={} steps={}",
        last.polarization_index,
        last.opinion_std,
        last.propagation_reach,
        last.cascade_size_max,
        last.n_posts,
        result.final_step
    );
}

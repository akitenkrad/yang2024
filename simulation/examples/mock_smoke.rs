//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (runvault の run ディレクトリ) と Python 可視化を検証する
//! ための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に leader の
//! 行動を駆動し，本番 `run` と同じ経路で結果を記録する．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};

use oasis_simulation::config::{Config, LlmSettings, Platform, RecSysConfig, RecSysKind};
use oasis_simulation::llm::wrap_client;
use oasis_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use oasis_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

/// シードは固定 (スモークの目的は «同じ入力で同じ出力» の確認)．
const SEED: u64 = 42;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());

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
        seed: Some(SEED),
        llm: LlmSettings::default(),
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
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let parameters = cfg.to_run_config_json(SEED, true);
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(SEED)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    record::log_simulation(&mut rv, &result);
    record::log_cascades(&mut rv, &result.cascade_rows);

    let last = result.metrics_history.last().unwrap();
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
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

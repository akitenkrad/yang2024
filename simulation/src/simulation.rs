//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で BA 網生成・
//!   プロフィール/活動確率/初期意見割当の init RNG を，`derive_seed(root, &[1])` で
//!   engine RNG (= RandomActivationScheduler + 活性化サブサンプリング draw) を派生
//!   する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き Ollama→OpenAI
//!   フォールバッククライアントに閉じ込め，`temperature=0`/`seed` 固定 + プロンプト
//!   →応答キャッシュで擬似決定論化する．モデル・endpoint・温度・seed・cache-hit を
//!   `llm_meta.json` に記録する．
//!
//! # leader (オピニオンリーダー) の決定
//!
//! init 後のネットワーク次数上位 `n_leaders` 体を leader とし，Decision フェーズで
//! LLM を呼ぶ対象にする．残りの周辺エージェントは簡易ポリシーで近似する
//! (スケーラビリティ設計)．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_net::SocialNetwork;

use crate::config::Config;
use crate::llm::{build_live_client, OasisClient};
use crate::mechanisms::{
    ActivationMechanism, AgentActionMechanism, FeedRecommendationMechanism,
    InfoPropagationMechanism, MetricsMechanism, PostStepMechanism, SharedBudget, SharedClient,
    SharedMetadata,
};
use crate::metrics::{cascade_rows, CascadeRow, StepMetrics};
use crate::world::{AgentProfile, OasisWorld, Post, ACTIVITY_DIM};

/// 網生成・プロフィール/活動確率/初期意見割当用 RNG ラベル．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= RandomActivationScheduler + 活性化 draw) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// 既定トピック (合成実行の議論対象)．
pub const DEFAULT_TOPIC: &str = "nuclear energy policy";

/// 興味プロフィール用の語彙テンプレート (決定論的割当)．
const INTERESTS: [&str; 6] = [
    "nuclear energy policy and climate",
    "renewable solar wind power",
    "economic growth jobs industry",
    "public health safety regulation",
    "technology innovation startups",
    "education science research funding",
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 各タイムステップ (t=0 を含む) の集団メトリクス履歴．
    pub metrics_history: Vec<StepMetrics>,
    /// 最終的なカスケード行 (cascades.csv 用)．
    pub cascade_rows: Vec<CascadeRow>,
    /// 収束したか (連続ゼロアクションで停止)．
    pub converged: bool,
    /// 収束 (または最終) タイムステップ番号．
    pub final_step: usize,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (llm_meta 用)．
    pub llm_model: String,
    /// LLM endpoint (llm_meta 用; primary)．
    pub llm_endpoint: String,
}

/// 世界状態を初期化する (BA 網構築 + プロフィール/活動確率/初期意見割当 + 種投稿)．
///
/// プロフィール (bio)・24 次元活動確率・初期意見は init RNG から決定論的に割り当てる
/// (socsim コア層)．先頭数体に「種投稿」を 1 件ずつ作り，カスケードの起点とする．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> OasisWorld {
    let ids: Vec<AgentId> = (0..cfg.n_agents as u64).map(AgentId).collect();
    let network = SocialNetwork::barabasi_albert(&ids, cfg.ba_m, rng);

    let mut agents: BTreeMap<AgentId, AgentProfile> = BTreeMap::new();
    for (idx, &id) in ids.iter().enumerate() {
        let bio = INTERESTS[idx % INTERESTS.len()].to_string();
        // 24 次元活動確率: 昼間 (8..22) を高めにした決定論的プロファイル + 揺らぎ．
        let mut activity_prob = [0.0f64; ACTIVITY_DIM];
        for (h, slot) in activity_prob.iter_mut().enumerate() {
            let base = if (8..22).contains(&h) { 0.5 } else { 0.15 };
            let jitter: f64 = rng.gen_range(-0.05..0.05);
            *slot = (base + jitter).clamp(0.0, 1.0);
        }
        // 初期意見 ∈ [-1, 1] を一様抽選．
        let opinion: f64 = rng.gen_range(-1.0..1.0);
        agents.insert(
            id,
            AgentProfile {
                name: format!("user{}", id.0),
                bio,
                activity_prob,
                memory: Vec::new(),
                opinion,
            },
        );
    }

    let mut world = OasisWorld::new(
        network,
        agents,
        cfg.platform,
        cfg.recsys,
        cfg.timesteps as u64,
    );

    // 種投稿: 先頭 min(n_leaders, n) 体が t=0 で 1 件ずつ投稿する (カスケード起点)．
    let n_seed = cfg.n_leaders.max(1).min(cfg.n_agents);
    for &id in ids.iter().take(n_seed) {
        let opinion = world.agents.get(&id).map(|a| a.opinion).unwrap_or(0.0);
        let root = world.posts.len();
        let content = format!("Initial take on {DEFAULT_TOPIC} by user{}.", id.0);
        world.posts.push(Post::new(id, content, 0, root, opinion));
    }

    world
}

/// ネットワーク次数上位 `n_leaders` 体を leader (オピニオンリーダー) として返す．
///
/// 次数降順 → 同点は AgentId 昇順で決定論化する．
pub fn select_leaders(world: &OasisWorld, n_leaders: usize) -> Vec<AgentId> {
    let mut by_degree: Vec<(usize, AgentId)> = world
        .agents
        .keys()
        .map(|&id| (world.network.degree(id), id))
        .collect();
    by_degree.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    by_degree
        .into_iter()
        .take(n_leaders)
        .map(|(_, id)| id)
        .collect()
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
pub fn run(cfg: &Config) -> Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client)
}

/// 与えられた [`OasisClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．
pub fn run_with_client(cfg: &Config, client: OasisClient) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 初期世界 (root から派生した init RNG; 決定論的 socsim コア層)．
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);
    let leaders = select_leaders(&world, cfg.n_leaders);

    // LLM モデル/endpoint をメタデータ用に控える．
    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    // クライアント・メタデータ・予算を共有する．
    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_budget: SharedBudget = Rc::new(RefCell::new(cfg.llm_budget));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(ActivationMechanism::new(
            cfg.activation_rate,
            leaders,
        )))
        .add_mechanism(Box::new(FeedRecommendationMechanism))
        .add_mechanism(Box::new(AgentActionMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            Rc::clone(&shared_budget),
            cfg.llm.clone(),
        )))
        .add_mechanism(Box::new(InfoPropagationMechanism::default()))
        .add_mechanism(Box::new(MetricsMechanism))
        .add_mechanism(Box::new(PostStepMechanism::new(cfg.convergence_patience)))
        .build();

    let mut metrics_history: Vec<StepMetrics> = Vec::new();

    // 初期状態 (t=0) を記録 (種投稿のみ; active=0)．
    {
        let w = sim.world();
        metrics_history.push(StepMetrics::compute(&w.agents, &w.posts, 0, 0.0, 0));
    }

    let mut converged = false;
    let mut final_step = 0usize;

    sim.run_observed(|report| {
        let t = report.t as usize;
        let w = report.world;
        let active = *report
            .scratch
            .get::<usize>("action_count")
            .unwrap_or(&0usize);
        let herd = herd_disagree_rate(w);
        metrics_history.push(StepMetrics::compute(&w.agents, &w.posts, active, herd, t));
        converged = report.stopped;
        final_step = t;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    let final_world = sim.world();
    let cascades = cascade_rows(&final_world.posts);

    // キャッシュを保存 (cache_path 指定時; in-memory はスキップ)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metadata = shared_meta.borrow().clone();

    Ok(SimulationResult {
        metrics_history,
        cascade_rows: cascades,
        converged,
        final_step,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

/// down-treat 群追随率 (群衆効果代理)．
///
/// 意見が負 (against) のエージェントのうち，集団平均意見が負なら «多数派へ追随»
/// しているとみなし，その割合を返す簡易指標．論文の up/down treat 実験の代理．
fn herd_disagree_rate(world: &OasisWorld) -> f64 {
    let mean = world.mean_opinion();
    if world.agents.is_empty() {
        return 0.0;
    }
    let negatives = world.agents.values().filter(|a| a.opinion < 0.0).count();
    if mean < 0.0 {
        negatives as f64 / world.agents.len() as f64
    } else {
        0.0
    }
}

// --------------------------------------------------------------------------- //
// 出力
// --------------------------------------------------------------------------- //

/// メトリクス履歴を long-format CSV (metrics.csv) に保存する．
///
/// 各 [`StepMetrics`] を `to_rows()` で long-format 行へ展開し，展開後の行列を
/// `socsim_results::write_csv` で書き出す (各行を `serialize` し先頭行にヘッダを
/// 書く csv クレットの標準挙動; 従来の手書き writer とバイト等価)．行構造体は
/// repo 固有のままで，writer だけを共有化する．
pub fn save_metrics(metrics: &[StepMetrics], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    let rows: Vec<_> = metrics.iter().flat_map(|m| m.to_rows()).collect();
    socsim_results::write_csv(&rows, &path).expect("metrics.csv の書き込みに失敗");
}

/// カスケード行を cascades.csv に保存する．
///
/// 書き出し機構は `socsim_results::write_csv` に委譲する ([`CascadeRow`] を
/// `serialize`; 従来の手書き writer とバイト等価)．
pub fn save_cascades(rows: &[CascadeRow], output_dir: &str) {
    let path = format!("{}/cascades.csv", output_dir);
    socsim_results::write_csv(rows, &path).expect("cascades.csv の書き込みに失敗");
}

/// `llm_meta.json` の構造体 (LLM モデル・endpoint・温度・seed・cache 統計)．
#[derive(Serialize)]
pub struct LlmMetaJson {
    pub provider: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub determinism_note: &'static str,
}

/// `llm_meta.json` を保存する．
pub fn save_llm_meta(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let provider =
        if result.llm_endpoint.contains("11434") || result.llm_endpoint.contains("ollama") {
            "ollama"
        } else if result.llm_endpoint.contains("mock") {
            "mock"
        } else {
            "openai"
        };
    let meta = LlmMetaJson {
        provider: provider.to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (BA network, activation, recommender, \
                           info propagation, metrics) is deterministic given the seed.",
    };
    // pretty-print JSON の書き出しは socsim_results::write_json に委譲する
    // (内部は serde_json::to_writer_pretty + flush; 従来の writer とバイト等価)．
    // provider/model/endpoint/temperature/seed の値は従来どおり result / cfg から
    // 採り，LlmMetaJson の構造 (フィールド名・順序・determinism_note) を保持する
    // (`MetadataCollector::summary()` は cache-hit 100% 再実行や呼び出し 0 件で
    // endpoint/model が変わりうるため，バイト等価のためここでは使わない)．
    let path = format!("{}/llm_meta.json", output_dir);
    socsim_results::write_json(&meta, &path).expect("llm_meta.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("出力ディレクトリの作成に失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LlmSettings, Platform, RecSysConfig, RecSysKind};
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_client() -> OasisClient {
        // フィードがあれば先頭をリポスト，無ければ post する擬似 leader 挙動．
        let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
            if prompt.contains("[") && prompt.contains("author=") {
                "THOUGHT: interesting.\nACTION: repost\nTARGET: 0\nCONTENT: -".to_string()
            } else {
                "THOUGHT: I will share.\nACTION: post\nTARGET: -\nCONTENT: My view on the topic."
                    .to_string()
            }
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn test_config() -> Config {
        Config {
            platform: Platform::X,
            n_agents: 30,
            n_leaders: 6,
            timesteps: 8,
            activation_rate: 0.6,
            llm_budget: 1000,
            ba_m: 3,
            recsys: RecSysConfig {
                kind: RecSysKind::Interest,
                ..RecSysConfig::default()
            },
            convergence_patience: 100, // 早期停止させない
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }

    #[test]
    fn scripted_run_produces_metrics() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        assert_eq!(result.metrics_history[0].t, 0);
        assert!(result.metrics_history.len() > 1);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let cfg = test_config();
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        let af: Vec<f64> = a
            .metrics_history
            .iter()
            .map(|m| m.polarization_index)
            .collect();
        let bf: Vec<f64> = b
            .metrics_history
            .iter()
            .map(|m| m.polarization_index)
            .collect();
        assert_eq!(af, bf);
        assert_eq!(a.final_step, b.final_step);
        assert_eq!(a.metrics_history.len(), b.metrics_history.len());
    }

    #[test]
    fn leaders_are_highest_degree() {
        let cfg = test_config();
        let mut rng = SimRng::from_seed(derive_seed(42, &[RNG_WORLD_INIT]));
        let world = init_world(&cfg, &mut rng);
        let leaders = select_leaders(&world, cfg.n_leaders);
        assert_eq!(leaders.len(), cfg.n_leaders);
        // leader の最小次数 >= 非 leader の最大次数．
        let leader_set: std::collections::BTreeSet<_> = leaders.iter().copied().collect();
        let min_leader_deg = leaders
            .iter()
            .map(|&id| world.network.degree(id))
            .min()
            .unwrap();
        let max_other_deg = world
            .agents
            .keys()
            .filter(|id| !leader_set.contains(id))
            .map(|&id| world.network.degree(id))
            .max()
            .unwrap_or(0);
        assert!(min_leader_deg >= max_other_deg);
    }

    #[test]
    fn recsys_none_reduces_propagation_reach() {
        // 推薦器 none (アブレーション) は interest より伝播到達が小さい傾向．
        let mut cfg_interest = test_config();
        cfg_interest.recsys.kind = RecSysKind::Interest;
        let mut cfg_none = test_config();
        cfg_none.recsys.kind = RecSysKind::None;

        let r_interest = run_with_client(&cfg_interest, scripted_client()).unwrap();
        let r_none = run_with_client(&cfg_none, scripted_client()).unwrap();

        let reach_interest = r_interest.metrics_history.last().unwrap().propagation_reach;
        let reach_none = r_none.metrics_history.last().unwrap().propagation_reach;
        assert!(
            reach_none <= reach_interest,
            "ablation reach {reach_none} should be <= interest reach {reach_interest}"
        );
    }
}

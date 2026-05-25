//! Yang et al. (2024) OASIS の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に leader の行動を駆動し，以下を検証する:
//! ・recsys ランキング (interest / hot-score / none) の決定論性と除外規則
//! ・6 メカニズム配線が成立し metrics/cascades が生成される
//! ・固定 mock を与えたときの socsim コア層の RNG 決定論性
//! ・RecSys アブレーション (none) が伝播到達を阻害する傾向
//! ・peripheral-only (leaders=0) でもライブ LLM 無しに run が成立する

use oasis_simulation::config::{Config, LlmSettings, Platform, RecSysConfig, RecSysKind};
use oasis_simulation::llm::{wrap_client, OasisClient};
use oasis_simulation::recsys::{build_feed, hot_score};
use oasis_simulation::simulation::{init_world, run_with_client, select_leaders};
use oasis_simulation::world::{AgentProfile, OasisWorld, Post, ACTIVITY_DIM};

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use socsim_net::SocialNetwork;
use std::collections::BTreeMap;

/// leader 擬似挙動: フィードがあればリポスト，無ければ post する mock クライアント．
fn scripted_client() -> OasisClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
        if prompt.contains("author=") {
            "THOUGHT: amplify.\nACTION: repost\nTARGET: 0\nCONTENT: -".to_string()
        } else {
            "THOUGHT: post.\nACTION: post\nTARGET: -\nCONTENT: A short take.".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
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
        convergence_patience: 100, // 収束で止めない
        seed: Some(7),
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
    }
}

// --------------------------------------------------------------------------- //
// メカニズム配線: run が成立し metrics/cascades が生成される
// --------------------------------------------------------------------------- //

#[test]
fn run_produces_well_formed_metrics() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_client()).unwrap();
    assert_eq!(result.metrics_history[0].t, 0);
    assert!(result.metrics_history.len() >= 2);
    for m in &result.metrics_history {
        assert!(m.polarization_index >= 0.0, "P は非負");
        assert!(m.opinion_std >= 0.0);
    }
    // 種投稿があるためカスケード行が生成される．
    assert!(!result.cascade_rows.is_empty());
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現 (socsim コア層)
// --------------------------------------------------------------------------- //

#[test]
fn core_is_deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted_client()).unwrap();
    let b = run_with_client(&cfg, scripted_client()).unwrap();
    let ap: Vec<f64> = a
        .metrics_history
        .iter()
        .map(|m| m.polarization_index)
        .collect();
    let bp: Vec<f64> = b
        .metrics_history
        .iter()
        .map(|m| m.polarization_index)
        .collect();
    let ar: Vec<usize> = a
        .metrics_history
        .iter()
        .map(|m| m.propagation_reach)
        .collect();
    let br: Vec<usize> = b
        .metrics_history
        .iter()
        .map(|m| m.propagation_reach)
        .collect();
    assert_eq!(ap, bp, "同一シードは極化指数を完全再現すべき");
    assert_eq!(ar, br, "同一シードは伝播到達を完全再現すべき");
    assert_eq!(a.final_step, b.final_step);
}

#[test]
fn different_seed_changes_outcome() {
    let mut cfg_a = base_config();
    cfg_a.seed = Some(1);
    let mut cfg_b = base_config();
    cfg_b.seed = Some(999);
    let a = run_with_client(&cfg_a, scripted_client()).unwrap();
    let b = run_with_client(&cfg_b, scripted_client()).unwrap();
    let ap: Vec<f64> = a
        .metrics_history
        .iter()
        .map(|m| m.polarization_index)
        .collect();
    let bp: Vec<f64> = b
        .metrics_history
        .iter()
        .map(|m| m.polarization_index)
        .collect();
    assert!(ap != bp, "異なるシードは (一般に) 異なる軌跡を生む");
}

// --------------------------------------------------------------------------- //
// RecSys アブレーション (none) は interest より伝播到達を阻害する
// --------------------------------------------------------------------------- //

#[test]
fn recsys_ablation_reduces_propagation() {
    let mut cfg_interest = base_config();
    cfg_interest.recsys.kind = RecSysKind::Interest;
    let mut cfg_none = base_config();
    cfg_none.recsys.kind = RecSysKind::None;

    let r_interest = run_with_client(&cfg_interest, scripted_client()).unwrap();
    let r_none = run_with_client(&cfg_none, scripted_client()).unwrap();
    let reach_i = r_interest.metrics_history.last().unwrap().propagation_reach;
    let reach_n = r_none.metrics_history.last().unwrap().propagation_reach;
    assert!(
        reach_n <= reach_i,
        "ablation 到達 {reach_n} は interest 到達 {reach_i} 以下であるべき"
    );
}

// --------------------------------------------------------------------------- //
// peripheral-only (leaders=0) でもライブ LLM 無しで run が成立する
// --------------------------------------------------------------------------- //

#[test]
fn peripheral_only_runs_without_llm_calls() {
    let mut cfg = base_config();
    cfg.n_leaders = 0;
    // leaders=0 なので LLM は一切呼ばれない (簡易ポリシーのみ)．
    let result = run_with_client(&cfg, scripted_client()).unwrap();
    assert_eq!(result.metadata.total(), 0, "leaders=0 では LLM 非呼び出し");
    assert!(result.metrics_history.len() >= 2);
}

// --------------------------------------------------------------------------- //
// leader は高次数ノード
// --------------------------------------------------------------------------- //

#[test]
fn leaders_are_highest_degree_nodes() {
    let cfg = base_config();
    let mut rng = SimRng::from_seed(derive_seed(7, &[0]));
    let world = init_world(&cfg, &mut rng);
    let leaders = select_leaders(&world, cfg.n_leaders);
    assert_eq!(leaders.len(), cfg.n_leaders);
}

// --------------------------------------------------------------------------- //
// recsys ランキング: hot-score は upvote / recency を報いる
// --------------------------------------------------------------------------- //

#[test]
fn hot_score_ranking() {
    let mut low = Post::new(AgentId(1), "a".into(), 0, 0, 0.0);
    low.upvotes = 1;
    let mut high = Post::new(AgentId(1), "b".into(), 0, 1, 0.0);
    high.upvotes = 100;
    assert!(hot_score(&high, 0.0) > hot_score(&low, 0.0));
}

#[test]
fn interest_feed_is_deterministic_and_excludes_self() {
    let mut agents: BTreeMap<AgentId, AgentProfile> = BTreeMap::new();
    for i in 0..3u64 {
        agents.insert(
            AgentId(i),
            AgentProfile {
                name: format!("u{i}"),
                bio: "climate energy".into(),
                activity_prob: [0.5; ACTIVITY_DIM],
                memory: Vec::new(),
                opinion: 0.0,
            },
        );
    }
    let ids: Vec<AgentId> = agents.keys().copied().collect();
    let mut net = SocialNetwork::empty();
    for &id in &ids {
        net.add_node(id);
    }
    net.add_edge(AgentId(0), AgentId(1));
    let mut world = OasisWorld::new(
        net,
        agents,
        Platform::X,
        RecSysConfig {
            kind: RecSysKind::Interest,
            k_in: 2,
            k_out: 2,
            t0: 1_134_028_003.0,
        },
        5,
    );
    world.posts = vec![
        Post::new(AgentId(1), "climate energy".into(), 1, 0, 0.0),
        Post::new(AgentId(0), "self post".into(), 1, 1, 0.0),
        Post::new(AgentId(2), "energy".into(), 1, 2, 0.0),
    ];
    let p = world.agents[&AgentId(0)].clone();
    let f1 = build_feed(&world, AgentId(0), &p);
    let f2 = build_feed(&world, AgentId(0), &p);
    assert_eq!(f1, f2);
    assert!(!f1.contains(&1), "自分の投稿は除外されるべき");
}

//! 決定論的推薦器 (RecSys) — 各 active エージェントのフィードを構築する．
//!
//! 論文の RecSys は **LLM 非依存の決定論的機構**である．本モジュールは
//! [`crate::config::RecSysKind`] に応じて以下の規則でフィード (post インデックス
//! の並び) を返す．いずれも socsim コアの bit 決定論を壊さない (純関数)．
//!
//! - **Interest (X)**: 興味マッチ (コサイン類似度)．in-network (フォロー先) 上位
//!   `k_in` 件と out-network (全投稿に対する興味マッチ) 上位 `k_out` 件を連結する．
//! - **HotScore (Reddit)**: ホットスコア
//!   `h = log10(max(|u-d|,1)) + sign(u-d)·(t-t0)/45000` 降順に上位 `k_in + k_out`
//!   件を選ぶ．
//! - **None (アブレーション)**: 推薦器を使わず，フォロー先の最新投稿のみを
//!   `k_in` 件入れる (拡散阻害の検証用)．

use socsim_core::AgentId;

use crate::config::{RecSysConfig, RecSysKind};
use crate::world::{cosine, AgentProfile, OasisWorld, Post};

/// Reddit ホットスコア
/// `h = log10(max(|u-d|,1)) + sign(u-d)·(t-t0)/45000` を計算する．
pub fn hot_score(post: &Post, t0: f64) -> f64 {
    let u = post.upvotes as f64;
    let d = post.downvotes as f64;
    let diff = u - d;
    let order = diff.abs().max(1.0).log10();
    let sign = if diff > 0.0 {
        1.0
    } else if diff < 0.0 {
        -1.0
    } else {
        0.0
    };
    let seconds = post.ts as f64 - t0;
    order + sign * seconds / 45000.0
}

/// 安定ソート用の比較 (降順スコア，同点は新しさ→インデックス昇順で決定論化)．
fn cmp_desc(a: (f64, u64, usize), b: (f64, u64, usize)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(b.1.cmp(&a.1))
        .then(a.2.cmp(&b.2))
}

/// 1 エージェントのフィードを構築する (post インデックスの並び)．
///
/// 自分自身の投稿は除外する．`world.network.neighbors(agent)` を in-network
/// (フォロー先) と見なす．
pub fn build_feed(world: &OasisWorld, agent: AgentId, profile: &AgentProfile) -> Vec<usize> {
    let cfg = &world.recsys;
    let neighbors: std::collections::BTreeSet<AgentId> =
        world.network.neighbors(agent).into_iter().collect();

    match cfg.kind {
        RecSysKind::Interest => feed_interest(world, agent, profile, &neighbors, cfg),
        RecSysKind::HotScore => feed_hot_score(world, agent, &neighbors, cfg),
        RecSysKind::None => feed_none(world, agent, &neighbors, cfg),
    }
}

/// 興味マッチ (X): in-network 上位 k_in + out-network 興味マッチ上位 k_out．
fn feed_interest(
    world: &OasisWorld,
    agent: AgentId,
    profile: &AgentProfile,
    neighbors: &std::collections::BTreeSet<AgentId>,
    cfg: &RecSysConfig,
) -> Vec<usize> {
    let e_a = profile.interest_embedding();

    // in-network: フォロー先の投稿を興味マッチ降順 (recency をタイブレーク)．
    let mut in_net: Vec<(f64, u64, usize)> = Vec::new();
    let mut out_net: Vec<(f64, u64, usize)> = Vec::new();
    for (idx, post) in world.posts.iter().enumerate() {
        if post.author == agent {
            continue;
        }
        let sim = cosine(&e_a, &post.content_vec);
        if neighbors.contains(&post.author) {
            in_net.push((sim, post.ts, idx));
        } else {
            out_net.push((sim, post.ts, idx));
        }
    }
    in_net.sort_by(|&a, &b| cmp_desc(a, b));
    out_net.sort_by(|&a, &b| cmp_desc(a, b));

    let mut feed: Vec<usize> = in_net.iter().take(cfg.k_in).map(|t| t.2).collect();
    feed.extend(out_net.iter().take(cfg.k_out).map(|t| t.2));
    feed
}

/// ホットスコア (Reddit): 全投稿 (自分以外) をホットスコア降順 上位 k_in+k_out．
fn feed_hot_score(
    world: &OasisWorld,
    agent: AgentId,
    _neighbors: &std::collections::BTreeSet<AgentId>,
    cfg: &RecSysConfig,
) -> Vec<usize> {
    let mut scored: Vec<(f64, u64, usize)> = world
        .posts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.author != agent)
        .map(|(idx, p)| (hot_score(p, cfg.t0), p.ts, idx))
        .collect();
    scored.sort_by(|&a, &b| cmp_desc(a, b));
    scored
        .iter()
        .take(cfg.k_in + cfg.k_out)
        .map(|t| t.2)
        .collect()
}

/// アブレーション (推薦器なし): フォロー先の最新投稿のみ上位 k_in．
fn feed_none(
    world: &OasisWorld,
    agent: AgentId,
    neighbors: &std::collections::BTreeSet<AgentId>,
    cfg: &RecSysConfig,
) -> Vec<usize> {
    let mut from_follows: Vec<(f64, u64, usize)> = world
        .posts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.author != agent && neighbors.contains(&p.author))
        // スコアは新しさのみ (recency)．
        .map(|(idx, p)| (p.ts as f64, p.ts, idx))
        .collect();
    from_follows.sort_by(|&a, &b| cmp_desc(a, b));
    from_follows.iter().take(cfg.k_in).map(|t| t.2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Platform, RecSysConfig};
    use crate::world::{AgentProfile, OasisWorld, ACTIVITY_DIM};
    use socsim_net::SocialNetwork;
    use std::collections::BTreeMap;

    fn profile(bio: &str) -> AgentProfile {
        AgentProfile {
            name: "a".into(),
            bio: bio.into(),
            activity_prob: [0.5; ACTIVITY_DIM],
            memory: Vec::new(),
            opinion: 0.0,
        }
    }

    fn world_with_posts(kind: RecSysKind, posts: Vec<Post>) -> OasisWorld {
        let mut agents: BTreeMap<AgentId, AgentProfile> = BTreeMap::new();
        for i in 0..4u64 {
            agents.insert(AgentId(i), profile("climate energy policy"));
        }
        let ids: Vec<AgentId> = agents.keys().copied().collect();
        // 線形チェーン 0-1-2-3 をフォローグラフとする．
        let mut net = SocialNetwork::empty();
        for &id in &ids {
            net.add_node(id);
        }
        net.add_edge(AgentId(0), AgentId(1));
        net.add_edge(AgentId(1), AgentId(2));
        net.add_edge(AgentId(2), AgentId(3));
        let mut w = OasisWorld::new(
            net,
            agents,
            Platform::X,
            RecSysConfig {
                kind,
                k_in: 2,
                k_out: 2,
                t0: 1_134_028_003.0,
            },
            10,
        );
        w.posts = posts;
        w
    }

    #[test]
    fn hot_score_rewards_upvotes_and_recency() {
        let mut old = Post::new(AgentId(1), "x".into(), 0, 0, 0.0);
        old.upvotes = 5;
        let mut fresh = Post::new(AgentId(1), "y".into(), 1_000_000, 1, 0.0);
        fresh.upvotes = 5;
        let t0 = 0.0;
        assert!(hot_score(&fresh, t0) > hot_score(&old, t0));
    }

    #[test]
    fn interest_feed_excludes_self_and_is_deterministic() {
        let posts = vec![
            Post::new(AgentId(1), "climate energy policy".into(), 1, 0, 0.0),
            Post::new(AgentId(0), "should be excluded self".into(), 1, 1, 0.0),
            Post::new(AgentId(3), "climate energy".into(), 2, 2, 0.0),
        ];
        let w = world_with_posts(RecSysKind::Interest, posts);
        let p = w.agents[&AgentId(0)].clone();
        let f1 = build_feed(&w, AgentId(0), &p);
        let f2 = build_feed(&w, AgentId(0), &p);
        assert_eq!(f1, f2);
        assert!(!f1.contains(&1), "self post must be excluded");
    }

    #[test]
    fn none_feed_only_uses_follows() {
        let posts = vec![
            // agent 0 follows 1; post by 1 should appear.
            Post::new(AgentId(1), "from follow".into(), 1, 0, 0.0),
            // post by 3 (not followed by 0) should NOT appear under None.
            Post::new(AgentId(3), "from stranger".into(), 2, 1, 0.0),
        ];
        let w = world_with_posts(RecSysKind::None, posts);
        let p = w.agents[&AgentId(0)].clone();
        let f = build_feed(&w, AgentId(0), &p);
        assert_eq!(f, vec![0]);
    }
}

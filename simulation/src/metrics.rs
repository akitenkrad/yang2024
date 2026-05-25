//! 評価指標 (論文 §5 の主要 Finding に対応)．
//!
//! 各タイムステップの世界状態から，極化指数・active-user 数・伝播到達数などの
//! 集団指標を計算する．カスケード規模 (投稿ごとのリポスト/コメント総数・最大幅)
//! は投稿ストアの `root` リンクから集計する．
//!
//! | 指標 | 定義 |
//! |------|------|
//! | `polarization_index` | 意見分散 P = (1/N)·Σ(o_a - ō)² |
//! | `active_user_count`  | 当該ステップで実際に行動した active エージェント数 |
//! | `propagation_reach`  | 情報がネットワーク上で到達したユニークノード数 |
//! | `cascade_size`       | 1 投稿が誘発したリポスト/コメント総数 (root ごと) |
//! | `cascade_max_breadth`| カスケード木の最大幅 |
//! | `herd_disagree_rate` | down-treat 群でのエージェント追随率 (簡易版) |

use std::collections::BTreeMap;

use serde::Serialize;
use socsim_core::AgentId;

use crate::world::{AgentProfile, Post};

/// 極化指数 P = (1/N)·Σ(o_a - ō)²．意見分散が大きいほど極化．
pub fn polarization_index(agents: &BTreeMap<AgentId, AgentProfile>) -> f64 {
    let n = agents.len();
    if n == 0 {
        return 0.0;
    }
    let mean: f64 = agents.values().map(|a| a.opinion).sum::<f64>() / n as f64;
    agents
        .values()
        .map(|a| (a.opinion - mean).powi(2))
        .sum::<f64>()
        / n as f64
}

/// 意見の標準偏差 (多様性の代理)．
pub fn opinion_std(agents: &BTreeMap<AgentId, AgentProfile>) -> f64 {
    polarization_index(agents).sqrt()
}

/// 各 root 投稿のカスケード規模 (root を含むカスケード内の投稿総数) を返す．
///
/// `posts[i].root` は所属するカスケードの起点投稿インデックスを指す (新規投稿は
/// 自身を指し，リポストは元投稿の root を継承する)．戻り値は (root_index, size)．
pub fn cascade_sizes(posts: &[Post]) -> Vec<(usize, usize)> {
    let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
    for post in posts {
        *sizes.entry(post.root).or_insert(0) += 1;
    }
    sizes.into_iter().collect()
}

/// 全カスケードの最大規模．カスケードが無ければ 0．
pub fn max_cascade_size(posts: &[Post]) -> usize {
    cascade_sizes(posts)
        .into_iter()
        .map(|(_, s)| s)
        .max()
        .unwrap_or(0)
}

/// カスケード木の最大幅 (同一 root・同一タイムステップに属する投稿数の最大値)．
pub fn cascade_max_breadth(posts: &[Post]) -> usize {
    let mut by_level: BTreeMap<(usize, u64), usize> = BTreeMap::new();
    for post in posts {
        *by_level.entry((post.root, post.ts)).or_insert(0) += 1;
    }
    by_level.into_values().max().unwrap_or(0)
}

/// 情報がネットワーク上で到達したユニークノード数の代理．
///
/// 「投稿の著者」と「いいね/リポストを行ったことで関与したノード」を含む，投稿
/// ストアに現れたユニーク著者数で近似する．
pub fn propagation_reach(posts: &[Post]) -> usize {
    let authors: std::collections::BTreeSet<AgentId> = posts.iter().map(|p| p.author).collect();
    authors.len()
}

/// 1 タイムステップ分の集団指標 (metrics.csv の long-format 行へ展開する元)．
#[derive(Debug, Clone, Serialize)]
pub struct StepMetrics {
    /// タイムステップ t．
    pub t: usize,
    /// 極化指数 P．
    pub polarization_index: f64,
    /// 意見標準偏差 (多様性代理)．
    pub opinion_std: f64,
    /// active-user 数 (当該ステップで行動したエージェント)．
    pub active_user_count: usize,
    /// 伝播到達ユニークノード数 (累積)．
    pub propagation_reach: usize,
    /// 最大カスケード規模 (累積)．
    pub cascade_size_max: usize,
    /// 最大カスケード幅 (累積)．
    pub cascade_max_breadth: usize,
    /// 投稿総数 (累積)．
    pub n_posts: usize,
    /// down-treat 群追随率 (群衆効果代理; 0 .. 1)．
    pub herd_disagree_rate: f64,
}

impl StepMetrics {
    /// 世界状態の現スナップショットから集団指標を計算する．
    pub fn compute(
        agents: &BTreeMap<AgentId, AgentProfile>,
        posts: &[Post],
        active_user_count: usize,
        herd_disagree_rate: f64,
        t: usize,
    ) -> Self {
        StepMetrics {
            t,
            polarization_index: polarization_index(agents),
            opinion_std: opinion_std(agents),
            active_user_count,
            propagation_reach: propagation_reach(posts),
            cascade_size_max: max_cascade_size(posts),
            cascade_max_breadth: cascade_max_breadth(posts),
            n_posts: posts.len(),
            herd_disagree_rate,
        }
    }
}

/// metrics.csv の long-format 1 行 (step, metric, value)．
#[derive(Debug, Clone, Serialize)]
pub struct MetricRow {
    /// タイムステップ t．
    pub t: usize,
    /// 指標名．
    pub metric: String,
    /// 値．
    pub value: f64,
}

impl StepMetrics {
    /// long-format 行の列へ展開する．
    pub fn to_rows(&self) -> Vec<MetricRow> {
        let pairs: [(&str, f64); 8] = [
            ("polarization_index", self.polarization_index),
            ("opinion_std", self.opinion_std),
            ("active_user_count", self.active_user_count as f64),
            ("propagation_reach", self.propagation_reach as f64),
            ("cascade_size_max", self.cascade_size_max as f64),
            ("cascade_max_breadth", self.cascade_max_breadth as f64),
            ("n_posts", self.n_posts as f64),
            ("herd_disagree_rate", self.herd_disagree_rate),
        ];
        pairs
            .iter()
            .map(|&(name, v)| MetricRow {
                t: self.t,
                metric: name.to_string(),
                value: v,
            })
            .collect()
    }
}

/// cascades.csv の 1 行 (root 投稿ごとのカスケード規模)．
#[derive(Debug, Clone, Serialize)]
pub struct CascadeRow {
    /// 起点投稿インデックス．
    pub root_post: usize,
    /// 起点投稿の著者．
    pub author: u64,
    /// カスケード規模 (root を含む所属投稿数)．
    pub size: usize,
}

/// 投稿ストアから cascades.csv 行を組み立てる．
pub fn cascade_rows(posts: &[Post]) -> Vec<CascadeRow> {
    cascade_sizes(posts)
        .into_iter()
        .map(|(root, size)| CascadeRow {
            root_post: root,
            author: posts.get(root).map(|p| p.author.0).unwrap_or(0),
            size,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::ACTIVITY_DIM;

    fn agent(opinion: f64) -> AgentProfile {
        AgentProfile {
            name: "a".into(),
            bio: "b".into(),
            activity_prob: [0.5; ACTIVITY_DIM],
            memory: Vec::new(),
            opinion,
        }
    }

    #[test]
    fn polarization_zero_when_all_equal() {
        let mut agents = BTreeMap::new();
        for i in 0..4u64 {
            agents.insert(AgentId(i), agent(0.5));
        }
        assert!(polarization_index(&agents).abs() < 1e-12);
    }

    #[test]
    fn polarization_grows_with_spread() {
        let mut tight = BTreeMap::new();
        let mut wide = BTreeMap::new();
        for i in 0..4u64 {
            tight.insert(AgentId(i), agent(if i % 2 == 0 { 0.1 } else { -0.1 }));
            wide.insert(AgentId(i), agent(if i % 2 == 0 { 1.0 } else { -1.0 }));
        }
        assert!(polarization_index(&wide) > polarization_index(&tight));
    }

    #[test]
    fn cascade_size_counts_root_group() {
        let posts = vec![
            Post::new(AgentId(0), "root".into(), 0, 0, 0.0),
            Post::new(AgentId(1), "repost".into(), 1, 0, 0.0),
            Post::new(AgentId(2), "repost".into(), 1, 0, 0.0),
            Post::new(AgentId(3), "other root".into(), 0, 3, 0.0),
        ];
        assert_eq!(max_cascade_size(&posts), 3);
        assert_eq!(cascade_max_breadth(&posts), 2); // root 0, ts 1 → 2 reposts
    }
}

//! socsim フレームワーク上の OASIS (Yang et al. 2024) の世界状態．
//!
//! エージェントは移動する空間主体ではなく，**動的ソーシャルグラフ上のノード**で
//! あり，投稿ストアと推薦バッファを共有する．したがって空間プリミティブ
//! (`socsim-grid`) は使わず，網プリミティブ [`socsim_net::SocialNetwork`] を採用
//! する．論文の大規模ユーザー生成は Barabási–Albert スケールフリーネットワークに
//! 基づくため，`SocialNetwork::barabasi_albert(ids, m, rng)` で初期化する．
//!
//! # フォロー方向の規約
//!
//! 無向 BA グラフの辺 `A — B` は相互フォロー (X の双方向可視性近似) として扱う．
//! `follow` アクションは新しいフォロー辺を動的に追加する．B の投稿は B の隣接
//! ノード (= フォロワ; `neighbors(B)`) のフィード候補に入る．
//!
//! `#[derive(Clone)]` でスナップショット (save/resume) と比較実験に対応する．
//! `agent_ids()` は `BTreeMap` のソート済みキーを返し決定論を保証する (socsim コア層)．

use std::collections::BTreeMap;

use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

use crate::config::{Platform, RecSysConfig};

/// 軽量埋め込みの次元数 (TwHIN-BERT の代替; ハッシュ化 bag-of-words)．
pub const EMBED_DIM: usize = 32;

/// 1 日 24 時間に対応する活動確率の次元数．
pub const ACTIVITY_DIM: usize = 24;

// --------------------------------------------------------------------------- //
// 軽量埋め込み (TwHIN-BERT の代替)
// --------------------------------------------------------------------------- //

/// テキストを決定論的な [`EMBED_DIM`] 次元の特徴ベクトルへ写す (ハッシュ化
/// bag-of-words)．TwHIN-BERT のスタンドインであり，興味マッチ (コサイン類似度)
/// の入力となる．同一テキストは必ず同一ベクトルになるため，socsim コアの bit
/// 決定論を壊さない．
pub fn embed(text: &str) -> [f64; EMBED_DIM] {
    let mut v = [0.0f64; EMBED_DIM];
    for token in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let lower = token.to_ascii_lowercase();
        // FNV-1a でトークンを安定ハッシュし，対応する次元のカウントを増やす．
        let mut h: u64 = 0xcbf29ce484222325;
        for b in lower.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let idx = (h % EMBED_DIM as u64) as usize;
        v[idx] += 1.0;
    }
    v
}

/// 2 つの埋め込みのコサイン類似度 ∈ [-1, 1]．いずれかがゼロベクトルなら 0．
pub fn cosine(a: &[f64; EMBED_DIM], b: &[f64; EMBED_DIM]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..EMBED_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// --------------------------------------------------------------------------- //
// エージェントプロフィール
// --------------------------------------------------------------------------- //

/// 1 エージェントの個別状態 (固定ノード; 移動しない)．
#[derive(Clone, Debug)]
pub struct AgentProfile {
    /// 表示名．
    pub name: String,
    /// 自己記述 (興味埋め込みの元になる; プロンプト文脈にも使う)．
    pub bio: String,
    /// 24 次元の時間活動確率 (時刻 h での活性化傾向)．
    pub activity_prob: [f64; ACTIVITY_DIM],
    /// メモリ (過去に見た/行った内容の短い記録)．
    pub memory: Vec<String>,
    /// 意見値 ∈ [-1, 1] (極化指標の元)．
    pub opinion: f64,
}

impl AgentProfile {
    /// 自己記述 (bio) の興味埋め込みを返す．
    pub fn interest_embedding(&self) -> [f64; EMBED_DIM] {
        embed(&self.bio)
    }
}

// --------------------------------------------------------------------------- //
// 投稿
// --------------------------------------------------------------------------- //

/// 投稿/コンテンツストアの 1 要素 (Environment Server 相当)．
#[derive(Clone, Debug)]
pub struct Post {
    /// 投稿者．
    pub author: AgentId,
    /// 本文．
    pub content: String,
    /// 内容埋め込み (興味マッチ用; init/生成時に固定)．
    pub content_vec: [f64; EMBED_DIM],
    /// 投稿時刻 (タイムステップ t; ホットスコア / recency 用)．
    pub ts: u64,
    /// アップボート (いいね) 数．
    pub upvotes: u64,
    /// ダウンボート数．
    pub downvotes: u64,
    /// リポスト数．
    pub reposts: u64,
    /// この投稿のカスケード起点となった元投稿のインデックス (リポストで継承)．
    pub root: usize,
    /// この投稿の意見符号 (極化追跡: 著者の意見をスナップショット)．
    pub opinion: f64,
}

impl Post {
    /// 著者・本文・時刻から新規投稿を作る (root は自身を指す)．
    pub fn new(author: AgentId, content: String, ts: u64, root: usize, opinion: f64) -> Self {
        let content_vec = embed(&content);
        Post {
            author,
            content,
            content_vec,
            ts,
            upvotes: 0,
            downvotes: 0,
            reposts: 0,
            root,
            opinion,
        }
    }
}

// --------------------------------------------------------------------------- //
// 世界状態
// --------------------------------------------------------------------------- //

/// OASIS の世界状態．
#[derive(Clone)]
pub struct OasisWorld {
    /// シミュレーションクロック．
    pub clock: SimClock,
    /// エージェント集合 (ソート済みキー; プロフィール + 記憶 + 活動確率 + 意見)．
    pub agents: BTreeMap<AgentId, AgentProfile>,
    /// 動的フォロー/ソーシャルグラフ (BA 初期化; follow アクションで辺追加)．
    pub network: SocialNetwork,
    /// 投稿/コンテンツストア (Environment Server 相当)．
    pub posts: Vec<Post>,
    /// エージェントごとの推薦バッファ (RecSys 出力フィード; post インデックス列)．
    pub feeds: BTreeMap<AgentId, Vec<usize>>,
    /// プラットフォーム種別 (X | Reddit)．
    pub platform: Platform,
    /// RecSys 設定 (k_in, k_out, hot_score 基準時刻 t0 など)．
    pub recsys: RecSysConfig,
}

impl OasisWorld {
    /// 構成済みフィールドから世界状態を組み立てる (網生成・初期化は
    /// [`crate::simulation::init_world`])．
    pub fn new(
        network: SocialNetwork,
        agents: BTreeMap<AgentId, AgentProfile>,
        platform: Platform,
        recsys: RecSysConfig,
        timesteps: u64,
    ) -> Self {
        let feeds = agents.keys().map(|&id| (id, Vec::new())).collect();
        OasisWorld {
            clock: SimClock::new(timesteps),
            agents,
            network,
            posts: Vec::new(),
            feeds,
            platform,
            recsys,
        }
    }

    /// エージェント数 N．
    pub fn n(&self) -> usize {
        self.agents.len()
    }

    /// 集団の平均意見 ō．
    pub fn mean_opinion(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.agents.values().map(|a| a.opinion).sum();
        sum / self.agents.len() as f64
    }
}

impl WorldState for OasisWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーはソート済み．契約 (sorted) を明示する．
        self.agents.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_deterministic() {
        assert_eq!(embed("hello world"), embed("hello world"));
        assert_ne!(embed("hello"), embed("world"));
    }

    #[test]
    fn cosine_identical_is_one() {
        let e = embed("nuclear energy policy debate");
        assert!((cosine(&e, &e) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        let z = [0.0; EMBED_DIM];
        let e = embed("anything");
        assert_eq!(cosine(&z, &e), 0.0);
    }
}

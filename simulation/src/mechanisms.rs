//! socsim フレームワーク上の OASIS メカニズム (6 機構 × 6 フェーズ)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 活性化・推薦・情報伝播・指標を `ctx.rng` (ChaCha20) とグラフ構造で行い，上層
//! (非決定的 LLM レイヤ) は [`AgentActionMechanism`] の `Decision` フェーズに
//! **のみ** 閉じ込める．オピニオンリーダー (高次数ノード) だけが LLM を呼び，
//! 周辺エージェントは確率的な簡易ポリシーで近似する (スケーラビリティ節)．
//!
//! # Mechanism × Phase
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`ActivationMechanism`]          | PreStep     | 24 次元活動確率 × activation_rate で active 集合を確定 |
//! | [`FeedRecommendationMechanism`]  | Environment | RecSys: 各 active エージェントのフィードを構築 (決定論的) |
//! | [`AgentActionMechanism`]         | Decision    | **LLM レイヤ**: leader は LLM, peripheral は簡易ポリシーで行動選択 |
//! | [`InfoPropagationMechanism`]     | Interaction | 選択アクションを投稿ストア・ソーシャルグラフへ反映 |
//! | [`MetricsMechanism`]             | Reward      | 集団指標を集計し scratch / recorder へ記録 |
//! | [`PostStepMechanism`]            | PostStep    | 記憶更新・収束判定 (連続ゼロアクションで request_stop) |

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::config::LlmSettings;
use crate::llm::{llm_config, OasisClient};
use crate::parse::{parse_action, ActionDecision, ActionKind};
use crate::prompts::action_prompt;
use crate::recsys::build_feed;
use crate::world::{embed, OasisWorld, Post};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<OasisClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 LLM 呼び出し予算カウンタ (run 全体で残数を管理)．
pub type SharedBudget = Rc<RefCell<usize>>;

/// scratch キー: 当該ステップの active エージェント集合．
const SCRATCH_ACTIVE: &str = "active_agents";
/// scratch キー: 当該ステップで決定された行動 (Decision → Interaction)．
const SCRATCH_ACTIONS: &str = "agent_actions";
/// scratch キー: 当該ステップで実際に行動した (active な) エージェント数．
const SCRATCH_ACTION_COUNT: &str = "action_count";
/// scratch キー: leader 集合 (高次数ノード)．
const SCRATCH_LEADERS: &str = "leaders";

// --------------------------------------------------------------------------- //
// ActivationMechanism (PreStep)
// --------------------------------------------------------------------------- //

/// 活性化 (`PreStep`) — Time Engine．
///
/// 各エージェントの 24 次元活動確率のうち «現タイムステップに対応する時刻» の値に
/// `activation_rate` を乗じた確率で active 集合に入れる．active 集合は scratch
/// (`SCRATCH_ACTIVE`) へ書き，その集合のみが当該ステップで行動する (同期更新)．
pub struct ActivationMechanism {
    /// 活性化サブサンプリング率 ∈ [0,1]．
    activation_rate: f64,
    /// leader 集合 (高次数ノード; init で計算し毎 PreStep で scratch へ複写)．
    leaders: Vec<AgentId>,
}

impl ActivationMechanism {
    /// サブサンプリング率と leader 集合から作る．
    pub fn new(activation_rate: f64, leaders: Vec<AgentId>) -> Self {
        ActivationMechanism {
            activation_rate: activation_rate.clamp(0.0, 1.0),
            leaders,
        }
    }
}

impl Mechanism<OasisWorld> for ActivationMechanism {
    fn name(&self) -> &str {
        "activation"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PreStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let hour = (ctx.clock.t() % crate::world::ACTIVITY_DIM as u64) as usize;
        let mut active: Vec<AgentId> = Vec::new();
        // agent_order (scheduler が決めた順) を尊重して走査する (決定論的)．
        for &id in ctx.agent_order {
            let p = match ctx.world.agents.get(&id) {
                Some(a) => a.activity_prob[hour] * self.activation_rate,
                None => continue,
            };
            if ctx.rng.gen_bool(p.clamp(0.0, 1.0)) {
                active.push(id);
            }
        }
        ctx.scratch.insert(SCRATCH_ACTIVE, active);
        // leader 集合を scratch へ複写する (Decision フェーズが参照する)．
        ctx.scratch.insert(SCRATCH_LEADERS, self.leaders.clone());
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// FeedRecommendationMechanism (Environment)
// --------------------------------------------------------------------------- //

/// フィード推薦 (`Environment`) — RecSys．
///
/// 各 active エージェントのフィードを [`crate::recsys::build_feed`] で構築し
/// `world.feeds` へ書き込む．**決定論的**で LLM 非依存．
pub struct FeedRecommendationMechanism;

impl Mechanism<OasisWorld> for FeedRecommendationMechanism {
    fn name(&self) -> &str {
        "feed_recommendation"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let active: Vec<AgentId> = ctx
            .scratch
            .get::<Vec<AgentId>>(SCRATCH_ACTIVE)
            .cloned()
            .unwrap_or_default();
        for id in active {
            if let Some(profile) = ctx.world.agents.get(&id).cloned() {
                let feed = build_feed(ctx.world, id, &profile);
                ctx.world.feeds.insert(id, feed);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// AgentActionMechanism (Decision) — LLM レイヤ
// --------------------------------------------------------------------------- //

/// 行動選択 (`Decision`) — **二層境界の唯一の LLM 呼び出し点**．
///
/// active エージェントのうち leader (高次数ノード) は LLM を CoT で呼び，
/// 行動 (post/repost/like/follow/none) を決める．peripheral エージェントは
/// 確率的な簡易ポリシー (like/repost ∝ 活動確率，意見はピア平均へドリフト) で
/// 近似する．LLM 予算 (`SharedBudget`) を超えたら leader も簡易ポリシーへ落ちる．
///
/// 決定は scratch (`SCRATCH_ACTIONS`) へ書き，[`InfoPropagationMechanism`] が
/// world へ反映する．
pub struct AgentActionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    budget: SharedBudget,
    settings: LlmSettings,
}

impl AgentActionMechanism {
    /// 共有クライアント・メタデータ・予算・LLM 設定から作る．
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        budget: SharedBudget,
        settings: LlmSettings,
    ) -> Self {
        AgentActionMechanism {
            client,
            metadata,
            budget,
            settings,
        }
    }

    /// peripheral エージェントの確率的簡易ポリシー (LLM 非依存)．
    ///
    /// フィードがあれば活動確率に比例して like / repost を選び，無ければ post か
    /// none を確率的に選ぶ．意見はフィードに現れた投稿の意見平均へ僅かにドリフト
    /// する (この drift は Interaction フェーズで適用; ここでは行動のみ決める)．
    fn cheap_policy(
        world: &OasisWorld,
        agent: AgentId,
        rng: &mut socsim_core::SimRng,
    ) -> ActionDecision {
        let profile = match world.agents.get(&agent) {
            Some(p) => p,
            None => {
                return ActionDecision {
                    kind: ActionKind::None,
                    target: None,
                    content: None,
                }
            }
        };
        let hour = (world.clock.t() % crate::world::ACTIVITY_DIM as u64) as usize;
        let act = profile.activity_prob[hour];
        let feed = world.feeds.get(&agent).cloned().unwrap_or_default();

        if !feed.is_empty() && rng.gen_bool((act * 0.8).clamp(0.0, 1.0)) {
            // フィードがある → like か repost．先頭の推薦投稿を対象に．
            let target = feed[0];
            let kind = if rng.gen_bool(0.5) {
                ActionKind::Repost
            } else {
                ActionKind::Like
            };
            ActionDecision {
                kind,
                target: Some(target),
                content: None,
            }
        } else if rng.gen_bool((act * 0.3).clamp(0.0, 1.0)) {
            // たまに新規投稿．
            ActionDecision {
                kind: ActionKind::Post,
                target: None,
                content: Some(format!("Agent {} shares a thought.", agent.0)),
            }
        } else {
            ActionDecision {
                kind: ActionKind::None,
                target: None,
                content: None,
            }
        }
    }
}

impl Mechanism<OasisWorld> for AgentActionMechanism {
    fn name(&self) -> &str {
        "agent_action"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let active: Vec<AgentId> = ctx
            .scratch
            .get::<Vec<AgentId>>(SCRATCH_ACTIVE)
            .cloned()
            .unwrap_or_default();
        let leaders: std::collections::BTreeSet<AgentId> = ctx
            .scratch
            .get::<Vec<AgentId>>(SCRATCH_LEADERS)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let mut decisions: BTreeMap<AgentId, ActionDecision> = BTreeMap::new();

        for id in active {
            let is_leader = leaders.contains(&id);
            let budget_left = *self.budget.borrow() > 0;

            let decision = if is_leader && budget_left {
                // --- LLM 呼び出し (leader のみ; 予算内) ---
                let profile = match ctx.world.agents.get(&id) {
                    Some(p) => p.clone(),
                    None => continue,
                };
                let feed_idx = ctx.world.feeds.get(&id).cloned().unwrap_or_default();
                let feed: Vec<(usize, &Post)> = feed_idx
                    .iter()
                    .filter_map(|&i| ctx.world.posts.get(i).map(|p| (i, p)))
                    .collect();
                let prompt = action_prompt(&profile, &feed);
                let text = {
                    let mut client = self.client.borrow_mut();
                    let resp = client
                        .complete(&prompt, &llm_config(&self.settings))
                        .map_err(|e| {
                            SocsimError::Mechanism(format!("agent action LLM call failed: {e}"))
                        })?;
                    self.metadata.borrow_mut().record(resp.metadata.clone());
                    resp.text
                };
                {
                    let mut b = self.budget.borrow_mut();
                    *b = b.saturating_sub(1);
                }
                parse_action(&text)
            } else {
                // --- 簡易ポリシー (peripheral，または予算切れ leader) ---
                Self::cheap_policy(ctx.world, id, ctx.rng)
            };
            decisions.insert(id, decision);
        }

        ctx.scratch.insert(SCRATCH_ACTIONS, decisions);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// InfoPropagationMechanism (Interaction)
// --------------------------------------------------------------------------- //

/// 情報伝播 (`Interaction`)．
///
/// `Decision` の各行動を投稿ストア・ソーシャルグラフへ反映する:
/// - **post**: 新規投稿を `world.posts` へ追加 (root=自身)．意見をスナップショット．
/// - **repost**: 対象投稿の root を継承した新規投稿を追加し，元投稿の reposts++．
///   リポスト側の意見は元投稿の意見へ僅かにドリフト (情報の影響)．
/// - **like**: 対象投稿の upvotes++．意見が元投稿へ僅かにドリフト．
/// - **follow**: 対象投稿の著者へフォロー辺を追加 (動的グラフ更新)．
/// - **none**: 何もしない．
///
/// 行動した (none 以外) エージェント数を `SCRATCH_ACTION_COUNT` に書く．
pub struct InfoPropagationMechanism {
    /// like/repost 時の意見ドリフト係数．
    drift: f64,
}

impl InfoPropagationMechanism {
    /// ドリフト係数から作る (既定 0.1)．
    pub fn new(drift: f64) -> Self {
        InfoPropagationMechanism { drift }
    }
}

impl Default for InfoPropagationMechanism {
    fn default() -> Self {
        InfoPropagationMechanism::new(0.1)
    }
}

impl Mechanism<OasisWorld> for InfoPropagationMechanism {
    fn name(&self) -> &str {
        "info_propagation"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let decisions: BTreeMap<AgentId, ActionDecision> = ctx
            .scratch
            .get::<BTreeMap<AgentId, ActionDecision>>(SCRATCH_ACTIONS)
            .cloned()
            .unwrap_or_default();

        let now = ctx.clock.t();
        let mut action_count = 0usize;

        // 決定論のため AgentId 昇順に処理する (BTreeMap は既にソート済み)．
        for (&id, decision) in decisions.iter() {
            match decision.kind {
                ActionKind::Post => {
                    let opinion = ctx.world.agents.get(&id).map(|a| a.opinion).unwrap_or(0.0);
                    let content = decision
                        .content
                        .clone()
                        .unwrap_or_else(|| format!("Agent {} posts.", id.0));
                    let root = ctx.world.posts.len();
                    ctx.world
                        .posts
                        .push(Post::new(id, content, now, root, opinion));
                    action_count += 1;
                }
                ActionKind::Repost => {
                    if let Some(target) = decision.target {
                        if let Some(src) = ctx.world.posts.get(target) {
                            let root = src.root;
                            let src_opinion = src.opinion;
                            let content = src.content.clone();
                            // 新規リポスト投稿を追加 (root 継承)．
                            let opinion =
                                ctx.world.agents.get(&id).map(|a| a.opinion).unwrap_or(0.0);
                            let mut post = Post::new(id, content, now, root, opinion);
                            post.content_vec = embed(&post.content);
                            ctx.world.posts.push(post);
                            if let Some(src_mut) = ctx.world.posts.get_mut(target) {
                                src_mut.reposts += 1;
                            }
                            // 意見ドリフト (元投稿の意見へ近づく)．
                            if let Some(a) = ctx.world.agents.get_mut(&id) {
                                a.opinion += self.drift * (src_opinion - a.opinion);
                            }
                            action_count += 1;
                        }
                    }
                }
                ActionKind::Like => {
                    if let Some(target) = decision.target {
                        let src_opinion = ctx.world.posts.get(target).map(|p| p.opinion);
                        if let Some(p) = ctx.world.posts.get_mut(target) {
                            p.upvotes += 1;
                        }
                        if let (Some(src_op), Some(a)) =
                            (src_opinion, ctx.world.agents.get_mut(&id))
                        {
                            a.opinion += 0.5 * self.drift * (src_op - a.opinion);
                        }
                        action_count += 1;
                    }
                }
                ActionKind::Follow => {
                    if let Some(target) = decision.target {
                        if let Some(author) = ctx.world.posts.get(target).map(|p| p.author) {
                            if author != id {
                                ctx.world.network.add_edge(id, author);
                                action_count += 1;
                            }
                        }
                    }
                }
                ActionKind::None => {}
            }
        }

        ctx.scratch.insert(SCRATCH_ACTION_COUNT, action_count);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// MetricsMechanism (Reward)
// --------------------------------------------------------------------------- //

/// 指標集計 (`Reward`)．
///
/// active-user 数 (= 行動数) を scratch から取り出し，recorder へ記録する．
/// 集団指標の CSV 化は run ドライバが `run_observed` のスナップショットで行うため，
/// ここでは recorder への記録のみ行う (拡張点)．
pub struct MetricsMechanism;

impl Mechanism<OasisWorld> for MetricsMechanism {
    fn name(&self) -> &str {
        "metrics"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let action_count = *ctx.scratch.get::<usize>(SCRATCH_ACTION_COUNT).unwrap_or(&0);
        let p = crate::metrics::polarization_index(&ctx.world.agents);
        let t = ctx.clock.t();
        ctx.recorder
            .record_metric(t, "active_user_count", action_count as f64);
        ctx.recorder.record_metric(t, "polarization_index", p);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// PostStepMechanism (PostStep)
// --------------------------------------------------------------------------- //

/// 記憶更新 + 収束判定 (`PostStep`)．
///
/// active エージェントの記憶を自分のフィードの先頭内容で更新し，当該ステップの
/// 行動数が 0 なら連続ゼロカウンタを進める．連続ゼロが `patience` に達したら
/// `request_stop` で収束停止を要求する．
pub struct PostStepMechanism {
    /// 連続ゼロアクション許容ステップ数 (これに達したら停止)．
    patience: usize,
    /// 連続ゼロアクションカウンタ．
    zero_streak: RefCell<usize>,
}

impl PostStepMechanism {
    /// patience から作る．
    pub fn new(patience: usize) -> Self {
        PostStepMechanism {
            patience: patience.max(1),
            zero_streak: RefCell::new(0),
        }
    }
}

impl Mechanism<OasisWorld> for PostStepMechanism {
    fn name(&self) -> &str {
        "post_step"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, OasisWorld>) -> Result<()> {
        let active: Vec<AgentId> = ctx
            .scratch
            .get::<Vec<AgentId>>(SCRATCH_ACTIVE)
            .cloned()
            .unwrap_or_default();

        // 記憶更新: active エージェントが見た先頭投稿内容を記憶に積む (上限 10)．
        for id in active {
            let head = ctx
                .world
                .feeds
                .get(&id)
                .and_then(|f| f.first().copied())
                .and_then(|i| ctx.world.posts.get(i).map(|p| p.content.clone()));
            if let (Some(content), Some(agent)) = (head, ctx.world.agents.get_mut(&id)) {
                agent.memory.push(content);
                if agent.memory.len() > 10 {
                    let overflow = agent.memory.len() - 10;
                    agent.memory.drain(0..overflow);
                }
            }
        }

        let action_count = *ctx.scratch.get::<usize>(SCRATCH_ACTION_COUNT).unwrap_or(&0);
        if action_count == 0 {
            *self.zero_streak.borrow_mut() += 1;
        } else {
            *self.zero_streak.borrow_mut() = 0;
        }
        if *self.zero_streak.borrow() >= self.patience {
            ctx.request_stop();
        }
        Ok(())
    }
}

//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` のどのサブコマンドでも
//! 同一なので，ここ 1 箇所で組み立てる．ステップごとの集団指標の落とし方，カスケード
//! 表の置き場，スイープの試行 1 本ぶんの終端行，`reproduce` の条件セルとアンカーの
//! 書き方もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::metrics::{CascadeRow, StepMetrics};
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名 (`oasis`) と揃える．
pub const EXPERIMENT: &str = "oasis";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "yang2024";
/// 分野．BA 網生成・プロフィール割当・活性化の draw に乱数を引くので `simulation`
/// (= `master_seed` が必須)．
///
/// リーダーは LLM で駆動されるが `llm-safety` ではない — 測っているのはモデルの
/// 安全性ではなく，ソーシャルメディア上の情報拡散・極化・群衆効果だからである．
/// LLM 側の同一性は `llm` ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// OASIS の刻みは離散タイムステップ (論文では 1 tick ≒ 3 分) そのもので，runvault の
/// 語彙では `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．集団指標はどれも母集団全体の集約なので `run`．
const SCOPE: &str = "run";

/// カスケード 1 本のイベント種別．コア語彙に無いので `x.<repo_id>.<name>` を使う．
pub const CASCADE_EVENT: &str = "x.yang2024.cascade";
/// アンカー判定のイベント種別．
pub const ANCHOR_EVENT: &str = "x.yang2024.anchor";

/// この再現実験が対象としている論文．
///
/// どのサブコマンドも同じ主張を対象とする — `sweep` は N × 活性化率の感度を見るが，
/// その存在理由は «創発がどの規模で立ち上がるか» を問うことなので，同じ target に
/// 属する．論文の特定の図表ではなく主張の再現を狙うので `Target::claim` を使う．
pub fn replication() -> Replication {
    Work::arxiv("2411.11581")
        .title("OASIS: Open Agent Social Interaction Simulations with One Million Agents")
        .year(2024)
        .source_version("arxiv-v1")
        .target(Target::claim(
            "emergent-social-phenomena",
            "Information cascades, group polarization and herd effects emerge from recommender-mediated LLM-agent interaction",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/yang2024/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (旧 `llm_meta.json` の `provider` と同じ判定)．推測しているのは分類だけで，
/// 値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.1` のような動くエイリアスであることが多い．
/// socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.contains("11434") || endpoint.contains("ollama") {
        "ollama"
    } else if endpoint.contains("mock") {
        "mock"
    } else {
        "openai"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // リーダーのプロンプトは profile / memory / フィードから毎回組み立てられ，
        // 固定の system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// シミュレーション 1 本
// ---------------------------------------------------------------------------

/// シミュレーション 1 本ぶんの記録 (`run` サブコマンド用)．
///
/// ステップごとの 8 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `converged` / `final_step` / LLM 呼び出しの内訳を書く．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    log_history(run, None, &result.metrics_history);
    run.log_metrics(
        SCOPE,
        &[
            ("converged", if result.converged { 1.0 } else { 0.0 }),
            ("final_step", result.final_step as f64),
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// メトリクス履歴をステップごとの指標として書く．
///
/// `prefix` は条件ラベル (`reproduce` は 3 つの推薦器条件を 1 本の run に書くので，
/// `(step, scope, name)` が衝突しないよう名前で条件を分ける)．`run` サブコマンドは
/// 条件が 1 つしかないので接頭辞なし．
pub fn log_history(run: &mut Run, prefix: Option<&str>, history: &[StepMetrics]) {
    for m in history {
        log_step(run, prefix, m);
    }
}

/// [`StepMetrics`] の 8 フィールドを 1 ステップぶんまとめて書く．
///
/// 個数の指標 (`active_user_count` / `propagation_reach` / `cascade_size_max` /
/// `cascade_max_breadth` / `n_posts`) はカテゴリではなく «そのステップの数» なので
/// そのまま指標にする．
fn log_step(run: &mut Run, prefix: Option<&str>, m: &StepMetrics) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    run.log_metrics_at(
        m.t as u64,
        T_UNIT,
        SCOPE,
        &[
            (name("polarization_index").as_str(), m.polarization_index),
            (name("opinion_std").as_str(), m.opinion_std),
            (name("active_user_count").as_str(), m.active_user_count as f64),
            (name("propagation_reach").as_str(), m.propagation_reach as f64),
            (name("cascade_size_max").as_str(), m.cascade_size_max as f64),
            (
                name("cascade_max_breadth").as_str(),
                m.cascade_max_breadth as f64,
            ),
            (name("n_posts").as_str(), m.n_posts as f64),
            (name("herd_disagree_rate").as_str(), m.herd_disagree_rate),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// 接頭辞付きの run スコープ指標をまとめて書く．
///
/// `reproduce` の条件セルに使う．接頭辞の付け方は [`log_history`] と同じ理由 —
/// 1 本の run に同居する条件を名前で分ける．
pub fn log_prefixed(run: &mut Run, prefix: &str, values: &[(&str, f64)]) {
    let named: Vec<(String, f64)> = values
        .iter()
        .map(|(name, v)| (format!("{prefix}_{name}"), *v))
        .collect();
    let pairs: Vec<(&str, f64)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &pairs)
        .unwrap_or_else(|e| panic!("{prefix} の run スコープ指標の記録に失敗: {e}"));
}

/// 接頭辞なしの run スコープ指標をまとめて書く．
pub fn log_scoped(run: &mut Run, values: &[(&str, f64)]) {
    run.log_metrics(SCOPE, values)
        .expect("run スコープの指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// カスケード表
// ---------------------------------------------------------------------------

/// 旧 `cascades.csv` の 1 行を `events.jsonl` へ書く．
///
/// この表は «カスケード 1 本» を行とし，root 投稿で識別される．時間軸を持たないので
/// `metrics.csv` には置けない — 全行が同じ主キー `(name, step=∅, step_unit=∅, scope)`
/// を名乗ってしまう．`observation` でもない (時刻を持たないので `t` を作る羽目になる)
/// ので，名前空間付きイベントにする．
pub fn log_cascades(run: &mut Run, rows: &[CascadeRow]) {
    for row in rows {
        run.log_event(CASCADE_EVENT, row)
            .unwrap_or_else(|e| panic!("カスケード {} の記録に失敗: {e}", row.root_post));
    }
}

// ---------------------------------------------------------------------------
// スイープの試行 (子 run の events.jsonl)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く観測行．
///
/// 予約キーだけを持つ．数はここには書かない — 試行の最終値は下の [`TerminalEvent`]
/// が正本なので，同じ数を 2 箇所に置かない．
///
/// `runvault verify --deep` は terminal の `unit_id` が observation にも現れ，
/// その最大 `t` が terminal の `t` と一致することを要求するので，観測した時刻を
/// 明示的に残す．
#[derive(Serialize)]
struct ObservationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// `events.jsonl` に書く終端行 (旧 `sweep_summary.csv` の 1 行に対応)．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは自由欄．
///
/// 派生シードを `seed` ではなく `trial_seed` と呼ぶのは，`runvault.read` の
/// `sweep_events_table` が条件パラメータの列をイベント列の上に書くからである．
/// 子 run の `parameters` は基点シードを `seed` という名前で持つので，同じ名前を
/// 使うと試行ごとのシードが黙って基点シードに潰される．
#[derive(Serialize)]
struct TerminalEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    trial_seed: u64,
    final_polarization_index: f64,
    final_opinion_std: f64,
    final_propagation_reach: usize,
    final_cascade_size_max: usize,
    cache_hit_rate: f64,
}

/// 試行 1 本を `terminal` イベントとして書く (観測時刻を 1 点添えて)．
///
/// 打ち切り (`censored`) の行は `t == budget` でなければならない．ドライバは
/// 新規アクション 0 が `convergence_patience` ステップ続いたら停止し，止まらなければ
/// `timesteps` まで回すので，収束しなかった試行は必ず上限に達している．この不変条件は
/// runvault が `log_event` の書き込み時に検査するので，ここでは二重に持たない．
///
/// `sweep` が見るのは各試行の最終ステップだけなので，観測時刻もそこ 1 点である．
pub fn log_trial(
    run: &mut Run,
    unit_id: &str,
    trial_seed: u64,
    budget: usize,
    result: &SimulationResult,
) {
    let last = result
        .metrics_history
        .last()
        .expect("metrics_history は t=0 を含む");

    run.log_event(
        "observation",
        &ObservationEvent {
            unit_id,
            t: result.final_step as u64,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の observation の記録に失敗: {e}"));

    let event = TerminalEvent {
        unit_id,
        t: result.final_step as u64,
        t_unit: T_UNIT,
        outcome: if result.converged {
            "converged"
        } else {
            "unconverged"
        },
        censored: !result.converged,
        budget: budget as u64,
        trial_seed,
        final_polarization_index: last.polarization_index,
        final_opinion_std: last.opinion_std,
        final_propagation_reach: last.propagation_reach,
        final_cascade_size_max: last.cascade_size_max,
        cache_hit_rate: result.metadata.cache_hit_rate(),
    };
    run.log_event("terminal", &event)
        .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// 条件 1 点ぶんの集約 (sweep の子 run)
// ---------------------------------------------------------------------------

/// 1 条件で回した試行 1 本の最終値．集約の材料になる．
pub struct TrialOutcome {
    /// 収束したか．
    pub converged: bool,
    /// 収束 (または打ち切り) したステップ．
    pub final_step: usize,
    /// 最終ステップの極化指数．
    pub polarization_index: f64,
    /// 最終ステップの意見標準偏差．
    pub opinion_std: f64,
    /// 最終ステップの伝播到達ノード数．
    pub propagation_reach: usize,
    /// 最終ステップの最大カスケード規模．
    pub cascade_size_max: usize,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終ステップから取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let last = result
            .metrics_history
            .last()
            .expect("metrics_history は t=0 を含む");
        TrialOutcome {
            converged: result.converged,
            final_step: result.final_step,
            polarization_index: last.polarization_index,
            opinion_std: last.opinion_std,
            propagation_reach: last.propagation_reach,
            cascade_size_max: last.cascade_size_max,
        }
    }
}

/// 1 条件 (N × 活性化率の 1 点) を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行ごとの
/// `polarization_index` を指標にすると (`run_uid`, `step`, `scope`, `name`) が重複
/// するので，散らばりが要る図は `events.jsonl` から組み直す．
pub fn log_condition_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;

    let n_converged = trials.iter().filter(|t| t.converged).count();
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("n_converged", n_converged as f64),
            ("mean_final_step", mean(&|t| t.final_step as f64)),
            (
                "mean_final_polarization_index",
                mean(&|t| t.polarization_index),
            ),
            ("mean_final_opinion_std", mean(&|t| t.opinion_std)),
            (
                "mean_final_propagation_reach",
                mean(&|t| t.propagation_reach as f64),
            ),
            (
                "mean_final_cascade_size_max",
                mean(&|t| t.cascade_size_max as f64),
            ),
        ],
    )
    .expect("条件 1 点の集約の記録に失敗");
}

// ---------------------------------------------------------------------------
// reproduce のアンカー
// ---------------------------------------------------------------------------

/// 判定の伴うイベント 1 行を書く．
///
/// PASS / off はカテゴリであって指標ではないので `events.jsonl` に置く．照合先の帯
/// (`target_lo` / `target_hi`) は論文が報告した数値ではなく，この再現実装が論文の
/// 定性記述から置いたアンカーなので，出典を要求する `reference.csv` には書かない —
/// 書くと論文の報告値と自前のアンカーが後から見分けられなくなる．
pub fn log_verdict<T: Serialize + ?Sized>(run: &mut Run, kind: &str, label: &str, payload: &T) {
    run.log_event(kind, payload)
        .unwrap_or_else(|e| panic!("{kind} の {label} の記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 派生シードのラベルに使う文字列ハッシュ (FNV-1a; explicit identity)．
pub fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// スイープの試行 1 本のシードを基点シードから決定的に派生させる．
///
/// `master_seed` として記録するのは `base` の方で，実際に各試行が使うシードは
/// これで作る．`(base, platform, n_agents, activation_rate, index)` が同じなら常に
/// 同じ値を返し，どれか 1 つでも違えば別の値になる — この性質が壊れると，記録した
/// `master_seed` から run を組み直せなくなる．
pub fn sweep_trial_seed(
    base: u64,
    platform: &str,
    n_agents: usize,
    activation_rate: f64,
    index: usize,
) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            label_hash(platform),
            n_agents as u64,
            (activation_rate * 1000.0) as u64,
            index as u64,
        ],
    )
}

/// `reproduce` の 1 セル内の試行 1 本のシードを派生させる．
pub fn repro_trial_seed(base: u64, platform: &str, recsys: &str, index: usize) -> u64 {
    socsim_core::derive_seed(
        base,
        &[label_hash(platform), label_hash(recsys), index as u64],
    )
}

#[cfg(test)]
mod tests {
    use super::{repro_trial_seed, sweep_trial_seed};

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(
            sweep_trial_seed(42, "x", 200, 0.3, 2),
            sweep_trial_seed(42, "x", 200, 0.3, 2)
        );
        assert_eq!(
            repro_trial_seed(42, "x", "interest", 1),
            repro_trial_seed(42, "x", "interest", 1)
        );
    }

    #[test]
    fn each_coordinate_changes_the_sweep_seed() {
        let base = sweep_trial_seed(42, "x", 200, 0.3, 0);
        assert_ne!(base, sweep_trial_seed(43, "x", 200, 0.3, 0), "base");
        assert_ne!(base, sweep_trial_seed(42, "reddit", 200, 0.3, 0), "platform");
        assert_ne!(base, sweep_trial_seed(42, "x", 1000, 0.3, 0), "n_agents");
        assert_ne!(base, sweep_trial_seed(42, "x", 200, 0.5, 0), "activation");
        assert_ne!(base, sweep_trial_seed(42, "x", 200, 0.3, 1), "index");
    }

    #[test]
    fn each_coordinate_changes_the_repro_seed() {
        let base = repro_trial_seed(42, "x", "interest", 0);
        assert_ne!(base, repro_trial_seed(43, "x", "interest", 0));
        assert_ne!(base, repro_trial_seed(42, "reddit", "interest", 0));
        assert_ne!(base, repro_trial_seed(42, "x", "hot-score", 0));
        assert_ne!(base, repro_trial_seed(42, "x", "interest", 1));
    }

    #[test]
    fn one_condition_gives_distinct_seeds_across_trials() {
        let seeds: std::collections::BTreeSet<u64> = (0..64)
            .map(|i| sweep_trial_seed(42, "x", 200, 0.3, i))
            .collect();
        assert_eq!(seeds.len(), 64, "同一条件の試行でシードが衝突した");
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは
    /// 過去の run と結果を比較できなくなっている．Cargo.lock が socsim の commit を
    /// 固定しているので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(sweep_trial_seed(42, "x", 200, 0.1, 0), 15_813_606_257_957_117_008);
        assert_eq!(repro_trial_seed(42, "x", "interest", 0), 6_074_958_179_555_378_911);
    }
}

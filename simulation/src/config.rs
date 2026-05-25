//! シミュレーション設定．
//!
//! Yang et al. (2024) "OASIS" のコアモデル (BA フォローグラフ上の LLM 駆動の
//! 行動選択 + 決定論的推薦器 + 情報伝播) と感度分析パラメータを保持する
//! [`Config`] と，その JSON シリアライズ表現を定義する．プラットフォーム種別・
//! 推薦器種別・LLM 設定などの列挙型もここに集約する．

use serde::Serialize;

// --------------------------------------------------------------------------- //
// プラットフォーム種別
// --------------------------------------------------------------------------- //

/// シミュレートするソーシャルメディアプラットフォーム．
///
/// OASIS は X (Twitter) と Reddit を模す．プラットフォームにより [`crate::recsys`]
/// の推薦規則が変わる: X は興味マッチ (コサイン類似度)，Reddit はホットスコア．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// X (Twitter) — 興味ベース推薦．
    X,
    /// Reddit — ホットスコアベース推薦．
    Reddit,
}

impl Platform {
    /// 短い識別ラベル (CLI / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            Platform::X => "x",
            Platform::Reddit => "reddit",
        }
    }
}

/// 文字列から [`Platform`] をパースする．
pub fn parse_platform(s: &str) -> Result<Platform, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "x" | "twitter" => Ok(Platform::X),
        "reddit" => Ok(Platform::Reddit),
        _ => Err(format!("不正なプラットフォーム: \"{}\" (x / reddit)", s)),
    }
}

// --------------------------------------------------------------------------- //
// 推薦器種別
// --------------------------------------------------------------------------- //

/// 推薦器 (RecSys) 種別．
///
/// `Interest` は X の興味マッチ (コサイン類似度)，`HotScore` は Reddit の
/// ホットスコアランキング，`None` はアブレーション (推薦器を使わずフォロー先の
/// 最新投稿のみをフィードに入れる; 拡散阻害の検証用)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecSysKind {
    /// 興味マッチ (X; コサイン類似度上位)．
    Interest,
    /// ホットスコア (Reddit)．
    HotScore,
    /// 推薦器なし (アブレーション; フォロー先の最新のみ)．
    None,
}

impl RecSysKind {
    /// 短い識別ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            RecSysKind::Interest => "interest",
            RecSysKind::HotScore => "hot-score",
            RecSysKind::None => "none",
        }
    }

    /// プラットフォームに対する既定の推薦器を返す．
    pub fn default_for(platform: Platform) -> Self {
        match platform {
            Platform::X => RecSysKind::Interest,
            Platform::Reddit => RecSysKind::HotScore,
        }
    }
}

/// 文字列から [`RecSysKind`] をパースする．
pub fn parse_recsys(s: &str) -> Result<RecSysKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "interest" | "x" => Ok(RecSysKind::Interest),
        "hot-score" | "hot_score" | "hotscore" | "reddit" => Ok(RecSysKind::HotScore),
        "none" | "off" | "ablation" => Ok(RecSysKind::None),
        _ => Err(format!(
            "不正な推薦器種別: \"{}\" (interest / hot-score / none)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// RecSys 設定
// --------------------------------------------------------------------------- //

/// 推薦器のチューニングパラメータ (各 active エージェントのフィード構築用)．
#[derive(Debug, Clone, Copy)]
pub struct RecSysConfig {
    /// 推薦器種別 (興味 / ホットスコア / なし)．
    pub kind: RecSysKind,
    /// in-network (フォロー先) から取り込む上位件数 k_in．
    pub k_in: usize,
    /// out-network (興味マッチ / ホットスコア) から取り込む上位件数 k_out．
    pub k_out: usize,
    /// Reddit ホットスコアの基準時刻 t0 (Unix epoch 秒; 論文の規約)．
    pub t0: f64,
}

impl Default for RecSysConfig {
    fn default() -> Self {
        RecSysConfig {
            kind: RecSysKind::Interest,
            k_in: 5,
            k_out: 5,
            t0: 1_134_028_003.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// プロバイダ優先順位は «Ollama 第一 → OpenAI フォールバック» 固定．モデル・
/// ホスト・API キーは環境変数で渡す (`OLLAMA_HOST` / `OLLAMA_MODEL` /
/// `OPENAI_API_KEY` / `OPENAI_MODEL`)．`temperature`/`seed` で擬似決定論化する．
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// 生成温度 (既定 0.0; 再現性のため)．
    pub temperature: f32,
    /// 生成シード (バックエンドへ渡す; Ollama は honour，OpenAI は best-effort)．
    pub seed: u64,
    /// プロンプト→応答キャッシュの保存先 (None なら in-memory)．
    pub cache_path: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        }
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
#[derive(Debug, Clone)]
pub struct Config {
    /// プラットフォーム (x / reddit)．
    pub platform: Platform,
    /// エージェント数 N (= ノード数)．
    pub n_agents: usize,
    /// オピニオンリーダー数 (= ネットワーク次数上位 N_leaders 体が LLM を呼ぶ)．
    pub n_leaders: usize,
    /// タイムステップ数 T (1 tick = 論文の 3 分相当)．
    pub timesteps: usize,
    /// 活性化サブサンプリング率 ∈ [0,1] (活動確率に乗算する)．
    pub activation_rate: f64,
    /// 1 実行あたりの最大 LLM 呼び出し数 (超過分は簡易ポリシーへフォールバック)．
    pub llm_budget: usize,
    /// BA の新規ノードあたりの結合数 m．
    pub ba_m: usize,
    /// 推薦器設定 (種別 / k_in / k_out / t0)．
    pub recsys: RecSysConfig,
    /// 収束判定: 新規アクション 0 が連続したら停止する連続ステップ数しきい値．
    pub convergence_patience: usize,
    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    /// 標準設定 (X, N=200, leaders=20, T=30, activation=0.3, budget=2000)．
    fn default() -> Self {
        Config {
            platform: Platform::X,
            n_agents: 200,
            n_leaders: 20,
            timesteps: 30,
            activation_rate: 0.3,
            llm_budget: 2000,
            ba_m: 4,
            recsys: RecSysConfig::default(),
            convergence_patience: 3,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub platform: String,
    pub n_agents: usize,
    pub n_leaders: usize,
    pub timesteps: usize,
    pub activation_rate: f64,
    pub llm_budget: usize,
    pub ba_m: usize,
    pub recsys: String,
    pub k_in: usize,
    pub k_out: usize,
    pub convergence_patience: usize,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            platform: self.platform.label().to_string(),
            n_agents: self.n_agents,
            n_leaders: self.n_leaders,
            timesteps: self.timesteps,
            activation_rate: self.activation_rate,
            llm_budget: self.llm_budget,
            ba_m: self.ba_m,
            recsys: self.recsys.kind.label().to_string(),
            k_in: self.recsys.k_in,
            k_out: self.recsys.k_out,
            convergence_patience: self.convergence_patience,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}

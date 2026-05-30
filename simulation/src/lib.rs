//! Yang et al. (2024) "OASIS: Open Agent Social Interaction Simulations with One
//! Million Agents" の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した，**動的ソーシャルネットワーク** (BA フォロー
//! グラフ) 上の LLM 駆動の行動選択 + 決定論的推薦器 + 情報伝播の公開 API を提供
//! する．設定 (`config`)・世界状態 (`world`)・推薦器 (`recsys`)・LLM クライアント層
//! (`llm`)・プロンプト生成 (`prompts`)・応答パース (`parse`)・更新メカニズム
//! (`mechanisms`)・実行ドライバ (`simulation`)・集計メトリクス (`metrics`) を
//! モジュールとして公開し，バイナリ (`oasis`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (BA 網生成・活性化・推薦・情報伝播・指標) は seed から bit 単位で
//! 決定論的である．LLM レイヤ (leader の行動選択) は socsim の bit 再現性の
//! **外側** にあり，`socsim-llm` のキャッシュ + `temperature=0` + `seed` 固定で
//! 擬似決定論化する．詳細は `crate::llm` を参照．

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod parse;
pub mod prompts;
pub mod recsys;
pub mod reproduce_mock;
pub mod simulation;
pub mod world;

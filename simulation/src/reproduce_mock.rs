//! オフライン (LLM 不要) 再現用のスクリプト化クライアント．
//!
//! Yang et al. (2024) OASIS の **見出し的創発現象** を，ライブ LLM 無しで構造的に
//! 再現するための決定論的 mock を提供する．`reproduce --mock` / `run --mock`，および
//! テストがこの mock を共用する．
//!
//! 再現する定性的挙動 (論文 §4 / §5):
//! - **情報拡散カスケード**: leader はフィードに推薦された投稿があれば，その **先頭**
//!   (= RecSys が最上位にランクした投稿) をリポストする．これにより推薦器が «見せた»
//!   投稿が多段にカスケードする．推薦器が良い候補を返すほど (interest / hot-score)，
//!   フィードが埋まり拡散が伸びる．推薦器 `none` はフォロー先の最新のみを返すため
//!   フィードが薄く，拡散が阻害される (RecSys アブレーションの効果)．
//! - **グループ極化 / 群衆効果**: リポスト/いいねは [`crate::mechanisms`] の
//!   `InfoPropagationMechanism` で意見を元投稿側へドリフトさせる．leader が最上位
//!   投稿に同調し続けることで，集団意見が特定方向へ寄り (群衆追随)，意見分布が
//!   推薦バイアスに沿って構造化する．
//!
//! この mock は ground-truth LLM ではなく，論文の定性的結論を再現するための «同調的
//! 増幅器の戯画» である．プロンプト文字列から «フィード先頭の post_index» を読み取り，
//! それをリポスト対象にする．フィードが空なら新規投稿する (カスケードの新たな起点)．
//! ライブ llama3.2 ではこの戯画ではなく実モデルの応答を用いる (cache 経由)．

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

use crate::llm::{wrap_client, OasisClient};

/// フィード行のマーカ ([`crate::prompts::action_prompt`] と一致させる)．
const FEED_HEADER: &str = "post_index | author_id | content";

/// プロンプトから «フィード先頭の post_index» を読み取る．
///
/// `action_prompt` はフィードを `[{idx}] author={id} | {content}` 形式で列挙する．
/// 最初の `[` 直後の整数を解析する．フィードが無ければ `None`．
fn first_feed_index(prompt: &str) -> Option<usize> {
    // フィードヘッダ以降に限定して最初の "[N]" を探す (記憶セクションの誤検出回避)．
    let after_header = prompt.split(FEED_HEADER).nth(1)?;
    let open = after_header.find('[')?;
    let rest = &after_header[open + 1..];
    let token: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    token.parse::<usize>().ok()
}

/// 再現用 leader プロンプトに対する応答テキストを決める (mock の中核ロジック)．
///
/// - フィードに推薦投稿がある → その先頭 (RecSys 最上位) を **リポスト** する
///   (情報カスケードの増幅 + 同調による意見ドリフト)．
/// - フィードが空 → 新規 **投稿** する (新たなカスケード起点)．
pub fn reproduce_reply(prompt: &str) -> String {
    match first_feed_index(prompt) {
        Some(idx) => format!(
            "THOUGHT: this resonates, I will amplify it.\n\
             ACTION: repost\nTARGET: {idx}\nCONTENT: -"
        ),
        None => "THOUGHT: I will share my own view.\n\
                 ACTION: post\nTARGET: -\nCONTENT: My stance on the topic."
            .to_string(),
    }
}

/// 再現用の決定論的スクリプトクライアントを構築する (in-memory cache)．
///
/// leader プロンプトには [`reproduce_reply`] の応答を返す．フィード先頭の推薦投稿を
/// リポストする «同調的増幅器» として振る舞い，推薦器が見せた投稿を多段に拡散させる．
pub fn build_reproduce_client() -> OasisClient {
    let backend = ScriptedClient::new("mock-oasis-reproduce", |prompt: &str| {
        reproduce_reply(prompt)
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_first_feed_index() {
        let prompt = "Your recommended feed (post_index | author_id | content):\n\
                      [7] author=3 | hello\n[2] author=4 | world\n";
        assert_eq!(first_feed_index(prompt), Some(7));
    }

    #[test]
    fn empty_feed_has_no_index() {
        let prompt = "Your recommended feed is empty.\n";
        assert_eq!(first_feed_index(prompt), None);
    }

    #[test]
    fn reply_reposts_when_feed_present() {
        let prompt = "Your recommended feed (post_index | author_id | content):\n\
                      [5] author=1 | take\n";
        let r = reproduce_reply(prompt);
        assert!(r.contains("ACTION: repost"));
        assert!(r.contains("TARGET: 5"));
    }

    #[test]
    fn reply_posts_when_feed_empty() {
        let r = reproduce_reply("Your recommended feed is empty.\n");
        assert!(r.contains("ACTION: post"));
    }

    #[test]
    fn does_not_pick_up_memory_section_brackets() {
        // 記憶セクションに "[" があってもフィードヘッダ以降のみ拾う．
        let prompt = "Your recent memory:\n- [old note]\n\n\
                      Your recommended feed (post_index | author_id | content):\n\
                      [9] author=2 | fresh\n";
        assert_eq!(first_feed_index(prompt), Some(9));
    }
}

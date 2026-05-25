//! LLM プロンプト生成 (Agent Module — CoT 行動選択)．
//!
//! オピニオンリーダー (高次数ノード) のフィード + 自己記述 + 記憶を文脈に，
//! CoT で 1 アクション (post/repost/like/follow/none) を選ばせるプロンプトを組む．
//! 応答は [`crate::parse::parse_action`] でパースする．

use crate::world::{AgentProfile, Post};

/// 行動選択プロンプトを組み立てる．
///
/// `feed` はこのエージェントに推薦された投稿 (新しい順を想定しない; RecSys の並び)．
/// `feed_authors` は対応する投稿のインデックス (follow ターゲット決定用に番号付け)．
pub fn action_prompt(profile: &AgentProfile, feed: &[(usize, &Post)]) -> String {
    let mut s = String::new();
    s.push_str("You are an agent on a social media platform. Decide your next single action.\n\n");
    s.push_str(&format!("Your name: {}\n", profile.name));
    s.push_str(&format!("Your bio: {}\n", profile.bio));
    s.push_str(&format!(
        "Your current opinion (-1=against, +1=in favor): {:.2}\n\n",
        profile.opinion
    ));

    if profile.memory.is_empty() {
        s.push_str("Your recent memory: (none)\n\n");
    } else {
        s.push_str("Your recent memory:\n");
        for m in profile.memory.iter().rev().take(3) {
            s.push_str(&format!("- {m}\n"));
        }
        s.push('\n');
    }

    if feed.is_empty() {
        s.push_str("Your recommended feed is empty.\n\n");
    } else {
        s.push_str("Your recommended feed (post_index | author_id | content):\n");
        for (idx, post) in feed {
            s.push_str(&format!(
                "[{idx}] author={} | {}\n",
                post.author.0, post.content
            ));
        }
        s.push('\n');
    }

    s.push_str(
        "Think step by step, then choose exactly one action. Reply in this exact format:\n\
         THOUGHT: <one sentence>\n\
         ACTION: <post|repost|like|follow|none>\n\
         TARGET: <post_index for repost/like/follow, or - for post/none>\n\
         CONTENT: <text for post, or - otherwise>\n",
    );
    s
}

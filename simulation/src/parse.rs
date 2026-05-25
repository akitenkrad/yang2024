//! LLM 応答パース (Agent Module の行動決定)．
//!
//! [`crate::prompts::action_prompt`] の応答テキストから `ACTION` / `TARGET` /
//! `CONTENT` を抽出する．読めない量は安全側 (None / `ActionKind::None`) に倒す．

/// エージェントが選んだ行動の種別．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// 新規投稿．
    Post,
    /// リポスト (TARGET の投稿を転送)．
    Repost,
    /// いいね (TARGET の投稿に upvote)．
    Like,
    /// フォロー (TARGET の投稿の著者をフォロー)．
    Follow,
    /// 何もしない．
    None,
}

impl ActionKind {
    /// 文字列トークンから行動種別をパースする (未知語は `None`)．
    pub fn parse(s: &str) -> ActionKind {
        match s.trim().to_ascii_lowercase().as_str() {
            "post" | "tweet" => ActionKind::Post,
            "repost" | "retweet" | "share" => ActionKind::Repost,
            "like" | "upvote" => ActionKind::Like,
            "follow" => ActionKind::Follow,
            _ => ActionKind::None,
        }
    }

    /// 短いラベル (出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::Post => "post",
            ActionKind::Repost => "repost",
            ActionKind::Like => "like",
            ActionKind::Follow => "follow",
            ActionKind::None => "none",
        }
    }
}

/// パース済みの行動決定．
#[derive(Debug, Clone)]
pub struct ActionDecision {
    /// 行動種別．
    pub kind: ActionKind,
    /// 対象 post インデックス (repost/like/follow 用; None なら無効)．
    pub target: Option<usize>,
    /// 新規投稿の本文 (post 用)．
    pub content: Option<String>,
}

/// `key:` で始まる行の値を取り出す (先頭一致，大文字小文字無視)．
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            return Some(rest.trim_start_matches(':').trim());
        }
        // 大文字小文字を無視した比較 (LLM が小文字で返すことがある)．
        let lower = line.to_ascii_lowercase();
        let klower = key.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix(&klower) {
            let offset = line.len() - rest.len();
            return Some(line[offset..].trim_start_matches(':').trim());
        }
    }
    None
}

/// 応答テキストから行動決定をパースする．
pub fn parse_action(text: &str) -> ActionDecision {
    let kind = field(text, "ACTION")
        .map(ActionKind::parse)
        .unwrap_or(ActionKind::None);

    let target = field(text, "TARGET").and_then(|s| {
        let cleaned: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        cleaned.parse::<usize>().ok()
    });

    let content = field(text, "CONTENT").and_then(|s| {
        let s = s.trim();
        if s.is_empty() || s == "-" {
            None
        } else {
            Some(s.to_string())
        }
    });

    ActionDecision {
        kind,
        target,
        content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_post() {
        let t =
            "THOUGHT: I want to share.\nACTION: post\nTARGET: -\nCONTENT: Nuclear energy is safe.";
        let d = parse_action(t);
        assert_eq!(d.kind, ActionKind::Post);
        assert_eq!(d.target, None);
        assert_eq!(d.content.as_deref(), Some("Nuclear energy is safe."));
    }

    #[test]
    fn parses_repost_with_target() {
        let t = "ACTION: repost\nTARGET: [3]\nCONTENT: -";
        let d = parse_action(t);
        assert_eq!(d.kind, ActionKind::Repost);
        assert_eq!(d.target, Some(3));
        assert_eq!(d.content, None);
    }

    #[test]
    fn unknown_action_is_none() {
        let d = parse_action("ACTION: ponder\nTARGET: -");
        assert_eq!(d.kind, ActionKind::None);
    }

    #[test]
    fn case_insensitive_keys() {
        let d = parse_action("action: like\ntarget: 2");
        assert_eq!(d.kind, ActionKind::Like);
        assert_eq!(d.target, Some(2));
    }
}

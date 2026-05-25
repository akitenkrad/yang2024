"""oasis-tools reproduce — 論文 Finding の一括再現 (Phase 3; 未実装スタブ)．

Yang et al. (2024) OASIS の主要 Finding (情報拡散カスケード / グループ極化 /
群衆効果 / RecSys アブレーション) を一括で再現・検証する予定のサブコマンド．
本フェーズ (Phase 1+2) では未実装であり，スタブとして案内のみ表示する．

当面は `oasis run` / `oasis sweep` を直接呼んで個別 Finding を検証してください:

    # 情報拡散 (X, 小規模) — カスケードが多段に広がるか
    cargo run --release -- run --platform x --n-agents 200 --n-leaders 20 --timesteps 30

    # グループ極化 — 極化指数 P が時間とともに増大するか
    uv run oasis-tools visualize

    # スケール効果 — N 増で P・多様性が増大するか
    cargo run --release -- sweep --n-agents-values 200,1000,5000

    # RecSys アブレーション — none で伝播到達が急減するか
    cargo run --release -- run --recsys none ...
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oasis-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    print(__doc__)
    print("reproduce は Phase 3 で実装予定です (現状は未実装スタブ)．")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""oasis-tools — Yang et al. (2024) OASIS ツール統合 CLI．

Usage:
    oasis-tools visualize [...]
    oasis-tools visualize-sweep [...]
    oasis-tools show-experiment-settings [...]
    oasis-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示/
再現の実体 (visualize / visualize_sweep / show_experiment_settings / reproduce_paper)
は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="oasis-tools",
    description="Yang et al. (2024) OASIS LLM ソーシャルメディアシミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (極化推移・カスケード木・active-user 数) の可視化",
            "oasis_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (N × activation の集団指標) の可視化",
            "oasis_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / llm_meta) の表示",
            "oasis_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "論文 Finding の一括再現 (Phase 3; 未実装スタブ)",
            "oasis_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()

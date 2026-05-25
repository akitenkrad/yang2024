"""oasis-tools — Yang et al. (2024) OASIS ツール統合 CLI．

Usage:
    oasis-tools visualize [...]
    oasis-tools visualize-sweep [...]
    oasis-tools show-experiment-settings [...]
    oasis-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="oasis-tools",
        description="Yang et al. (2024) OASIS LLM ソーシャルメディアシミュレーション 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (極化推移・カスケード木・active-user 数) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (N × activation の集団指標) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / llm_meta) の表示",
        add_help=False,
    )
    subparsers.add_parser(
        "reproduce",
        help="論文 Finding の一括再現 (Phase 3; 未実装スタブ)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from oasis_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from oasis_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from oasis_tools.show_experiment_settings import main as run_main

        run_main(rest)
    elif command == "reproduce":
        from oasis_tools.reproduce_paper import main as run_main

        run_main(rest)
    else:
        # 未知のコマンドは argparse のエラーメッセージに委ねる
        parser.parse_args(argv)


if __name__ == "__main__":
    main()

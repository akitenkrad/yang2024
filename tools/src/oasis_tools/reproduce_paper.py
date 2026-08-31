#!/usr/bin/env python3
"""reproduce_paper.py — Yang et al. (2024) OASIS 創発現象の一括再現レポート + 図．

Rust の `oasis reproduce` が書いた run ディレクトリを読み，論文の中心的な創発現象を
3 つの図で可視化しつつ PASS/off テーブルを表示する．RecSys アブレーション行列は
`metrics.csv` の run スコープ指標 (`<推薦器>_mean_*`)，アンカーの判定は `events.jsonl`
の `x.yang2024.anchor` にある:

    1. recsys_diffusion.png
       推薦器 (interest / hot-score / none) 別の最終 伝播到達・最大カスケード規模・
       最大カスケード幅 棒グラフ．グローバル人気で全員に最ホット投稿を見せる
       hot-score が，フォロー先ローカルの最新のみを見せる none より大きなカスケード
       を生むこと (= 推薦器が拡散を形作る; RecSys アブレーション) を一目で示す．
    2. polarization_crowd.png
       推薦器別の最終 極化指数 P・極化増分・群衆追随率 棒グラフ．LLM (mock) の同調的
       増幅が集団意見を構造化し，極化と群衆効果を創発させることを示す．
    3. cascade_timeseries.png
       代表 run の最大カスケード規模・伝播到達の時系列を推薦器ごとに重ね描き．
       カスケードが多段に成長する過程 (情報拡散) を時系列で対比する．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する．

--results-dir を省略すると
`runvault path --experiment oasis --latest --subcommand reproduce`
が返す run ディレクトリを対象にする (`runvault` が PATH にある必要がある)．

Usage:
    uv run oasis-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run oasis-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run oasis-tools reproduce                        # 既存の最新 reproduce run を可視化
    uv run oasis-tools reproduce --results-dir "$(runvault path --experiment oasis --latest --subcommand reproduce)"
    uv run oasis-tools reproduce --json

Outputs:
    <experiment>/figures/<run_slug>/{recsys_diffusion,polarization_crowd,cascade_timeseries}.png
    stdout: アンカーごとの PASS / OFF．
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runvault.read import (
    config_parameters,
    events_table,
    figures_dir,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

# --------------------------------------------------------------------------- #
# runvault 側の名前 (Rust 側 record.rs と揃える)
# --------------------------------------------------------------------------- #
EXPERIMENT = "oasis"
ANCHOR_EVENT = "x.yang2024.anchor"

#: 条件セルの run スコープ指標 (Rust 側 ReproCell::metrics と同じ並び)．
CELL_METRICS = [
    "mean_propagation_reach",
    "mean_cascade_size_max",
    "mean_cascade_max_breadth",
    "mean_n_posts",
    "mean_polarization_index",
    "mean_polarization_gain",
    "mean_herd_disagree_rate",
    "mean_final_step",
]

# --------------------------------------------------------------------------- #
# 表示設定 (CJK フォントが利用不能でも落ちないように try)
# --------------------------------------------------------------------------- #
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

COLOR_BG = "#FAFAF8"
RECSYS_COLORS = {
    "interest": "#2196F3",
    "hot-score": "#FF9800",
    "none": "#9C27B0",
}


# --------------------------------------------------------------------------- #
# Rust バイナリ実行
# --------------------------------------------------------------------------- #


def _run_binary(*, mock: bool, quick: bool, seed: int, output_dir: str) -> None:
    """`cargo run --release -- reproduce ...` を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed),
           "--output-dir", output_dir]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def cell_table(scoped: dict[str, float], recsys_values: list[str]) -> list[dict]:
    """RecSys アブレーション行列を 1 行 1 条件の並びに組み直す．

    runvault にはこの表がファイルとして存在しない．`metrics.csv` の run スコープ指標は
    `<推薦器ラベル>_<指標名>` という名前で 1 本の run に同居しているので，ラベルで
    切り分ける．
    """
    cells: list[dict] = []
    for label in recsys_values:
        if f"{label}_{CELL_METRICS[0]}" not in scoped:
            continue
        cell = {"label": label}
        cell.update({name: scoped[f"{label}_{name}"] for name in CELL_METRICS})
        cells.append(cell)
    return cells


def anchor_rows(run_dir: Path) -> list[dict]:
    """`events.jsonl` のアンカー判定．無ければ空 (表を 1 つ落とすだけ)．"""
    try:
        return events_table(run_dir, kind=ANCHOR_EVENT).to_dict(orient="records")
    except (FileNotFoundError, SystemExit):
        return []


def _recsys_color(label: str) -> str:
    return RECSYS_COLORS.get(label, "#607D8B")


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _recsys_diffusion(cells: list[dict], out_path: Path) -> None:
    """推薦器別の最終 伝播到達・最大カスケード規模・幅 棒グラフ (情報拡散)．"""
    labels = [c["label"] for c in cells]
    colors = [_recsys_color(t) for t in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Yang et al. (2024) OASIS — RecSys アブレーション (情報拡散)",
        fontsize=13,
    )

    panels = [
        ("mean_propagation_reach", "伝播到達 (ユニークノード)", "拡散の広さ"),
        ("mean_cascade_size_max", "最大カスケード規模", "拡散の深さ (推薦器が増幅)"),
        ("mean_cascade_max_breadth", "最大カスケード幅", "同時拡散の幅"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(COLOR_BG)
        ax.bar(x, [c[key] for c in cells], color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("推薦器")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _polarization_crowd(cells: list[dict], out_path: Path) -> None:
    """推薦器別の最終 極化指数 P・極化増分・群衆追随率 棒グラフ．"""
    labels = [c["label"] for c in cells]
    colors = [_recsys_color(t) for t in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Yang et al. (2024) OASIS — 極化・群衆効果 (推薦器別)",
        fontsize=13,
    )

    panels = [
        ("mean_polarization_index", "最終 極化指数 P", "グループ極化"),
        ("mean_polarization_gain", "極化増分 (最終 − 初期)", "極化の進行 (符号に注目)"),
        ("mean_herd_disagree_rate", "群衆追随率", "群衆効果 (down-treat 群追随)"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(COLOR_BG)
        ax.bar(x, [c[key] for c in cells], color=colors, alpha=0.9)
        ax.axhline(0.0, color="#888888", lw=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("推薦器")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _cascade_timeseries(cells: list[dict], wide: pd.DataFrame, out_path: Path) -> None:
    """推薦器別の最大カスケード規模・伝播到達 時系列 (代表 run)．

    3 条件が 1 本の run に同居するので，系列は `<推薦器ラベル>_<指標名>` という名前で
    区別されている．
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Yang et al. (2024) OASIS — カスケード成長 (代表 run; 推薦器別)",
        fontsize=13,
    )

    plotted = 0
    for ax, (metric, ylabel) in zip(
        axes,
        [("cascade_size_max", "最大カスケード規模"), ("propagation_reach", "伝播到達")],
    ):
        ax.set_facecolor(COLOR_BG)
        for c in cells:
            label = c["label"]
            column = f"{label}_{metric}"
            if column not in wide.columns:
                continue
            # 条件ごとに停止するステップが違う．pivot は足りない側を NaN で埋めるので，
            # 切り出した後に落とす — 描かない点と «値が 0» を取り違えないため．
            series = wide[["step", column]].dropna()
            if series.empty:
                continue
            ax.plot(series["step"], series[column], color=_recsys_color(label),
                    lw=2, marker="o", markersize=3, label=label)
            plotted += 1
        ax.set_xlabel("時刻 t (ステップ)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} の時間発展", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if plotted == 0:
        print("  警告: 条件別の系列が無いため cascade_timeseries をスキップ")
        plt.close(fig)
        return

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _print_report(
    params: dict,
    scoped: dict[str, float],
    cells: list[dict],
    anchors: list[dict],
    results_dir: Path,
) -> None:
    print("=" * 78)
    print("Yang et al. (2024) OASIS — 創発現象 一括再現レポート")
    mode = "mock" if params.get("mock") else "live"
    print(f"  source: {results_dir}  (mode={mode})")
    print("=" * 78)

    print("\n[RecSys アブレーション行列 (拡散 / 極化 / 群衆効果)]")
    print(f"  {'recsys':<12}{'reach':>8}{'casc':>8}{'breadth':>8}"
          f"{'P':>10}{'P-gain':>9}{'herd':>8}")
    for c in cells:
        print(f"  {c['label']:<12}{c['mean_propagation_reach']:>8.2f}"
              f"{c['mean_cascade_size_max']:>8.2f}{c['mean_cascade_max_breadth']:>8.2f}"
              f"{c['mean_polarization_index']:>10.4f}{c['mean_polarization_gain']:>9.4f}"
              f"{c['mean_herd_disagree_rate']:>8.3f}")

    print("\n[論文知見アンカー (観測 vs 論文)]")
    for a in anchors:
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or pd.isna(hi) else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        print(f"  [{status}] {a['name']:<26} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{int(scoped.get('anchors_passed', 0))}/"
          f"{int(scoped.get('anchors_total', len(anchors)))} アンカーが in-band")
    print("(中核知見: 推薦器が情報カスケードを形作る / 同調的増幅で極化・群衆効果が創発)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oasis-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="`oasis reproduce` の run ディレクトリ "
                             "(省略時は runvault path --latest --subcommand reproduce)")
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="runvault の results ルート (default: results)")
    parser.add_argument("--experiment", default=EXPERIMENT,
                        help=f"runvault の experiment 名 (default: {EXPERIMENT})")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: <experiment>/figures/<run_slug>)")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時にライブ LLM を使わず mock で駆動する．")
    parser.add_argument("--quick", action="store_true",
                        help="--run 時に軽量モードで実行する (動作確認用)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--cargo-output-dir", "--cargo_output_dir", default="results",
                        help="--run 時に cargo の --output-dir へ渡すパス (既定: results)．")
    parser.add_argument("--json", action="store_true", help="JSON 形式で要約を出力する．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(mock=args.mock, quick=args.quick, seed=args.seed,
                    output_dir=args.cargo_output_dir)

    results_dir = Path(
        args.results_dir
        or runvault_path(args.experiment, args.results_root, subcommand="reproduce")
    )
    if not (results_dir / "metrics.csv").exists():
        print(f"エラー: metrics.csv が見つかりません: {results_dir}\n"
              f"  先に `oasis-tools reproduce --run --mock` を実行してください．",
              file=sys.stderr)
        return 1

    params = config_parameters(results_dir) or {}
    scoped = run_scope_metrics(results_dir)
    cells = cell_table(scoped, list(params.get("recsys_values") or RECSYS_COLORS))
    anchors = anchor_rows(results_dir)

    if args.json:
        payload = {
            "source": str(results_dir),
            "parameters": params,
            "run_scope_metrics": scoped,
            "recsys_ablation": cells,
            "anchors": anchors,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    _print_report(params, scoped, cells, anchors, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _recsys_diffusion(cells, out_dir / "recsys_diffusion.png")
    _polarization_crowd(cells, out_dir / "polarization_crowd.png")
    _cascade_timeseries(
        cells, metrics_wide(results_dir / "metrics.csv"), out_dir / "cascade_timeseries.png"
    )

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

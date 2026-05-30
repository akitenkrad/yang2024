#!/usr/bin/env python3
"""reproduce_paper.py — Yang et al. (2024) OASIS 創発現象の一括再現レポート + 図．

Rust の `oasis reproduce` が書き出す `reproduce_summary.json` (RecSys アブレーション
行列・論文知見アンカー) と条件別 `metrics_<recsys>.csv` を読み，論文の中心的な創発
現象を 3 つの図で可視化しつつ PASS/off テーブルを表示する:

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

Usage:
    uv run oasis-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run oasis-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run oasis-tools reproduce                        # 既存 results/latest を可視化
    uv run oasis-tools reproduce --results-dir results/reproduce_20260530_000000
    uv run oasis-tools reproduce --json

Outputs:
    {results_dir}/figures/{recsys_diffusion,polarization_crowd,cascade_timeseries}.png
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

from socsim_tools.io import resolve_results_dir

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


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `oasis-tools reproduce --run --mock` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _recsys_color(label: str) -> str:
    return RECSYS_COLORS.get(label, "#607D8B")


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _recsys_diffusion(summary: dict, out_path: Path) -> None:
    """推薦器別の最終 伝播到達・最大カスケード規模・幅 棒グラフ (情報拡散)．"""
    cells = summary["recsys_ablation"]
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


def _polarization_crowd(summary: dict, out_path: Path) -> None:
    """推薦器別の最終 極化指数 P・極化増分・群衆追随率 棒グラフ．"""
    cells = summary["recsys_ablation"]
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


def _metric_series(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """long-format metrics.csv から 1 指標の (t, value) 系列を取り出す．"""
    sub = df[df["metric"] == metric][["t", "value"]].sort_values("t")
    return sub


def _cascade_timeseries(summary: dict, results_dir: Path, out_path: Path) -> None:
    """推薦器別の最大カスケード規模・伝播到達 時系列 (代表 run)．"""
    cells = summary["recsys_ablation"]
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
            path = results_dir / f"metrics_{label}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            series = _metric_series(df, metric)
            if series.empty:
                continue
            ax.plot(series["t"], series["value"], color=_recsys_color(label),
                    lw=2, marker="o", markersize=3, label=label)
            plotted += 1
        ax.set_xlabel("時刻 t (ステップ)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} の時間発展", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if plotted == 0:
        print("  警告: metrics_<recsys>.csv が無いため cascade_timeseries をスキップ")
        plt.close(fig)
        return

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _print_report(summary: dict, results_dir: Path) -> None:
    print("=" * 78)
    print("Yang et al. (2024) OASIS — 創発現象 一括再現レポート")
    print(f"  source: {results_dir}  (mode={summary.get('mode', '?')})")
    print("=" * 78)

    print("\n[RecSys アブレーション行列 (拡散 / 極化 / 群衆効果)]")
    print(f"  {'recsys':<12}{'reach':>8}{'casc':>8}{'breadth':>8}"
          f"{'P':>10}{'P-gain':>9}{'herd':>8}")
    for c in summary["recsys_ablation"]:
        print(f"  {c['label']:<12}{c['mean_propagation_reach']:>8.2f}"
              f"{c['mean_cascade_size_max']:>8.2f}{c['mean_cascade_max_breadth']:>8.2f}"
              f"{c['mean_polarization_index']:>10.4f}{c['mean_polarization_gain']:>9.4f}"
              f"{c['mean_herd_disagree_rate']:>8.3f}")

    print("\n[論文知見アンカー (観測 vs 論文)]")
    n_pass = 0
    for a in summary["anchors"]:
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi > 1e30 else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(f"  [{status}] {a['name']:<50} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{n_pass}/{len(summary['anchors'])} アンカーが in-band")
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
                        help="reproduce_summary.json のあるディレクトリ (既定: results/latest)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: {results_dir}/figures)")
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

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_report(summary, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _recsys_diffusion(summary, out_dir / "recsys_diffusion.png")
    _polarization_crowd(summary, out_dir / "polarization_crowd.png")
    _cascade_timeseries(summary, results_dir, out_dir / "cascade_timeseries.png")

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

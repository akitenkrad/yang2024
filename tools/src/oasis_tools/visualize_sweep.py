#!/usr/bin/env python3
"""
visualize_sweep.py — Yang et al. (2024) OASIS スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
N (エージェント数) × activation_rate (活性化率) の格子について最終集団指標
(極化指数 P・意見多様性・伝播到達数・最大カスケード規模) を集計し，ヒートマップと
折れ線で可視化する．論文の核心であるスケール効果 (N 増 → P・多様性増) の確認用．

Usage:
    uv run oasis-tools visualize-sweep
    uv run oasis-tools visualize-sweep --sweep_dir results/20260525_160000_sweep

Outputs:
    output_dir/
    ├── sweep_polarization_heatmap.png ← 極化指数 P (N × activation)
    ├── sweep_reach_heatmap.png        ← 伝播到達数 (N × activation)
    └── sweep_metrics_vs_n.png         ← 指標 vs N (activation 別折れ線)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(activation_rate, n_agents) ごとに metric の試行平均をピボットする．"""
    agg = df.groupby(["activation_rate", "n_agents"])[metric].mean().reset_index()
    return agg.pivot(index="activation_rate", columns="n_agents", values=metric)


def save_heatmap(table: pd.DataFrame, title: str, out_path: str, cmap: str) -> None:
    """activation_rate × n_agents のヒートマップを保存する．"""
    fig, ax = plt.subplots(
        figsize=(2.2 + 1.3 * table.shape[1], 1.8 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([f"{r:.2f}" for r in table.index])
    ax.set_xlabel("エージェント数 N")
    ax.set_ylabel("活性化率")
    ax.set_title(title, fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_metrics_vs_n(df: pd.DataFrame, out_path: str) -> None:
    """エージェント数に対する指標を activation 別の折れ線で比較する (スケール効果)．"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=COLOR_BG)
    metrics = [
        ("final_polarization_index", "極化指数 P"),
        ("final_opinion_std", "意見多様性 (std)"),
        ("final_propagation_reach", "伝播到達数"),
    ]
    activations = sorted(df["activation_rate"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, (col, label) in zip(axes, metrics):
        ax.set_facecolor(COLOR_BG)
        for k, a in enumerate(activations):
            sub = df[df["activation_rate"] == a]
            agg = sub.groupby("n_agents")[col].mean().reset_index()
            color = cmap(k / max(1, len(activations) - 1))
            ax.plot(agg["n_agents"], agg[col], marker="o", lw=2,
                    label=f"act={a:.2f}", color=color)
        ax.set_xlabel("エージェント数 N")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs N")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="oasis-tools visualize-sweep",
        description="Yang et al. (2024) OASIS スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Yang et al. (2024) OASIS スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/3] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(f"      N {df['n_agents'].nunique()} 種 × activation {df['activation_rate'].nunique()} 種")

    print("[2/3] ヒートマップを保存中 ...")
    save_heatmap(
        pivot_metric(df, "final_polarization_index"),
        "最終 極化指数 P (N × activation)",
        os.path.join(out_dir, "sweep_polarization_heatmap.png"),
        cmap="RdYlGn_r",
    )
    save_heatmap(
        pivot_metric(df, "final_propagation_reach"),
        "最終 伝播到達数 (N × activation)",
        os.path.join(out_dir, "sweep_reach_heatmap.png"),
        cmap="YlGnBu",
    )

    print("[3/3] 指標 vs N 折れ線を保存中 ...")
    save_metrics_vs_n(df, os.path.join(out_dir, "sweep_metrics_vs_n.png"))

    print("-------------------------------------------------")
    print("エージェント数別の平均 極化指数 P:")
    for n in sorted(df["n_agents"].unique()):
        v = df[df["n_agents"] == n]["final_polarization_index"].mean()
        print(f"  N={n:<6} → P̄ = {v:.4f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
visualize.py — Yang et al. (2024) OASIS 単一実行結果 可視化スクリプト

results/latest (または --results_dir 指定先) の metrics.csv (long-format) と
cascades.csv を読み，以下の図を生成する:
(1) 極化指数 P の時系列 (グループ極化; Finding 2)
(2) active-user 数の時系列 (Time Engine 検証)
(3) 伝播到達数・最大カスケード規模の時系列 (情報拡散; Finding 1)
(4) カスケード木 (規模上位カスケードの root → リポストの簡易ツリー; networkx)

Usage:
    uv run oasis-tools visualize
    uv run oasis-tools visualize --results_dir results/20260525_103000
    uv run oasis-tools visualize --output_dir out --no-graph

Outputs:
    output_dir/
    ├── metrics_timeseries.png ← 極化・active-user・伝播到達・カスケード規模
    └── cascade_tree.png       ← 規模上位カスケードの木 (任意)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_POL = "#F44336"
COLOR_ACTIVE = "#2196F3"
COLOR_REACH = "#4CAF50"
COLOR_CASC = "#FF9800"


def load_metrics(path: str) -> pd.DataFrame:
    """metrics.csv (long-format: t, metric, value) を wide-format にピボットする．"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    long_df = pd.read_csv(path)
    wide = long_df.pivot_table(index="t", columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide.sort_values("t").reset_index(drop=True)


def load_cascades(results_dir: str) -> pd.DataFrame | None:
    path = os.path.join(results_dir, "cascades.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def save_metrics_timeseries(df: pd.DataFrame, out_path: str) -> None:
    """集団指標の時系列図 (4 パネル) を保存する．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=COLOR_BG)
    fig.suptitle("Yang et al. (2024) OASIS — 集団指標の時系列", fontsize=14)
    t = df["t"]

    # (1) 極化指数 P
    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    if "polarization_index" in df:
        ax.plot(t, df["polarization_index"], color=COLOR_POL, lw=2, marker="o", ms=3)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("極化指数 P")
    ax.set_title("グループ極化 (P = 意見分散)")
    ax.grid(True, alpha=0.3)

    # (2) active-user 数
    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    if "active_user_count" in df:
        ax.plot(t, df["active_user_count"], color=COLOR_ACTIVE, lw=2, marker="s", ms=3)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("active-user 数")
    ax.set_title("Time Engine (active-user 数)")
    ax.grid(True, alpha=0.3)

    # (3) 伝播到達数
    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    if "propagation_reach" in df:
        ax.plot(t, df["propagation_reach"], color=COLOR_REACH, lw=2, marker="^", ms=3)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("伝播到達ユニークノード数")
    ax.set_title("情報伝播到達 (RecSys 媒介)")
    ax.grid(True, alpha=0.3)

    # (4) 最大カスケード規模
    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    if "cascade_size_max" in df:
        ax.plot(t, df["cascade_size_max"], color=COLOR_CASC, lw=2, marker="d", ms=3,
                label="最大カスケード規模")
    if "cascade_max_breadth" in df:
        ax.plot(t, df["cascade_max_breadth"], color="#9C27B0", lw=1.5, ls="--",
                marker="x", ms=3, label="最大カスケード幅")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("カスケード規模 / 幅")
    ax.set_title("情報拡散カスケード (Finding 1)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_cascade_tree(cascades: pd.DataFrame, out_path: str, top_k: int = 6) -> None:
    """規模上位カスケードを root → リポスト の星形ツリーで簡易描画する．

    cascades.csv は (root_post, author, size) のみを持つため，root を中心に
    size-1 個のリポストノードを放射状に配置した近似ツリーを描く．
    """
    import networkx as nx

    if cascades.empty:
        print("  カスケードが空のためツリー描画をスキップ．")
        return

    top = cascades.sort_values("size", ascending=False).head(top_k)
    n_panels = len(top)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), facecolor=COLOR_BG)
    axes = [axes] if n_panels == 1 else list(axes.flatten())

    for ax, (_, row) in zip(axes, top.iterrows()):
        ax.set_facecolor(COLOR_BG)
        size = int(row["size"])
        g = nx.Graph()
        root = f"root#{int(row['root_post'])}\n(author {int(row['author'])})"
        g.add_node(root)
        for i in range(max(0, size - 1)):
            g.add_edge(root, f"r{i}")
        pos = nx.spring_layout(g, seed=0)
        node_colors = ["#F44336" if n == root else "#90CAF9" for n in g.nodes()]
        node_sizes = [400 if n == root else 120 for n in g.nodes()]
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes,
                               alpha=0.9, ax=ax)
        nx.draw_networkx_edges(g, pos, alpha=0.4, edge_color="#555555", ax=ax)
        ax.set_title(f"カスケード規模 = {size}", fontsize=11)
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("情報拡散カスケード木 (規模上位)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="oasis-tools visualize",
        description="Yang et al. (2024) OASIS 単一実行結果 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="Rust シミュレーションの出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {results_dir}/figures)",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="カスケード木描画を抑止する．",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Yang et al. (2024) OASIS 単一実行結果 可視化 ===")
    print(f"メトリクス: {metrics_path}")
    print(f"出力先:     {out_dir}")
    print("-----------------------------------------")

    print("[1/2] メトリクス時系列を保存中 ...")
    df = load_metrics(metrics_path)
    print(f"      {len(df)} timestep")
    save_metrics_timeseries(df, os.path.join(out_dir, "metrics_timeseries.png"))

    if not args.no_graph:
        cascades = load_cascades(args.results_dir)
        if cascades is not None:
            print("[2/2] カスケード木を保存中 ...")
            save_cascade_tree(cascades, os.path.join(out_dir, "cascade_tree.png"))
        else:
            print("[2/2] cascades.csv が無いためカスケード木描画をスキップ．")
    else:
        print("[2/2] --no-graph 指定によりカスケード木描画をスキップ．")

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()

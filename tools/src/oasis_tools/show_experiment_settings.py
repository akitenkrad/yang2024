"""oasis-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば llm_meta.json の LLM 情報
(プロバイダ・モデル・endpoint・温度・seed・cache-hit 率) も併せて表示する．
`results/latest` も解決される．

Usage:
    oasis-tools show-experiment-settings
    oasis-tools show-experiment-settings --results-dir results/20260525_103000
    oasis-tools show-experiment-settings --results-dir results/latest --json

results_dir の解決は共有ヘルパ `socsim_tools.io.resolve_results_dir` に委譲する
(出力はバイト等価)．run 設定テーブルは複合行 (`k_in / k_out`) を含み，LLM メタは
`llm_meta.json` (provider フィールド付き) を読むため，そのレンダラ・ローダと
`--json` の `kind`/`llm_meta` フィールドは oasis 固有なので本モジュールに残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import resolve_results_dir


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run) または sweep_config.json (sweep)"
    )


def _load_llm_meta(results_dir: Path) -> dict | None:
    path = results_dir / "llm_meta.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def render_run_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"プラットフォーム : {cfg.get('platform', '-')}")
    lines.append(f"推薦器           : {cfg.get('recsys', '-')}")
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"リーダー数       : {cfg.get('n_leaders', '-')}")
    lines.append(f"タイムステップ T : {cfg.get('timesteps', '-')}")
    lines.append(f"活性化率         : {cfg.get('activation_rate', '-')}")
    lines.append(f"LLM 予算         : {cfg.get('llm_budget', '-')}")
    lines.append(f"BA m             : {cfg.get('ba_m', '-')}")
    lines.append(f"k_in / k_out     : {cfg.get('k_in', '-')} / {cfg.get('k_out', '-')}")
    lines.append(f"収束 patience    : {cfg.get('convergence_patience', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先           : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"プラットフォーム : {cfg.get('platform', '-')}")
    lines.append(f"推薦器           : {cfg.get('recsys', '-')}")
    ns = cfg.get("n_agents_values", [])
    lines.append(f"エージェント数   : {', '.join(str(x) for x in ns)}")
    acts = cfg.get("activation_rate_values", [])
    lines.append(f"活性化率         : {', '.join(str(x) for x in acts)}")
    lines.append(f"リーダー数       : {cfg.get('n_leaders', '-')}")
    lines.append(f"タイムステップ T : {cfg.get('timesteps', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (llm_meta.json)")
    lines.append("-" * 70)
    lines.append(f"プロバイダ       : {meta.get('provider', '-')}")
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint         : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oasis-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = _load_llm_meta(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())

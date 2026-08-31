"""oasis-tools show-experiment-settings — 実行結果の設定表示．

run ディレクトリの config.json (runvault の封筒; 条件は parameters の下) を読み，
実行時に使われた全パラメータを整形表示する．run.json の llm ブロックと metrics.csv の
run スコープ指標 (LLM 呼び出し数・cache-hit) も併せて表示する．移行前の flat な
config.json / sweep_config.json / llm_meta.json の LLM 情報
(プロバイダ・モデル・endpoint・温度・seed・cache-hit 率) も併せて表示する．
--results-dir を省略すると `runvault path --experiment oasis --latest` が返す run を対象にする．

Usage:
    oasis-tools show-experiment-settings
    oasis-tools show-experiment-settings --results-dir results/20260525_103000
    oasis-tools show-experiment-settings --json

run ディレクトリの読み方は `runvault.read` に委譲する
(出力はバイト等価)．run 設定テーブルは複合行 (`k_in / k_out`) を含み，LLM メタは
`llm_meta.json` (provider フィールド付き) を読むため，そのレンダラ・ローダと
`--json` の `kind`/`llm_meta` フィールドは oasis 固有なので本モジュールに残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, run_scope_metrics, runvault_path

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "oasis"

#: sweep 親の設定にだけ現れるキー．これがあればグリッドの表として描く．
SWEEP_MARKER = "n_agents_values"


def _find_config_file(results_dir: Path) -> Path:
    """`config.json` (runvault の封筒 / legacy の flat) か legacy の `sweep_config.json`．"""
    for name in ("config.json", "sweep_config.json"):
        path = results_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
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
    lines.append(f"mock 駆動        : {cfg.get('mock', '-')}")
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


def render_llm_block(meta: dict, scoped: dict[str, float]) -> str:
    """`run.json` の `llm` ブロックと LLM 呼び出しの内訳．

    endpoint と «決定論の注記» は旧 `llm_meta.json` にあったが持ち込まない．前者は
    provider の分類に使われるだけで run の同一性には効かず，後者は設計の説明
    (docs/architecture.ja.md) であって run ごとの記録ではない．
    """
    llm = meta.get("llm") or {}
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run.json の llm ブロック / metrics.csv)")
    lines.append("-" * 70)
    lines.append(f"プロバイダ       : {llm.get('provider', '-')}")
    lines.append(f"モデル           : {llm.get('model_snapshot', '-')}")
    lines.append(f"温度             : {llm.get('temperature', '-')}")
    rng = meta.get("rng") or {}
    lines.append(f"master_seed      : {rng.get('master_seed', '-')}")
    if "llm_calls" in scoped:
        lines.append(f"呼び出し総数     : {int(scoped['llm_calls'])}")
    if "llm_cache_hits" in scoped:
        lines.append(f"cache-hit        : {int(scoped['llm_cache_hits'])}")
    if "llm_cache_hit_rate" in scoped:
        lines.append(f"cache-hit 率     : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    """移行前の `llm_meta.json` の表示 (legacy な results/ 用)．"""
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (llm_meta.json; 移行前の run)")
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
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT,
        help=f"runvault の experiment 名 (default: {EXPERIMENT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = Path(
        args.results_dir or runvault_path(args.experiment, args.results_root)
    )
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    if cfg_path.name == "sweep_config.json":
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        cfg = config_parameters(results_dir) or {}
    # runvault の run では sweep 親も config.json を持つので，どちらの表を描くかは
    # 格子の定義キーがあるかで決める．
    kind = "sweep" if SWEEP_MARKER in cfg else "run"
    run_meta = load_run_meta(results_dir, required=False)
    # 指標を読むのは runvault の run だけ．legacy の metrics.csv は run スコープ行を
    # 持たず，時間軸の列名も `step` ではないので `run_scope_metrics` を通せない．
    scoped = run_scope_metrics(results_dir) if run_meta is not None else {}
    legacy = _load_llm_meta(results_dir)

    if args.json:
        payload = {
            "source": str(cfg_path),
            "kind": kind,
            "config": cfg,
            "run_meta": run_meta,
            "run_scope_metrics": scoped,
            "llm_meta": legacy,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if run_meta is not None:
            print(render_llm_block(run_meta, scoped))
        elif legacy is not None:
            print(render_llm_meta(legacy))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
用 Golden Set 校准可回答性阈值（v2 §5.2：阈值不能拍脑袋）。

方法：
- 用 full 模式跑出的每条 query 的 top rerank 分数；
- 遍历候选阈值，计算 answerability recall、abstention recall、abstention precision；
- 在 abstention precision ≥ 目标值（默认 0.9）的约束下，选 balanced accuracy 最高的阈值。

用法：
    uv run -m eval.calibrate_threshold
    uv run -m eval.calibrate_threshold --report eval/results/eval-xxxx.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_report() -> Path:
    files = sorted(glob.glob(str(PROJECT_ROOT / "eval" / "results" / "eval-*.json")))
    if not files:
        raise SystemExit("没有找到 eval 报告，请先运行 eval.run_eval")
    return Path(files[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None)
    parser.add_argument("--target-precision", type=float, default=0.9)
    args = parser.parse_args()

    path = Path(args.report) if args.report else latest_report()
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = report["modes"]["full"]["queries"]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    print(f"报告：{path}")
    print(f"可回答 {len(answerable)} 条，不可回答 {len(unanswerable)} 条\n")
    print(f"{'threshold':>10} {'ANS-R':>8} {'ABST-R':>8} {'ABST-P':>8} {'BAL-ACC':>8} {'FP':>4} {'FN':>4}")

    best = None
    for step in range(0, 51):
        threshold = round(step * 0.01, 2)
        tp = sum(1 for row in answerable if (row.get("top_rerank") or 0) >= threshold)
        fn = len(answerable) - tp
        tn = sum(1 for row in unanswerable if (row.get("top_rerank") or 0) < threshold)
        fp = len(unanswerable) - tn
        ans_recall = tp / len(answerable) if answerable else 0.0
        abst_recall = tn / len(unanswerable) if unanswerable else 0.0
        abst_precision = tn / (tn + fn) if (tn + fn) else 0.0
        balanced = (ans_recall + abst_recall) / 2
        if abst_precision >= args.target_precision:
            if best is None or balanced > best[1]:
                best = (threshold, balanced, ans_recall, abst_recall, abst_precision, fp, fn)
        if step % 5 == 0:
            print(
                f"{threshold:>10.2f} {ans_recall:>8.3f} {abst_recall:>8.3f} "
                f"{abst_precision:>8.3f} {balanced:>8.3f} {fp:>4} {fn:>4}"
            )

    if best:
        print(
            f"\n推荐阈值：{best[0]:.2f}（balanced_acc={best[1]:.3f}, "
            f"answerability_recall={best[2]:.3f}, abstention_recall={best[3]:.3f}, "
            f"abstention_precision={best[4]:.3f}, FP={best[5]}, FN={best[6]}）"
        )
    else:
        print(f"\n在 abstention_precision ≥ {args.target_precision} 约束下没有可行阈值")


if __name__ == "__main__":
    main()

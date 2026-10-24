"""
对比两份 eval 报告（同一 Golden Set、同一索引快照）。

用法：
    .venv\\Scripts\\python.exe -m eval.compare_reports \\
        --baseline hybrid=eval/results/eval-xxx.json \\
        --candidate qwen3=eval/results/eval-yyy.json \\
        --candidate qwen3vl=eval/results/eval-zzz.json \\
        --out eval/results/rerank-comparison.json

输出：
- 每个模型的汇总指标（直接取报告里的值）；
- 候选相对基线的逐 query 胜负（按第一个相关块的倒数排名 RR 比较）；
- top-1 命中变化、最大提升/退化的 query 例子；
- 控制台表格。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _first_mode(report: dict) -> str:
    for name in ("full", "hybrid", "dense"):
        if name in report.get("modes", {}):
            return name
    return next(iter(report["modes"]))


def _rr(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def load_report(name: str, path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    mode = _first_mode(report)
    result = report["modes"][mode]
    rows = {}
    for item in result.get("queries", []):
        relevant = set(item.get("relevant") or [])
        retrieved = item.get("retrieved") or []
        rows[item["id"]] = {
            "id": item["id"],
            "query": item.get("query", ""),
            "query_type": item.get("query_type", "unknown"),
            "answerable": bool(item.get("answerable", True)),
            "relevant_count": len(relevant),
            "retrieved": retrieved,
            "rr": _rr(retrieved, relevant),
            "hit": 1.0 if set(retrieved[:10]) & relevant else 0.0,
            "top1_relevant": bool(retrieved and retrieved[0] in relevant),
            "top_rerank": item.get("top_rerank"),
            "latency_ms": item.get("latency_ms", 0),
            "status": item.get("status"),
        }
    return {
        "name": name,
        "path": path,
        "mode": mode,
        "collection": report.get("collection"),
        "rerank_model": report.get("rerank_model"),
        "metrics": {
            "recall@10": result.get("recall@10"),
            "hit@10": result.get("hit@10"),
            "mrr@10": result.get("mrr@10"),
            "ndcg@10": result.get("ndcg@10"),
            "degraded_ratio": result.get("degraded_ratio"),
            "latency_avg_ms": result.get("latency_avg_ms"),
            "latency_p95_ms": result.get("latency_p95_ms"),
        },
        "rows": rows,
    }


def compare(base: dict, candidate: dict) -> dict:
    shared = sorted(set(base["rows"]) & set(candidate["rows"]))
    wins = losses = ties = 0
    top1_gain = top1_loss = 0
    deltas = []
    for qid in shared:
        b = base["rows"][qid]
        c = candidate["rows"][qid]
        delta = c["rr"] - b["rr"]
        deltas.append(
            {
                "id": qid,
                "query": c["query"],
                "query_type": c["query_type"],
                "delta_rr": round(delta, 6),
                "baseline_rr": round(b["rr"], 6),
                "candidate_rr": round(c["rr"], 6),
                "baseline_top1_relevant": b["top1_relevant"],
                "candidate_top1_relevant": c["top1_relevant"],
            }
        )
        if abs(delta) < 1e-9:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
        if c["top1_relevant"] and not b["top1_relevant"]:
            top1_gain += 1
        if b["top1_relevant"] and not c["top1_relevant"]:
            top1_loss += 1

    deltas.sort(key=lambda item: item["delta_rr"])
    return {
        "baseline": base["name"],
        "candidate": candidate["name"],
        "shared_queries": len(shared),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "top1_gain": top1_gain,
        "top1_loss": top1_loss,
        "avg_rr_delta": round(
            sum(item["delta_rr"] for item in deltas) / len(deltas), 6
        ) if deltas else 0.0,
        "worst": deltas[:5],
        "best": list(reversed(deltas[-5:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="name=path")
    parser.add_argument("--candidate", action="append", default=[], help="name=path，可重复")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    def parse_pair(value: str) -> tuple[str, str]:
        if "=" not in value:
            raise SystemExit(f"格式应为 name=path：{value}")
        return tuple(value.split("=", 1))

    base_name, base_path = parse_pair(args.baseline)
    base = load_report(base_name, base_path)
    candidates = [load_report(*parse_pair(value)) for value in args.candidate]

    print(f"{'model':<12} {'R@10':>8} {'Hit@10':>8} {'MRR@10':>8} {'NDCG@10':>8} {'avg ms':>8} {'p95 ms':>8}")
    for item in [base] + candidates:
        m = item["metrics"]
        print(
            f"{item['name']:<12} {m['recall@10']:>8.4f} {m['hit@10']:>8.4f} "
            f"{m['mrr@10']:>8.4f} {m['ndcg@10']:>8.4f} "
            f"{m['latency_avg_ms']:>8.1f} {m['latency_p95_ms']:>8.1f}"
        )

    comparisons = []
    for candidate in candidates:
        comp = compare(base, candidate)
        comparisons.append(comp)
        print(
            f"\n{candidate['name']} vs {base['name']}: "
            f"win/loss/tie={comp['wins']}/{comp['losses']}/{comp['ties']}, "
            f"top1 +{comp['top1_gain']}/-{comp['top1_loss']}, "
            f"avg RR delta={comp['avg_rr_delta']:+.4f}"
        )
        for item in comp["best"][-3:]:
            print(f"  + {item['id']} {item['query']} (RR {item['baseline_rr']:.2f}→{item['candidate_rr']:.2f})")
        for item in comp["worst"][:3]:
            print(f"  - {item['id']} {item['query']} (RR {item['baseline_rr']:.2f}→{item['candidate_rr']:.2f})")

    payload = {
        "reports": [
            {key: item[key] for key in ("name", "path", "mode", "collection", "rerank_model", "metrics")}
            for item in [base] + candidates
        ],
        "comparisons": comparisons,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()

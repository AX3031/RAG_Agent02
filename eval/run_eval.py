"""
离线检索评测入口。

用法：
    uv run -m eval.run_eval --modes dense,hybrid,full --top-k 10
    uv run -m eval.run_eval --save-baseline

设计要点：
- 三种模式共用同一份索引与同一批 Golden Set，只切换检索链路：
    dense  = 纯向量（旧实现基线）
    hybrid = Dense + BM25 + RRF
    full   = Hybrid + gte-rerank（新实现）
- 不调用生成模型；除 rerank/embedding 外不依赖网络，结果确定可复现；
- 输出 JSON 快照到 `eval/results/`，便于用 diff 对比每次改动。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

from rag.retrieval_service import RetrievalService
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.logger_handler import logger

from eval.retrieval_metrics import (
    average,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    random_recall_at_k,
    recall_at_k,
    resolve_snippets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = [PROJECT_ROOT / "eval" / "golden_set.jsonl"]


def load_golden(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            logger.warning("Golden Set 不存在，跳过：%s", path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_mode_config(mode: str, top_k: int) -> dict:
    conf = copy.deepcopy(rag_conf)
    retrieval = conf["retrieval"]
    retrieval["enable_sparse"] = mode in {"hybrid", "full"}
    retrieval["enable_rerank"] = mode == "full"
    retrieval["final_top_n"] = top_k
    retrieval["rerank_top_n"] = top_k
    retrieval["answerability_threshold"] = 0.0
    retrieval["max_context_chars"] = 10_000_000  # 评测不看上下文裁剪
    retrieval["cache_size"] = 0
    # 只有 full 模式需要评估 answerability；dense/hybrid 不调用额外 LLM
    retrieval["answerability_mode"] = "llm" if mode == "full" else "threshold"
    return conf


def evaluate_mode(mode: str, golden: list[dict], documents: list, top_k: int) -> dict:
    service = RetrievalService(
        vector_store=VectorStoreService(),
        config=build_mode_config(mode, top_k),
    )
    per_query: list[dict] = []
    unresolved: list[dict] = []

    for entry in golden:
        answerable = bool(entry.get("answerable", True))
        snippets = entry.get("relevant_snippets") or []
        relevant, missing = resolve_snippets(snippets, documents)
        if missing:
            unresolved.append({"id": entry["id"], "missing": missing})

        result = service.retrieve(entry["query"], top_k=top_k)
        retrieved = [item.chunk_id for item in result.evidence]
        row = {
            "id": entry["id"],
            "query_type": entry.get("query_type", "unknown"),
            "query": entry["query"],
            "answerable": answerable,
            "predicted_answerable": bool(result.answerable),
            "status": result.status.value,
            "retrieved": retrieved,
            "relevant": sorted(relevant),
            "top_rerank": (
                result.evidence[0].score.get("rerank") if result.evidence else None
            ),
            "degraded_stages": list(result.degraded_stages),
            "latency_ms": result.latency_ms.get("total", 0),
        }
        if answerable and relevant:
            row.update(
                {
                    f"recall@{top_k}": recall_at_k(retrieved, relevant, top_k),
                    f"hit@{top_k}": hit_at_k(retrieved, relevant, top_k),
                    f"mrr@{top_k}": mrr_at_k(retrieved, relevant, top_k),
                    f"ndcg@{top_k}": ndcg_at_k(retrieved, relevant, top_k),
                }
            )
        per_query.append(row)

    answerable_rows = [row for row in per_query if row["answerable"] and row.get(f"recall@{top_k}") is not None]
    unanswerable_rows = [row for row in per_query if not row["answerable"]]
    abstentions = [row for row in per_query if not row["predicted_answerable"]]

    relevant_counts = [len(row["relevant"]) for row in answerable_rows]
    total_chunks = len(documents)
    random_recall = average(
        [
            random_recall_at_k(total_chunks, count, top_k)
            for count in relevant_counts
            if count
        ]
    )

    def bucket(rows: list[dict]) -> dict:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["query_type"], []).append(row)
        return {
            qtype: {
                "count": len(items),
                f"recall@{top_k}": round(average([item[f"recall@{top_k}"] for item in items]), 4),
                f"mrr@{top_k}": round(average([item[f"mrr@{top_k}"] for item in items]), 4),
                f"ndcg@{top_k}": round(average([item[f"ndcg@{top_k}"] for item in items]), 4),
            }
            for qtype, items in sorted(grouped.items())
        }

    latencies = [float(row["latency_ms"]) for row in per_query]
    degraded_rows = [row for row in per_query if row["degraded_stages"]]
    return {
        "mode": mode,
        "count": len(per_query),
        "degraded_ratio": round(len(degraded_rows) / len(per_query), 4) if per_query else 0.0,
        "rerank_degraded_count": sum(
            1
            for row in per_query
            if any(stage.startswith("rerank") for stage in row["degraded_stages"])
        ),
        f"recall@{top_k}": round(average([row[f"recall@{top_k}"] for row in answerable_rows]), 4),
        f"hit@{top_k}": round(average([row[f"hit@{top_k}"] for row in answerable_rows]), 4),
        f"mrr@{top_k}": round(average([row[f"mrr@{top_k}"] for row in answerable_rows]), 4),
        f"ndcg@{top_k}": round(average([row[f"ndcg@{top_k}"] for row in answerable_rows]), 4),
        "random_recall": round(random_recall, 4),
        "answerability_recall": round(
            average([1.0 if row["predicted_answerable"] else 0.0 for row in answerable_rows]), 4
        ),
        "abstention_recall": round(
            average([1.0 if not row["predicted_answerable"] else 0.0 for row in unanswerable_rows]), 4
        ),
        "abstention_precision": round(
            average([1.0 if not row["answerable"] else 0.0 for row in abstentions]), 4
        ),
        "latency_avg_ms": round(average(latencies), 2),
        "latency_p95_ms": round(percentile(latencies, 0.95), 2),
        "buckets": bucket(answerable_rows),
        "unresolved": unresolved,
        "queries": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索侧离线评测")
    parser.add_argument("--modes", default="dense,hybrid,full", help="逗号分隔：dense,hybrid,full")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--golden", default=None, help="逗号分隔的 Golden Set 路径")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "eval" / "results"))
    parser.add_argument("--save-baseline", action="store_true", help="把 full 模式结果写为 baseline.json")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    args = parser.parse_args()

    if args.golden:
        golden_paths = [Path(p.strip()) for p in args.golden.split(",") if p.strip()]
    else:
        golden_paths = list(DEFAULT_GOLDEN)
        local_path = PROJECT_ROOT / "eval" / "golden_set.local.jsonl"
        if local_path.exists():
            golden_paths.append(local_path)

    golden = load_golden(golden_paths)
    if args.limit:
        golden = golden[: args.limit]
    if not golden:
        raise SystemExit("没有可用的 Golden Set")

    vector_store = VectorStoreService()
    documents = vector_store.get_all_documents()
    stats = vector_store.stats()
    logger.info("评测语料：%s，Golden Set：%s 条", stats, len(golden))

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "golden_files": [str(path) for path in golden_paths],
        "golden_count": len(golden),
        "top_k": args.top_k,
        "corpus": stats,
        "modes": {},
    }

    for mode in modes:
        logger.info("开始评测模式：%s", mode)
        report["modes"][mode] = evaluate_mode(mode, golden, documents, args.top_k)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = outdir / f"eval-{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.save_baseline and "full" in report["modes"]:
        full_result = report["modes"]["full"]
        if full_result.get("degraded_ratio", 1.0) >= 0.5:
            logger.warning(
                "full 模式降级比例 %.0f%%，不覆盖 baseline（请恢复 rerank 额度后重跑）",
                full_result["degraded_ratio"] * 100,
            )
        else:
            baseline_path = outdir / "baseline.json"
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generated_at": report["generated_at"],
                        "golden_files": report["golden_files"],
                        "golden_count": report["golden_count"],
                        "top_k": args.top_k,
                        "corpus": stats,
                        "mode": "full",
                        "result": full_result,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("baseline 已写入 %s", baseline_path)

    # 控制台摘要
    print(f"\n{'mode':<8} {'R@K':>8} {'Hit@K':>8} {'MRR@K':>8} {'NDCG@K':>8} "
          f"{'ABST-R':>8} {'ABST-P':>8} {'avg ms':>8} {'p95 ms':>8}")
    for mode, result in report["modes"].items():
        print(
            f"{mode:<8} {result[f'recall@{args.top_k}']:>8.4f} {result[f'hit@{args.top_k}']:>8.4f} "
            f"{result[f'mrr@{args.top_k}']:>8.4f} {result[f'ndcg@{args.top_k}']:>8.4f} "
            f"{result['abstention_recall']:>8.4f} {result['abstention_precision']:>8.4f} "
            f"{result['latency_avg_ms']:>8.1f} {result['latency_p95_ms']:>8.1f}"
        )
    print(f"\n报告：{out_path}")
    print(f"随机 Recall 基线：{report['modes'][modes[-1]]['random_recall']:.4f}")

    unresolved = report["modes"][modes[-1]].get("unresolved") or []
    if unresolved:
        print("\n[警告] 以下标注片段未能在语料中定位（请检查 Golden Set）：")
        for item in unresolved:
            print("  -", item["id"], item["missing"])


if __name__ == "__main__":
    main()

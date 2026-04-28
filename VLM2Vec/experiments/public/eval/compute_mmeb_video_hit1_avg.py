#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


# MMEB V2 video task taxonomy (aligned with video.yaml)
TASK_TO_DATASETS: Dict[str, List[str]] = {
    "V-CLS": [
        "SmthSmthV2",
        "HMDB51",
        "UCF101",
        "K700",
        "Breakfast",
    ],
    "V-QA": [
        "Video-MME",
        "NExTQA",
        "EgoSchema",
        "MVBench",
        "ActivityNetQA",
    ],
    "V-RET": [
        "MSR-VTT",
        "MSVD",
        "DiDeMo",
        "YouCook2",
        "VATEX",
    ],
    "V-MR": [
        "QVHighlight",
        "Charades-STA",
        "MomentSeeker",
    ],
}


def build_dataset_to_task() -> Dict[str, str]:
    dataset_to_task: Dict[str, str] = {}
    for task, datasets in TASK_TO_DATASETS.items():
        for dataset in datasets:
            if dataset in dataset_to_task:
                raise ValueError(f"Dataset {dataset} appears in multiple task groups.")
            dataset_to_task[dataset] = task
    return dataset_to_task


def parse_hit1(score_file: Path) -> float:
    with score_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "hit@1" not in data:
        raise KeyError(f"`hit@1` not found in {score_file}")
    return float(data["hit@1"])


def load_benchmark_scores(benchmark_json: Path) -> Dict[str, float]:
    with benchmark_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    video_metrics = metrics.get("video", {})
    benchmark_scores: Dict[str, float] = {}
    for dataset in build_dataset_to_task():
        hit1 = video_metrics.get(dataset, {}).get("hit@1", None)
        if hit1 is not None:
            benchmark_scores[dataset] = float(hit1)
    return benchmark_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average hit@1 for MMEB V2 video evaluation."
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=Path(
            "/home/guohaiyun/yangtianyu/UME-R1/output/Eval/UME-R1_2B/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52_checkpoint-1431/video-gen"
        ),
        help="Directory containing *_gen_gen_score.json files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_gen_gen_score.json",
        help="Score filename suffix.",
    )
    parser.add_argument(
        "--benchmark_json",
        type=Path,
        default=Path(
            "/home/guohaiyun/yangtianyu/UME-R1/src/eval/VLM2Vec/experiments/public/all_scores/UME-R1-Qwen2VL-2B.json"
        ),
        help="Optional benchmark JSON in public/all_scores format.",
    )
    args = parser.parse_args()

    score_dir = args.score_dir
    if not score_dir.exists() or not score_dir.is_dir():
        raise FileNotFoundError(f"score_dir does not exist or is not a directory: {score_dir}")

    dataset_to_task = build_dataset_to_task()

    benchmark_scores: Dict[str, float] = {}
    if args.benchmark_json.exists():
        benchmark_scores = load_benchmark_scores(args.benchmark_json)
    else:
        print(f"[WARN] benchmark_json not found, skip benchmark diff: {args.benchmark_json}")

    all_scores: Dict[str, float] = {}
    task_scores: Dict[str, List[float]] = {k: [] for k in TASK_TO_DATASETS}
    unknown_datasets: List[str] = []

    for score_file in sorted(score_dir.glob(f"*{args.suffix}")):
        dataset_name = score_file.name[: -len(args.suffix)]
        hit1 = parse_hit1(score_file)
        all_scores[dataset_name] = hit1
        task = dataset_to_task.get(dataset_name)
        if task is None:
            unknown_datasets.append(dataset_name)
            continue
        task_scores[task].append(hit1)

    if not all_scores:
        raise RuntimeError(f"No score files matched in {score_dir} with suffix {args.suffix}")

    missing = sorted(set(dataset_to_task) - set(all_scores))
    if missing:
        print("[WARN] Missing expected MMEB V2 video datasets:")
        print("  " + ", ".join(missing))

    if unknown_datasets:
        print("[WARN] Found datasets not in MMEB V2 video taxonomy:")
        print("  " + ", ".join(sorted(unknown_datasets)))

    overall = sum(all_scores.values()) / len(all_scores)
    bench_matched = {k: v for k, v in benchmark_scores.items() if k in all_scores}
    bench_overall = sum(bench_matched.values()) / len(bench_matched) if bench_matched else 0.0

    print(f"score_dir: {score_dir}")
    print(f"num_files: {len(all_scores)}")
    if bench_matched:
        print(
            f"overall_hit@1_avg: {overall:.6f}  "
            f"(benchmark: {bench_overall:.6f}, diff: {overall - bench_overall:+.6f})"
        )
    else:
        print(f"overall_hit@1_avg: {overall:.6f}")

    for task_name in ["V-CLS", "V-QA", "V-RET", "V-MR"]:
        scores = task_scores[task_name]
        if not scores:
            print(f"{task_name}_hit@1_avg: N/A (0 files)")
            continue
        avg = sum(scores) / len(scores)
        task_ds = TASK_TO_DATASETS[task_name]
        task_bench = [benchmark_scores[d] for d in task_ds if d in all_scores and d in benchmark_scores]
        if task_bench:
            task_bench_avg = sum(task_bench) / len(task_bench)
            print(
                f"{task_name}_hit@1_avg: {avg:.6f}  "
                f"(benchmark: {task_bench_avg:.6f}, diff: {avg - task_bench_avg:+.6f}, n={len(scores)})"
            )
        else:
            print(f"{task_name}_hit@1_avg: {avg:.6f}  (n={len(scores)})")

    print()
    print(f"{'Dataset':<18} {'Task':<6} {'Score':>8} {'Bench':>8} {'Diff':>9}")
    print("-" * 55)
    for ds in sorted(all_scores):
        score = all_scores[ds]
        task = dataset_to_task.get(ds, "N/A")
        bench = benchmark_scores.get(ds)
        if bench is not None:
            print(f"{ds:<18} {task:<6} {score:>8.4f} {bench:>8.4f} {score - bench:>+9.4f}")
        else:
            print(f"{ds:<18} {task:<6} {score:>8.4f} {'N/A':>8} {'':>9}")


if __name__ == "__main__":
    main()

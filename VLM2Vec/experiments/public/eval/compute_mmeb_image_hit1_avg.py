#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

benchmark_score={
        "ImageNet-1K": 0.753,
        "N24News": 0.811,
        "HatefulMemes": 0.752,
        "VOC2007": 0.80,
        "SUN397": 0.794,
        "Place365": 0.426,
        "ImageNet-A": 0.504,
        "ImageNet-R": 0.887 ,
        "ObjectNet": 0.520  ,
        "Country211": 0.234,
        "OK-VQA": 0.624,
        "A-OKVQA": 0.511,
        "DocVQA": 0.922,
        "InfographicsVQA": 0.677,
        "ChartQA": 0.649,
        "Visual7W": 0.541,
        "ScienceQA": 0.427,
        "VizWiz": 0.468,
        "GQA": 0.673,
        "TextVQA": 0.786,
        "VisDial": 0.766,
        "CIRR": 0.537,
        "VisualNews_t2i": 0.717,
        "VisualNews_i2t": 0.742,
        "MSCOCO_t2i": 0.751,
        "MSCOCO_i2t": 0.689,
        "NIGHTS": 0.672,
        "WebQA": 0.900,
        "FashionIQ": 0.171,
        "Wiki-SS-NQ": 0.620,
        "OVEN": 0.669,
        "EDIS": 0.88,
        "MSCOCO": 0.695,
        "RefCOCO": 0.833,
        "RefCOCO-Matching": 0.844,
        "Visual7W-Pointing": 0.715,
    }
# MMEB image task taxonomy (aligned with image.yaml)
TASK_TO_DATASETS: Dict[str, List[str]] = {
    "I-CLS": [
        "ImageNet-1K",
        "N24News",
        "HatefulMemes",
        "VOC2007",
        "SUN397",
        "Place365",
        "ImageNet-A",
        "ImageNet-R",
        "ObjectNet",
        "Country211",
    ],
    "I-QA": [
        "OK-VQA",
        "A-OKVQA",
        "DocVQA",
        "InfographicsVQA",
        "ChartQA",
        "Visual7W",
        "ScienceQA",
        "VizWiz",
        "GQA",
        "TextVQA",
    ],
    "I-RET": [
        "MSCOCO_i2t",
        "VisualNews_i2t",
        "VisDial",
        "MSCOCO_t2i",
        "VisualNews_t2i",
        "WebQA",
        "EDIS",
        "Wiki-SS-NQ",
        "CIRR",
        "NIGHTS",
        "OVEN",
        "FashionIQ",
    ],
    "I-VG": [
        "MSCOCO",
        "RefCOCO",
        "RefCOCO-Matching",
        "Visual7W-Pointing",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average hit@1 for MMEB image evaluation."
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        default=Path(
            "/home/guohaiyun/yangtianyu/UME-R1/output/Eval/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52_checkpoint-1431/image-gen"
        ),
        help="Directory containing *_gen_gen_score.json files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        # default="_gen_gen_score.json",
        default="_gen_gen_score.json",
        help="Score filename suffix.",
    )
    args = parser.parse_args()

    score_dir = args.score_dir
    if not score_dir.exists() or not score_dir.is_dir():
        raise FileNotFoundError(f"score_dir does not exist or is not a directory: {score_dir}")

    dataset_to_task = build_dataset_to_task()

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
        print("[WARN] Missing expected MMEB image datasets:")
        print("  " + ", ".join(missing))

    if unknown_datasets:
        print("[WARN] Found datasets not in MMEB image taxonomy:")
        print("  " + ", ".join(sorted(unknown_datasets)))

    overall = sum(all_scores.values()) / len(all_scores)
    bench_matched = {k: v for k, v in benchmark_score.items() if k in all_scores}
    bench_overall = sum(bench_matched.values()) / len(bench_matched) if bench_matched else 0.0

    print(f"score_dir: {score_dir}")
    print(f"num_files: {len(all_scores)}")
    print(f"overall_hit@1_avg: {overall:.6f}  (benchmark: {bench_overall:.6f}, diff: {overall - bench_overall:+.6f})")

    for task_name in ["I-CLS", "I-QA", "I-RET", "I-VG"]:
        scores = task_scores[task_name]
        if not scores:
            print(f"{task_name}_hit@1_avg: N/A (0 files)")
            continue
        avg = sum(scores) / len(scores)
        task_ds = TASK_TO_DATASETS[task_name]
        task_bench = [benchmark_score[d] for d in task_ds if d in all_scores and d in benchmark_score]
        task_bench_avg = sum(task_bench) / len(task_bench) if task_bench else 0.0
        print(f"{task_name}_hit@1_avg: {avg:.6f}  (benchmark: {task_bench_avg:.6f}, diff: {avg - task_bench_avg:+.6f}, n={len(scores)})")

    print()
    print(f"{'Dataset':<25} {'Score':>8} {'Bench':>8} {'Diff':>9}")
    print("-" * 53)
    for ds in sorted(all_scores, key=lambda d: all_scores[d] - benchmark_score.get(d, all_scores[d]), reverse=True):
        score = all_scores[ds]
        bench = benchmark_score.get(ds)
        if bench is not None:
            print(f"{ds:<25} {score:>8.4f} {bench:>8.4f} {score - bench:>+9.4f}")
        else:
            print(f"{ds:<25} {score:>8.4f} {'N/A':>8} {'':>9}")


if __name__ == "__main__":
    main()

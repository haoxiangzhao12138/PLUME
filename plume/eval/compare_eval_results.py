#!/usr/bin/env python3
"""
Compare hit@1 metrics between two evaluation result directories.

Usage:
    python eval/tools/compare_eval_results.py dir1/ dir2/
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_score_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def get_dataset_name(filename: str) -> str:
    return filename.replace('_gen_gen_score.json', '').replace('_score.json', '')


def compare_results(dir1: str, dir2: str, metric: str = 'hit@1') -> List[Tuple[str, float, float, float]]:
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    score_files1 = {f.name: f for f in dir1_path.glob('*_score.json')}
    score_files2 = {f.name: f for f in dir2_path.glob('*_score.json')}

    common_files = set(score_files1.keys()) & set(score_files2.keys())

    results = []
    for filename in sorted(common_files):
        dataset_name = get_dataset_name(filename)

        score1 = load_score_file(score_files1[filename])
        score2 = load_score_file(score_files2[filename])

        value1 = score1.get(metric, None)
        value2 = score2.get(metric, None)

        if value1 is not None and value2 is not None:
            diff = value2 - value1
            results.append((dataset_name, value1, value2, diff))

    return results


def print_comparison(results: List[Tuple[str, float, float, float]],
                    dir1_name: str, dir2_name: str, metric: str = 'hit@1'):
    print(f"\n{'='*80}")
    print(f"Evaluation Comparison - {metric}")
    print(f"{'='*80}")
    print(f"\nDir 1: {dir1_name}")
    print(f"Dir 2: {dir2_name}\n")

    print(f"{'Dataset':<30} {'Dir1':<12} {'Dir2':<12} {'Diff':<12} {'Change'}")
    print(f"{'-'*80}")

    total_diff = 0
    for dataset, val1, val2, diff in results:
        if val1 > 0:
            change_pct = (diff / val1) * 100
            change_str = f"{change_pct:+.2f}%"
        else:
            change_str = "N/A"

        if diff > 0:
            indicator = "up"
        elif diff < 0:
            indicator = "down"
        else:
            indicator = "="

        print(f"{dataset:<30} {val1:<12.4f} {val2:<12.4f} {diff:+<12.4f} {change_str:<8} {indicator}")
        total_diff += diff

    print(f"{'-'*80}")
    avg_val1 = sum(r[1] for r in results) / len(results) if results else 0
    avg_val2 = sum(r[2] for r in results) / len(results) if results else 0
    avg_diff = avg_val2 - avg_val1

    print(f"{'Average':<30} {avg_val1:<12.4f} {avg_val2:<12.4f} {avg_diff:+<12.4f}")
    print(f"\nCompared {len(results)} datasets")

    improved = sum(1 for r in results if r[3] > 0)
    declined = sum(1 for r in results if r[3] < 0)
    unchanged = sum(1 for r in results if r[3] == 0)

    print(f"Improved: {improved}, Declined: {declined}, Unchanged: {unchanged}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare evaluation metrics between two result directories')
    parser.add_argument('dir1', type=str, help='First result directory')
    parser.add_argument('dir2', type=str, help='Second result directory')
    parser.add_argument('--metric', type=str, default='hit@1',
                       help='Metric to compare (default: hit@1)')

    args = parser.parse_args()

    if not os.path.exists(args.dir1):
        print(f"Error: directory does not exist: {args.dir1}")
        return
    if not os.path.exists(args.dir2):
        print(f"Error: directory does not exist: {args.dir2}")
        return

    results = compare_results(args.dir1, args.dir2, args.metric)

    if not results:
        print("No comparable datasets found")
        return

    print_comparison(results, args.dir1, args.dir2, args.metric)


if __name__ == '__main__':
    main()

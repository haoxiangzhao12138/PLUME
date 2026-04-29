#!/usr/bin/env python3
"""
比较两个评测结果文件夹中的hit@1指标 
python src/eval/VLM2Vec/experiments/public/eval/compare_eval_results.py \
    output/Eval/UME-R1_2B/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52_checkpoint-1431/image-gen \
    output/Eval/UME-R1-2B-Coconut-Fulldata-NoAns-4node-2026-03-10-10-03-52_checkpoint-1431/image-gen
    
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_score_file(file_path: str) -> Dict:
    """加载score.json文件"""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_dataset_name(filename: str) -> str:
    """从文件名提取数据��名称"""
    # 例如: A-OKVQA_gen_gen_score.json -> A-OKVQA
    return filename.replace('_gen_gen_score.json', '').replace('_score.json', '')


def compare_results(dir1: str, dir2: str, metric: str = 'hit@1') -> List[Tuple[str, float, float, float]]:
    """
    比较两个目录中的评测结果

    Args:
        dir1: 第一个结果目录
        dir2: 第二个结果目录
        metric: 要比较的指标，默认为'hit@1'

    Returns:
        List of (dataset_name, score1, score2, diff)
    """
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    # 获取所有score.json文件
    score_files1 = {f.name: f for f in dir1_path.glob('*_score.json')}
    score_files2 = {f.name: f for f in dir2_path.glob('*_score.json')}

    # 找到共同的数据集
    common_files = set(score_files1.keys()) & set(score_files2.keys())

    results = []
    for filename in sorted(common_files):
        dataset_name = get_dataset_name(filename)

        # 加载两个文件
        score1 = load_score_file(score_files1[filename])
        score2 = load_score_file(score_files2[filename])

        # 获取指标值
        value1 = score1.get(metric, None)
        value2 = score2.get(metric, None)

        if value1 is not None and value2 is not None:
            diff = value2 - value1
            results.append((dataset_name, value1, value2, diff))

    return results


def print_comparison(results: List[Tuple[str, float, float, float]],
                    dir1_name: str, dir2_name: str, metric: str = 'hit@1'):
    """打印比较结果"""
    print(f"\n{'='*80}")
    print(f"评测结果比较 - {metric}")
    print(f"{'='*80}")
    print(f"\n目录1: {dir1_name}")
    print(f"目录2: {dir2_name}\n")

    # 打印表头
    print(f"{'数据集':<30} {'目录1':<12} {'目录2':<12} {'差异':<12} {'变化'}")
    print(f"{'-'*80}")

    # 打印每个数据集的结果
    total_diff = 0
    for dataset, val1, val2, diff in results:
        # 计算变化百分比
        if val1 > 0:
            change_pct = (diff / val1) * 100
            change_str = f"{change_pct:+.2f}%"
        else:
            change_str = "N/A"

        # 根据差异添加颜色标记
        if diff > 0:
            indicator = "↑"
        elif diff < 0:
            indicator = "↓"
        else:
            indicator = "="

        print(f"{dataset:<30} {val1:<12.4f} {val2:<12.4f} {diff:+<12.4f} {change_str:<8} {indicator}")
        total_diff += diff

    # 打印统计信息
    print(f"{'-'*80}")
    avg_val1 = sum(r[1] for r in results) / len(results) if results else 0
    avg_val2 = sum(r[2] for r in results) / len(results) if results else 0
    avg_diff = avg_val2 - avg_val1

    print(f"{'平均值':<30} {avg_val1:<12.4f} {avg_val2:<12.4f} {avg_diff:+<12.4f}")
    print(f"\n共比较 {len(results)} 个数据集")

    # 统计提升和下降的数量
    improved = sum(1 for r in results if r[3] > 0)
    declined = sum(1 for r in results if r[3] < 0)
    unchanged = sum(1 for r in results if r[3] == 0)

    print(f"提升: {improved} 个, 下降: {declined} 个, 不变: {unchanged} 个")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='比较两个评测结果文件夹中的指标')
    parser.add_argument('dir1', type=str, help='第一个结果目录')
    parser.add_argument('dir2', type=str, help='第二个结果目录')
    parser.add_argument('--metric', type=str, default='hit@1',
                       help='要比较的指标 (默认: hit@1)')

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.exists(args.dir1):
        print(f"错误: 目录不存在: {args.dir1}")
        return
    if not os.path.exists(args.dir2):
        print(f"错误: 目录不存在: {args.dir2}")
        return

    # 比较结果
    results = compare_results(args.dir1, args.dir2, args.metric)

    if not results:
        print("没有找到可比较的数据集")
        return

    # 打印结果
    print_comparison(results, args.dir1, args.dir2, args.metric)


if __name__ == '__main__':
    main()

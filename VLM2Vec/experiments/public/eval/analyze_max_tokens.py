#!/usr/bin/env python3
"""
分析评测结果中的max_new_tokens问题
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List


def analyze_score_file(score_path: str) -> Dict:
    """分析单个score文件"""
    with open(score_path, 'r') as f:
        score = json.load(f)

    result = {
        'dataset': Path(score_path).stem.replace('_gen_gen_score', '').replace('_score', ''),
        'num_data': score.get('num_data', 0),
        'hit@1': score.get('hit@1', 0),
    }

    # Query侧的max_tokens统计
    if 'qry_hit_max_new_tokens' in score:
        qry_hit = score['qry_hit_max_new_tokens']
        result['qry_hit_max_tokens'] = qry_hit
        result['qry_hit_rate'] = score.get('qry_hit_max_new_tokens_rate',
                                           f"{qry_hit / result['num_data'] * 100:.2f}%" if result['num_data'] > 0 else "0%")

    # Target侧的max_tokens统计
    if 'tgt_hit_max_new_tokens' in score:
        tgt_hit = score['tgt_hit_max_new_tokens']
        result['tgt_hit_max_tokens'] = tgt_hit
        # 注意：target的总数可能不等于num_data（可能是候选集大小）
        result['tgt_hit_rate'] = score.get('tgt_hit_max_new_tokens_rate', 'N/A')

    return result


def analyze_directory(result_dir: str) -> List[Dict]:
    """分析整个结果目录"""
    result_path = Path(result_dir)
    score_files = list(result_path.glob('*_score.json'))

    results = []
    for score_file in sorted(score_files):
        try:
            result = analyze_score_file(str(score_file))
            results.append(result)
        except Exception as e:
            print(f"Error processing {score_file}: {e}")

    return results


def print_analysis(results: List[Dict], title: str):
    """打印分析结果"""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}\n")

    # 表头
    print(f"{'Dataset':<30} {'Samples':<10} {'hit@1':<10} {'Qry Hit Max':<15} {'Qry Rate':<12} {'Tgt Hit Max':<15} {'Tgt Rate':<12}")
    print(f"{'-'*100}")

    total_samples = 0
    total_qry_hit = 0
    total_tgt_hit = 0
    datasets_with_qry_hit = 0
    datasets_with_tgt_hit = 0

    for result in results:
        dataset = result['dataset']
        num_data = result['num_data']
        hit1 = result.get('hit@1', 0)

        qry_hit = result.get('qry_hit_max_tokens', '-')
        qry_rate = result.get('qry_hit_rate', '-')
        tgt_hit = result.get('tgt_hit_max_tokens', '-')
        tgt_rate = result.get('tgt_hit_rate', '-')

        print(f"{dataset:<30} {num_data:<10} {hit1:<10.4f} {str(qry_hit):<15} {qry_rate:<12} {str(tgt_hit):<15} {tgt_rate:<12}")

        total_samples += num_data
        if isinstance(qry_hit, int):
            total_qry_hit += qry_hit
            datasets_with_qry_hit += 1
        if isinstance(tgt_hit, int):
            total_tgt_hit += tgt_hit
            datasets_with_tgt_hit += 1

    print(f"{'-'*100}")
    print(f"\n统计信息:")
    print(f"  总数据集数: {len(results)}")
    print(f"  总样本数: {total_samples}")

    if datasets_with_qry_hit > 0:
        avg_qry_rate = total_qry_hit / total_samples * 100 if total_samples > 0 else 0
        print(f"  Query侧达到max_tokens: {total_qry_hit} ({avg_qry_rate:.2f}%)")
        print(f"    - 有问题的数据集数: {sum(1 for r in results if r.get('qry_hit_max_tokens', 0) > 0)}/{datasets_with_qry_hit}")

    if datasets_with_tgt_hit > 0:
        print(f"  Target侧达到max_tokens: {total_tgt_hit}")
        print(f"    - 有问题的数据集数: {sum(1 for r in results if r.get('tgt_hit_max_tokens', 0) > 0)}/{datasets_with_tgt_hit}")

    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description='分析评测结果中的max_new_tokens问题')
    parser.add_argument('result_dirs', nargs='+', help='结果目录路径（可以指定多个）')

    args = parser.parse_args()

    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"错误: 目录不存在: {result_dir}")
            continue

        results = analyze_directory(result_dir)

        if not results:
            print(f"警告: 在 {result_dir} 中没有找到score文件")
            continue

        print_analysis(results, f"分析结果: {result_dir}")


if __name__ == '__main__':
    main()

# Token生成调试功能使用说明

## 功能概述

添加了一个可控的开关 `DEBUG_LOG_TOKENS`，用于控制是否输出详细的token生成记录。这个功能可以帮助诊断为什么模型会达到 `max_new_tokens` 限制。

## 为什么需要这个开关？

- **默认关闭**：平时评测时不需要详细日志，避免影响性能
- **按需���启**：当遇到 max_new_tokens 问题时，可以开启来诊断问题
- **性能考虑**：token解码和日志输出会增加评测时间

## 使用方法

### 1. 正常评测（不输出详细日志）

```bash
# 默认情况下，DEBUG_LOG_TOKENS=False
bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
```

### 2. 开启详细日志进行调试

```bash
# 设置 DEBUG_LOG_TOKENS=True
DEBUG_LOG_TOKENS=True \
bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
```

或者在脚本中修改：

```bash
# 在 eval_coconut_all_modalities.sh 中修改
DEBUG_LOG_TOKENS="${DEBUG_LOG_TOKENS:-True}"  # 改为 True
```

### 3. 针对特定数据集调试

```bash
DEBUG_LOG_TOKENS=True \
MODALITIES=image \
DATASET_NAMES=CIRR \
bash src/eval/VLM2Vec/experiments/public/eval/eval_coconut_all_modalities.sh
```

## 输出内容

当 `DEBUG_LOG_TOKENS=True` 时，会输出：

### 1. 在评测开始时显示配置
```
============================================================
  COCONUT Multi-Modal Evaluation
============================================================
  ...
  Debug logging : True
  ...
============================================================
```

### 2. 在日志中记录详细信息

对于达到 max_new_tokens 的样本（100%记录）或随机抽样的正常样本（1%），会输出：

```
================================================================================
[COCONUT LATENT DEBUG] Rank 0
  Latent steps: 6
  Prefix text: <think><bot>
  Suffix text: <eot></think><answer>
  Generated tokens count: 2048
  Max new tokens: 2048
  Reached max: True
  Stopped by special: False
  Gen_emb token ID: 151667
  EOS token IDs: {151643, 151645}
  Generated text:
  <actual decoded text here...>
  Last 20 token IDs: [123, 456, 789, ...]
  Has <gen_emb>: False
  Has EOS: False
================================================================================
```

### 3. 在评测结束时显示统计

```
[EVAL] Encoding queries: hit max_new_tokens without <gen_emb>/eos = 150/1000 (15.00%)
[WARNING] 150 samples reached max_new_tokens! This may indicate the model is not
generating <gen_emb> token properly. Check the debug logs above for token sequences.
```

### 4. 在 score.json 中记录

```json
{
  "hit@1": 0.49,
  ...
  "qry_hit_max_new_tokens": 150,
  "qry_hit_max_new_tokens_rate": "15.00%",
  "tgt_hit_max_new_tokens": 0,
  "tgt_hit_max_new_tokens_rate": "0.00%"
}
```

## 分析工具

使用 `analyze_max_tokens.py` 分析评测结果：

```bash
python src/eval/VLM2Vec/experiments/public/eval/analyze_max_tokens.py \
  output/Eval/newpixel/UME-R1-2B-Coconut-FullData-8node-2026-03-01-00-15-25_checkpoint-1429/image-gen
```

输出示例：
```
====================================================================================================
分析结果: output/Eval/.../image-gen
====================================================================================================

Dataset                        Samples    hit@1      Qry Hit Max     Qry Rate     Tgt Hit Max     Tgt Rate
----------------------------------------------------------------------------------------------------
A-OKVQA                        1000       0.4900     150             15.00%       0               0.00%
CIRR                           500        0.6200     0               0.00%        0               0.00%
...
----------------------------------------------------------------------------------------------------

统计信息:
  总数据集数: 20
  总样本数: 15000
  Query侧达到max_tokens: 150 (1.00%)
    - 有问题的数据集数: 1/20
  Target侧达到max_tokens: 0
    - 有问题的数据集数: 0/20
====================================================================================================
```

## 性能影响

- **关闭时** (`DEBUG_LOG_TOKENS=False`)：无性能影响
- **开启时** (`DEBUG_LOG_TOKENS=True`)：
  - 每个达到 max_tokens 的样本会进行 token 解码（较慢）
  - 1% 的正常样本会被随机记录
  - 预计增加 5-10% 的评测时间

## 建议使用场景

1. **首次评测新模型**：开启，检查是否有 max_tokens 问题
2. **日常评测**：关闭，提高效率
3. **调试问题**：开启，查看具体生成内容
4. **性能测试**：关闭，获得准确的性能数据

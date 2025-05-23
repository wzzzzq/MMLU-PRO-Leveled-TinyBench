# MMLU-PRO-Leveled-TinyBench 测评框架


## 项目目标 🎯
本仓库旨在通过分层基准测试，验证不同API提供商声称的"满血版"Deepseek模型（即未进行性能降级的完整版本）是否真实可信。核心功能包括：
1. **难度分级测试**：通过`MMLU-PRO-Leveled`子集，量化模型在10个难度层级的准确率差异
2. **学科专项测评**：即将推出的`Subject-Specific`子集将按学科（如化学、物理等）划分问题，帮助选择最优领域模型
3. **供应商对比**：支持横向对比不同API服务商提供的同源模型性能表现

## 数据集
https://huggingface.co/datasets/wzzzq/MMLU-PRO-Leveled-TinyBench

## 快速开始 🚀
### 环境配置
```bash
git clone https://github.com/wzzzzq/MMLU-PRO-Leveled-TinyBench
pip install -r requirements.txt
```

### API密钥设置
在`.env`中配置各平台密钥：
```
SILICONFLOW_API_KEY=...
VOLCANO_API_KEY=...
GLM_API_KEY=...
```

### 运行测试
```python
python main.py \
  --num 50      # 每个难度级别的抽样题量
```
结果将生成于`results/`目录，含json统计文件和markdown准确率表格

## 结果解析 📊

# Model Performance Across Difficulty Levels

| Model | Combined Accuracy | Extremely Hard (0.0-0.1) | Very Hard (0.1-0.2) | Hard (0.2-0.3) | Moderately Hard (0.3-0.4) | Intermediate (0.4-0.5) | Medium (0.5-0.6) | Moderately Easy (0.6-0.7) | Easy (0.7-0.8) | Very Easy (0.8-0.9) | Extremely Easy (0.9-1.0) |
|-------|------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| DeepSeek-R1 Official | **71.00%** | 10.00% | 50.00% | 60.00% | 60.00% | 70.00% | 80.00% | 90.00% | 90.00% | 100.00% | 100.00% |
| 火山方舟DeepSeek-R1 | **59.00%** | 6.00% | 36.00% | 36.00% | 44.00% | 68.00% | 62.00% | 68.00% | 90.00% | 84.00% | 96.00% |
| 火山方舟DeepSeek-V3 | **56.80%** | 2.00% | 14.00% | 32.00% | 42.00% | 46.00% | 66.00% | 82.00% | 90.00% | 94.00% | 100.00% |
| DeepSeek-V3 Official | **54.00%** | 0.00% | 12.00% | 32.00% | 38.00% | 40.00% | 58.00% | 78.00% | 88.00% | 94.00% | 100.00% |
| GLM-4-PLUS | **51.00%** | 4.00% | 20.00% | 26.00% | 36.00% | 48.00% | 46.00% | 66.00% | 82.00% | 88.00% | 94.00% |
| GLM-4-Air | **44.00%** | 4.00% | 16.00% | 28.00% | 20.00% | 22.00% | 38.00% | 60.00% | 76.00% | 86.00% | 90.00% |

⚠️ 典型异常表现：若某供应商在专业级题目准确率显著低于同类服务商（如>15%差距），可能暗示模型被降级

## 路线图 🗺️
- [x] 基础难度分级测试框架
- [ ] 学科专项测评模块（预计2024.Q3完成）
- [ ] 增加GPT-4/Claude等对照组模型
- [ ] 自动化结果分析报告生成


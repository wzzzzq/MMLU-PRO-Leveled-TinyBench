# MMLU-PRO-Leveled-TinyBench 测评框架


## 项目目标 🎯
本仓库旨在通过分层基准测试，验证不同API提供商声称的"满血版"Deepseek模型（即未进行性能降级的完整版本）是否真实可信。核心功能包括：
1. **难度分级测试**：通过`MMLU-PRO-Leveled`子集，量化模型在10个难度层级的准确率差异
2. **学科专项测评**：即将推出的`Subject-Specific`子集将按学科（如化学、物理等）划分问题，帮助选择最优领域模型
3. **供应商对比**：支持横向对比不同API服务商提供的同源模型性能表现

## 数据集结构 📂


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
python benchmark.py \
  --provider deepseek \  # 可选供应商
  --level college \      # 指定难度层级
  --num_samples 50      # 抽样题量
```
结果将生成于`results/`目录，含json统计文件和markdown准确率表格

## 结果解析 📊
示例输出（`result_sample.csv`）：
| 模型供应商 | 初中正确率 | 高中正确率 | 大学正确率 | 专业正确率 |
|------------|------------|------------|------------|------------|
| 供应商A    | 82%        | 75%        | 68%        | 52%        |
| 供应商B    | 79%        | 73%        | 62%        | 47%        |

⚠️ 典型异常表现：若某供应商在专业级题目准确率显著低于同类服务商（如>15%差距），可能暗示模型被降级

## 路线图 🗺️
- [x] 基础难度分级测试框架
- [ ] 学科专项测评模块（预计2024.Q3完成）
- [ ] 增加GPT-4/Claude等对照组模型
- [ ] 自动化结果分析报告生成


## 许可协议 ©️
MIT License
```

---

<details>
<summary>English Version (Coming Soon)</summary>
Work in progress...
</details>
```
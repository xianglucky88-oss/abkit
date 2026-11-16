# abkit

标准库实现的 A/B 实验统计推断工具，当前是可运行的半成品：分流和两组均值检验较完整，其余模块提供最小可用实现，仍保留统计口径与性能方面的题目空间。

```bash
make test
make demo
make verify
```

模块：`assignment`、`variance`、`sequential`、`multiple`、`segment`、`power`、`report`、`simulator`。

## 多指标校正报告

一个实验同时看转化率、收入、留存时，用 `report.summarize_metrics` 生成结构化报告，
JSON（`to_json`）与 Markdown（`markdown`）渲染自同一个 dict，字段保证一致：

```python
from abkit.report import markdown, summarize_metrics, to_json

entries = [
    {"metric": "conversion", "p_value": 0.005, "estimate": 0.012},
    {"metric": "revenue",    "p_value": 0.02,  "estimate": 0.8},
    {"metric": "retention",  "p_value": 0.03,  "estimate": 0.03},
    {"metric": "latency",    "p_value": None},          # 缺失/非法 p 值会被保留并注明原因
]
report = summarize_metrics(
    entries,
    method="holm",             # "bh" 或 "holm"
    alpha=0.05,
    key_metrics=["conversion", "revenue"],
    decision_rule="all_key",   # "any"=至少一个显著；"all_key"=所有关键指标通过
)
print(to_json(report))         # 严格 JSON，NaN 不泄漏
print(markdown(report))        # 同一 dict 渲染，含排名、校正 p、并列组与结论翻转
```

报告保留输入顺序与指标身份、校正方法与 alpha、每个指标的排名/校正后 p/拒绝决策、
并列 p 值的输入序打破规则、缺失或非法 p 值的原因，以及哪些指标在校正后改变结论
（`lost_significance` / `gained_significance`），审阅者可从报告反推完整决策依据。
`multiple.bh` / `multiple.holm` 的布尔数组接口保持向后兼容。


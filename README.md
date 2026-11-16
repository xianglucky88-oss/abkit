# abkit

标准库实现的 A/B 实验统计推断工具，当前是可运行的半成品：分流和两组均值检验较完整，其余模块提供最小可用实现，仍保留统计口径与性能方面的题目空间。

```bash
make test       # 单元测试
make demo       # 一次合成实验：固定 horizon + 序贯查看
make verify     # 统计输出断言（含序贯校准验收）
make calibrate  # 批量校准报告
```

模块：`assignment`、`variance`、`sequential`、`multiple`、`segment`、`power`、`report`、`simulator`。

## 序贯分析与校准

实验团队在数据积累到不同阶段查看结果时，不能每次都用固定 alpha 的单次
检验（5 次等距查看、每次 0.05 时真实一类错误约 0.142）。`sequential`
模块把信息比例、查看次数、早停原因和最终结论连成可复现的流程：

- `alpha_spending(t, alpha, family)`：Lan-DeMets 消耗函数，给出信息比例
  `t` 处累计可消耗的 alpha；`family` 取 `obrien_fleming`（默认）、
  `pocock`、`linear`，均满足 `spent(1) == alpha`。
- `GroupSequentialDesign(looks, alpha, spending)`：由消耗函数经递归数值
  积分求出每次查看的双侧临界值，零假设下累计越界概率恰好等于累计消耗
  的 alpha；`spending="unadjusted"` 是未校正对照设计，其 `spent` 为真实
  累计一类错误（> alpha），用于演示过早查看的代价。
- `monitor(design, control, treatment)`：逐次查看累计样本——未越界继续
  观察（`continue`），越界以 `efficacy` 早停，达到查看上限仍未显著以
  `max_looks` 收尾并给出 `inconclusive`；空样本自然走到 `max_looks`，
  不产生 NaN。

`simulator` 在已知真值下批量评估该流程：`batch_evaluate` 覆盖零效应
（一类错误）、正/负效应（检出率与方向）、小样本（稳健性）四个场景，
种子按 `(主种子, 场景, 重复序号)` 确定性派生，重复运行结果完全一致；
汇总含拒绝率及 Wilson 置信区间、平均查看次数、早停率与停止原因分布。

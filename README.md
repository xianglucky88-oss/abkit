# abkit

标准库实现的 A/B 实验统计推断工具，当前是可运行的半成品：分流和两组均值检验较完整，其余模块提供最小可用实现，仍保留统计口径与性能方面的题目空间。

```bash
make test
make demo
make verify
```

模块：`assignment`、`variance`、`sequential`、`multiple`、`segment`、`power`、`planning`、`report`、`simulator`。

`planning` 把方差缩减（CUPED / 比率 delta 方法）与功效计算串成一条口径一致的规划流程：调整前后的效应、标准误与每臂样本量在同一份报告中对照呈现，推荐样本量显式标注依赖的是调整后还是未调整的不确定性（`sample_size.basis` / `basis_detail`）；分母全零、效应为零、样本不足等退化情形以 `status` + `reason` 返回可解释结果。

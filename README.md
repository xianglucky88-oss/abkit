# abkit

标准库实现的 A/B 实验统计推断工具。分流与 SRM（样本比例失衡）检测支持任意
组数与指定期望比例，其余模块提供最小可用实现。

```bash
make test     # 单元测试（含多臂 SRM 全套验收）
make demo     # 多臂不等比例分流与告警演示
make verify   # 端到端断言
```

模块：`assignment`、`variance`、`sequential`、`multiple`、`segment`、`power`、`report`、`simulator`。

## 多臂分流与 SRM 检测（`assignment`）

### 分流

- `bucket(user_id, salt, buckets)`：底层稳定哈希原语，SHA-256 取模；
  salt、桶数、用户 ID 不变时落桶确定，桶数范围 `[1, 2^32]`。
- `arm_bucket(user_id, salt, buckets, expected_ratio)`：不等比例分流。
  等比例时与 `bucket` 完全一致；不等比例时先哈希到 100 万个虚拟槽，再按
  归一化后的比例做确定性槽区间映射，配置不变则同一用户永远落入同一臂。
- `assign(...)` / `assign_detail(...)`：批量分桶，默认对重复用户 ID 去重，
  返回的计数始终包含全部桶（零样本桶计数为 0），并附带重复数与逐用户映射。
- `normalize_ratios(...)`：比例归一化，接受整数权重或非负比例；长度不匹配、
  全零、负数、NaN/Inf 一律按配置错误拒绝。

### SRM 检验

- `srm_p_value(counts, expected_ratio=None)`：皮尔逊卡方拟合优度，
  支持任意组数（两组等比例时与双侧 z 检验数学等价），零概率臂进样直接判 p=0。
- `srm_report(...)` / `run_srm_check(...)`：从分桶计数到最终告警的完整诊断，
  结论分四类，并逐臂给出观测/期望频数、份额与调整后标准化残差：

| status | 含义 |
| --- | --- |
| `ok` | 检验不显著，分流正常 |
| `imbalance` | 分流异常，`deviating_arms` 定位具体偏离的臂（残差 ≥ 3.29） |
| `insufficient_sample` | 存在零样本桶或正概率臂期望频数 < 5，卡方近似不可靠 |
| `config_error` | 期望比例/计数输入非法（长度不匹配、全零、零概率臂进样等） |

`run_srm_check` 串起「用户 ID → 稳定分桶 → 卡方检验 → 可解释告警」全流程，
配置错误以报告状态返回而不是抛异常，方便实验平台直接展示给负责人。

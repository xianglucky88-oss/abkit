"""演示：多臂不等比例分流 + SRM 端到端诊断。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abkit.assignment import run_srm_check, srm_report
from abkit.report import markdown, summarize
from abkit.simulator import simulate
from abkit.stats import welch_t

# 1) 传统两组均值检验（保留原演示）
control, treatment = simulate()
t_stat, p_value = welch_t(treatment, control)
estimate = sum(treatment) / len(treatment) - sum(control) / len(control)
print(markdown(summarize("treatment mean difference", estimate, p_value)))
print(f"t-statistic: {t_stat:.4f}\n")

# 2) 四臂不等比例（1:2:3:4）健康分流
users = [f"user-{i}" for i in range(8000)]
healthy = run_srm_check(users, salt="launch-2026", buckets=4,
                        expected_ratio=[1, 2, 3, 4])
print(f"[健康实验] {healthy['alert']}")
for arm in healthy["per_arm"]:
    print(f"  臂 {arm['arm']}: 观测 {arm['observed']:5d}  "
          f"期望 {arm['expected']:8.1f}  份额 {arm['observed_ratio']:.3f} "
          f"(目标 {arm['expected_ratio']:.2f})")

# 3) 5 臂实验中第 2 号臂被污染 -> 应定位到具体臂
print()
tampered = srm_report({0: 2500, 1: 2500, 2: 3200, 3: 2500, 4: 2500})
print(f"[污染实验] {tampered['alert']}")
print(f"  状态={tampered['status']}  偏离臂={tampered['deviating_arms']}  "
      f"p={tampered['p_value']:.2e}")

# 4) 样本不足与配置错误的区分
print()
print(f"[样本不足] {srm_report({0: 50, 1: 0, 2: 50})['alert']}")
print(f"[配置错误] {srm_report({0: 100, 1: 3, 2: 97}, [0.5, 0.0, 0.5])['alert']}")

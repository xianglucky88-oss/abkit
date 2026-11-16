import random

from abkit.planning import plan_mean_metric, plan_ratio_metric
from abkit.report import markdown, plan_markdown, summarize
from abkit.simulator import simulate
from abkit.stats import welch_t

control, treatment = simulate()
t_stat, p_value = welch_t(treatment, control)
estimate = sum(treatment) / len(treatment) - sum(control) / len(control)
print(markdown(summarize("treatment mean difference", estimate, p_value)))
print(f"t-statistic: {t_stat:.4f}")

print("\n" + "=" * 60 + "\n")

# 规划流程：CUPED 方差缩减 → 调整后不确定性 → 样本量建议
rng = random.Random(11)
n = 300
x_c = [rng.gauss(0.0, 1.0) for _ in range(n)]
x_t = [rng.gauss(0.0, 1.0) for _ in range(n)]
y_c = [0.9 * x + rng.gauss(0.0, 0.4) for x in x_c]
y_t = [0.15 + 0.9 * x + rng.gauss(0.0, 0.4) for x in x_t]
print(plan_markdown(plan_mean_metric(y_c, y_t, x_c, x_t)))

print("\n" + "=" * 60 + "\n")

# 比率指标：delta 方法 → 样本量建议；含一个分母全零的对照例
den_c = [rng.gauss(8.0, 1.5) for _ in range(n)]
den_t = [rng.gauss(8.0, 1.5) for _ in range(n)]
num_c = [0.5 * d + rng.gauss(0.0, 0.3) for d in den_c]
num_t = [0.55 * d + rng.gauss(0.0, 0.3) for d in den_t]
print(plan_markdown(plan_ratio_metric(num_c, den_c, num_t, den_t)))

print("\n" + "=" * 60 + "\n")
print(plan_markdown(plan_ratio_metric([1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
                                      [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])))

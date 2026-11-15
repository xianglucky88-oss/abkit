from abkit.report import markdown, summarize
from abkit.simulator import simulate
from abkit.stats import welch_t

control, treatment = simulate()
t_stat, p_value = welch_t(treatment, control)
estimate = sum(treatment) / len(treatment) - sum(control) / len(control)
print(markdown(summarize("treatment mean difference", estimate, p_value)))
print(f"t-statistic: {t_stat:.4f}")

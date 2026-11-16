from abkit.report import markdown, summarize
from abkit.sequential import GroupSequentialDesign, monitor
from abkit.simulator import simulate
from abkit.stats import welch_t

control, treatment = simulate(n_per_arm=400, effect=0.25, seed="demo")

# 固定 horizon（只在实验结束时看一次）
t_stat, p_value = welch_t(treatment, control)
estimate = sum(treatment) / len(treatment) - sum(control) / len(control)
print(markdown(summarize("treatment mean difference", estimate, p_value)))
print(f"t-statistic: {t_stat:.4f}\n")

# 序贯流程：数据积累过程中分 5 次查看，alpha 消耗控制总一类错误
design = GroupSequentialDesign(looks=5, alpha=0.05)
report = monitor(design, control, treatment)
print(f"序贯设计: {design.spending}, {design.looks} 次查看, alpha={design.alpha}")
for record in report.records:
    print(f"  第{record.look}次查看 t={record.info_fraction:.1f} "
          f"n={record.n_control}+{record.n_treatment} "
          f"z={record.z:+.3f} 临界值={record.critical:.3f} -> {record.reason}")
print(f"结论: {report.conclusion}（第{report.looks_used}次查看，原因={report.stop_reason}）")

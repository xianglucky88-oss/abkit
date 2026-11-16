"""序贯设计的批量校准报告：零效应 / 正效应 / 负效应 / 小样本。

对照同一批数据上的两种设计：
- obrien_fleming：Lan-DeMets 消耗函数推出的序贯边界，总一类错误恰为 alpha；
- unadjusted：每次查看都套用完整 alpha（未校正），用于演示过早查看的膨胀。

重复运行结果完全一致（种子按场景与重复序号确定性派生）。
"""

from abkit.sequential import GroupSequentialDesign
from abkit.simulator import batch_evaluate

REPETITIONS = 800
N_PER_ARM = 400


def report(design):
    return batch_evaluate(repetitions=REPETITIONS, n_per_arm=N_PER_ARM,
                          design=design, seed="calibrate")


def main():
    designs = {
        "obrien_fleming": GroupSequentialDesign(looks=5, alpha=0.05),
        "unadjusted": GroupSequentialDesign(looks=5, alpha=0.05, spending="unadjusted"),
    }
    for name, design in designs.items():
        print(f"== 设计: {name} (looks={design.looks}, alpha={design.alpha}) ==")
        print(f"   临界值: {[round(b, 3) for b in design.boundaries_]}")
        print(f"   累计一类错误: {[round(s, 4) for s in design.spent]}")
        for scenario, r in report(design).items():
            lo, hi = r["rejection_ci"]
            print(f"   {scenario:<12} 真值={r['true_effect']:+.2f} "
                  f"拒绝率={r['rejection_rate']:.3f} CI=[{lo:.3f},{hi:.3f}] "
                  f"平均查看={r['mean_looks_used']:.2f} 早停率={r['early_stop_rate']:.3f} "
                  f"方向正确率={r['direction_correct_rate']:.3f}")
        print()


if __name__ == "__main__":
    main()

#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPYCACHEPREFIX=/tmp/abkit-pycache python3 - <<'PY'
from abkit.assignment import assign, srm_p_value
from abkit.multiple import bh, holm
from abkit.power import detectable_effect, required_sample_size
from abkit.report import summarize
from abkit.sequential import GroupSequentialDesign, alpha_spending, monitor
from abkit.simulator import batch_evaluate, calibration, run_scenario, simulate
from abkit.variance import cuped

checks = []


def check(name, ok, detail):
    checks.append((name, ok, detail))


counts = assign([f"u{i}" for i in range(1000)], "verify")
check("稳定分桶", sum(counts.values()) == 1000, str(counts))
p_balanced = srm_p_value({0: 500, 1: 500})
check("平衡样本不报警", p_balanced > 0.05, str(p_balanced))
adjusted, theta = cuped([1, 2, 3, 4], [1, 2, 3, 4])
check("CUPED 输出", len(adjusted) == 4 and theta > 0, f"theta={theta}")
check("多重比较", bh([0.001, 0.04, 0.9]) == [True, False, False], str(bh([0.001, 0.04, 0.9])))
check("Holm 校正", holm([0.001, 0.04, 0.9]) == [True, False, False], str(holm([0.001, 0.04, 0.9])))
n = required_sample_size(0.2)
check("功效分析", n > 0 and detectable_effect(n) <= 0.2, f"n={n}, detectable={detectable_effect(n)}")
cal = calibration(10)
check("合成校准", 0.0 <= cal["rejection_rate"] <= 1.0, str(cal))
check("结论报告", summarize("metric", 1.0, 0.01)["significant"], "p=0.01")

# ---- 序贯设计与校准验收 ----
obf = GroupSequentialDesign(looks=5, alpha=0.05)
naive = GroupSequentialDesign(looks=5, alpha=0.05, spending="unadjusted")
check("序贯边界消耗恰为 alpha",
      abs(obf.spent[-1] - 0.05) < 1e-9 and all(
          abs(s - alpha_spending(t, 0.05)) < 1e-12
          for s, t in zip(obf.spent, obf.info_fractions)),
      f"spent={['%.4f' % s for s in obf.spent]}")
check("未校正设计累计一类错误膨胀（设计口径）",
      0.10 < naive.spent[-1] < 0.20, f"naive spent={naive.spent[-1]:.4f}")

cal_obf = batch_evaluate(repetitions=800, n_per_arm=400, design=obf, seed="verify")
cal_naive = batch_evaluate(repetitions=800, n_per_arm=400, design=naive, seed="verify")
null_obf, null_naive = cal_obf["null"], cal_naive["null"]
check("OBF 零效应一类错误受控（模拟口径）",
      null_obf["rejection_rate"] <= 0.08,
      f"rate={null_obf['rejection_rate']:.3f} CI={['%.3f' % x for x in null_obf['rejection_ci']]}")
check("过早查看造成错误率膨胀（模拟口径）",
      null_naive["rejection_rate"] >= 0.10,
      f"rate={null_naive['rejection_rate']:.3f} CI={['%.3f' % x for x in null_naive['rejection_ci']]}")
check("正效应检出率", cal_obf["positive"]["rejection_rate"] >= 0.6,
      f"power={cal_obf['positive']['rejection_rate']:.3f}")
check("负效应方向正确",
      cal_obf["negative"]["rejection_rate"] >= 0.6
      and cal_obf["negative"]["direction_correct_rate"] >= 0.95,
      f"power={cal_obf['negative']['rejection_rate']:.3f} "
      f"direction={cal_obf['negative']['direction_correct_rate']:.3f}")
small = cal_obf["small_sample"]
check("小样本场景稳健",
      0.0 <= small["rejection_rate"] <= 1.0 and 1.0 <= small["mean_looks_used"] <= 5.0
      and small["repetitions"] == 800,
      f"rate={small['rejection_rate']:.3f} looks={small['mean_looks_used']:.2f}")
check("早停原因与结论关联",
      set(null_obf["stop_reasons"]) <= {"efficacy", "max_looks"}
      and sum(null_obf["stop_reasons"].values()) == 800,
      str(null_obf["stop_reasons"]))
check("校准结果可重复",
      batch_evaluate(repetitions=200, n_per_arm=400, design=obf, seed="verify")
      == batch_evaluate(repetitions=200, n_per_arm=400, design=obf, seed="verify"),
      "相同种子两次批量评估完全一致")
empty_reps = run_scenario("null", 0.0, repetitions=0, design=obf)
empty_data = run_scenario("null", 0.0, repetitions=3, n_per_arm=0, design=obf, seed="empty")
check("空样本不失真",
      empty_reps["rejection_rate"] == 0.0 and empty_reps["rejection_ci"] == (0.0, 1.0)
      and empty_data["rejection_rate"] == 0.0
      and empty_data["stop_reasons"] == {"max_looks": 3},
      f"reps=0 -> {empty_reps['rejection_rate']}, n=0 -> {empty_data['stop_reasons']}")
report = monitor(obf, *simulate(400, 0.0, "verify:monitor"))
check("逐次查看记录完整",
      len(report.records) == report.looks_used
      and all(r.info_fraction > 0 and r.critical > 0 and r.reason for r in report.records)
      and report.records[-1].stop,
      f"looks_used={report.looks_used} reason={report.stop_reason}")

failed = sum(not ok for _, ok, _ in checks)
print("== 断言 ==")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print(f"\n结果: {len(checks)-failed}/{len(checks)} 通过")
raise SystemExit(1 if failed else 0)
PY

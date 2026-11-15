#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPYCACHEPREFIX=/tmp/abkit-pycache python3 - <<'PY'
from abkit.assignment import assign, srm_p_value
from abkit.multiple import bh, holm
from abkit.power import detectable_effect, required_sample_size
from abkit.report import summarize
from abkit.simulator import calibration
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

failed = sum(not ok for _, ok, _ in checks)
print("== 断言 ==")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print(f"\n结果: {len(checks)-failed}/{len(checks)} 通过")
raise SystemExit(1 if failed else 0)
PY

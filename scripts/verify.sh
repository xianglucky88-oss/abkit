#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPYCACHEPREFIX=/tmp/abkit-pycache python3 - <<'PY'
from abkit.assignment import assign, srm_p_value
from abkit.multiple import bh, holm
from abkit.power import detectable_effect, required_sample_size
from abkit.report import summarize
from abkit.segment import group_means, interaction_effect
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

seg_rows = [
    {"g": "a", "arm": "c", "m": 1.0}, {"g": "a", "arm": "c", "m": 2.0},
    {"g": "a", "arm": "t", "m": 2.0}, {"g": "a", "arm": "t", "m": 3.0},
    {"g": "b", "arm": "c", "m": 1.0}, {"g": "b", "arm": "c", "m": 2.0},
]
seg_result = interaction_effect(seg_rows, "g", "arm", "m")
seg_status = {s["segment"]: s["status"] for s in seg_result["segments"]}
check("分层状态", seg_status == {"a": "ok", "b": "not_estimable"}, str(seg_status))
seg_effect = seg_result["segments"][0]["contrasts"]["t"]["effect"]
check("分层效应", seg_effect == 1.0 and seg_result["segments"][1]["contrasts"]["t"]["effect"] is None,
      f"effect={seg_effect}")
check("分层均值", group_means(seg_rows, "g", "m")["a"] == {"n": 4, "mean": 2.0},
      str(group_means(seg_rows, "g", "m")))

failed = sum(not ok for _, ok, _ in checks)
print("== 断言 ==")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print(f"\n结果: {len(checks)-failed}/{len(checks)} 通过")
raise SystemExit(1 if failed else 0)
PY

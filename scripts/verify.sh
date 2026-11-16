#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPYCACHEPREFIX=/tmp/abkit-pycache python3 - <<'PY'
import json

from abkit.assignment import assign, srm_p_value
from abkit.multiple import bh, correct, holm
from abkit.power import detectable_effect, required_sample_size
from abkit.report import markdown, summarize, summarize_metrics, to_json
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

# Dataset A: one missing p-value; correction runs on the 3 valid metrics only.
entries = [
    {"metric": "conversion", "p_value": 0.005},
    {"metric": "revenue", "p_value": 0.02},
    {"metric": "retention", "p_value": 0.03},
    {"metric": "latency", "p_value": None},
]
multi = summarize_metrics(
    entries, method="holm",
    key_metrics=["conversion", "revenue", "latency"],
    decision_rule="all_key",
)
json.loads(to_json(multi))  # strict JSON round-trip, NaN must not leak
check(
    "多指标报告-输入顺序与身份",
    multi["input_order"] == ["conversion", "revenue", "retention", "latency"]
    and [row["input_order"] for row in multi["metrics"]] == [0, 1, 2, 3],
    str(multi["input_order"]),
)
check("多指标报告-缺失p值", multi["counts"]["invalid"] == 1
      and multi["metrics"][3]["reason"] == "missing p-value"
      and multi["metrics"][3]["adjusted_p_value"] is None,
      str(multi["counts"]))
# The invalid key metric makes the strict all_key rule fail.
check("多指标报告-两种决策规则",
      multi["decision_rules"]["any"]["satisfied"]
      and not multi["decision_rules"]["all_key"]["satisfied"]
      and not multi["decision"],
      str(multi["decision_rules"]))

# Dataset B: four valid metrics, BH (FDR) vs Holm (FWER) disagree.
valid_entries = [
    {"metric": "conversion", "p_value": 0.005},
    {"metric": "revenue", "p_value": 0.02},
    {"metric": "retention", "p_value": 0.03},
    {"metric": "latency", "p_value": 0.08},
]
bh_multi = summarize_metrics(valid_entries, method="bh")
holm_multi = summarize_metrics(valid_entries, method="holm")
check("校正方法结论差异",
      bh_multi["rejected_metrics"] == ["conversion", "revenue", "retention"]
      and holm_multi["rejected_metrics"] == ["conversion"],
      f"bh={bh_multi['rejected_metrics']}, holm={holm_multi['rejected_metrics']}")
check("多指标报告-结论可追溯",
      {c["metric"] for c in holm_multi["changed_metrics"]}
      == {"revenue", "retention"},
      str(holm_multi["changed_metrics"]))
tied = correct([0.01, 0.01], method="bh")
check("并列p值排序", tied["tie_groups"] == [[0, 1]]
      and [i["rank"] for i in tied["items"]] == [1, 2], str(tied["tie_groups"]))
check("Markdown与JSON同源",
      markdown(holm_multi).count("conversion") >= 1 and "HOLM" in markdown(holm_multi)
      and "lost_significance" in markdown(holm_multi),
      "rendered from same dict")

failed = sum(not ok for _, ok, _ in checks)
print("== 断言 ==")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print(f"\n结果: {len(checks)-failed}/{len(checks)} 通过")
raise SystemExit(1 if failed else 0)
PY

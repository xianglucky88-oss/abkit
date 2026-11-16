#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHONPYCACHEPREFIX=/tmp/abkit-pycache python3 - <<'PY'
from abkit.assignment import (
    arm_bucket,
    assign,
    normalize_ratios,
    run_srm_check,
    srm_p_value,
    srm_report,
)
from abkit.multiple import bh, holm
from abkit.power import detectable_effect, required_sample_size
from abkit.report import summarize
from abkit.simulator import calibration
from abkit.stats import chi2_sf
from abkit.variance import cuped

checks = []


def check(name, ok, detail):
    checks.append((name, ok, detail))


# ---- 既有两组能力保持兼容 ----
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

# ---- 多臂 + 不等比例：完整流程 ----
check("比例归一化", normalize_ratios([1, 2, 1], 3) == [0.25, 0.5, 0.25], "1:2:1 归一化")
check(
    "卡方实现与 z 检验一致",
    abs(chi2_sf(1.96 ** 2, 1) - 0.049957) < 1e-4,
    str(chi2_sf(1.96 ** 2, 1)),
)

ratio = [1, 2, 3, 4]
healthy = run_srm_check([f"u{i}" for i in range(8000)], "launch", 4, ratio)
check("四臂不等比例健康", healthy["status"] == "ok", healthy["alert"])
shares = [e["observed_ratio"] for e in healthy["per_arm"]]
check(
    "实测份额接近期望比例",
    all(abs(o - e) < 0.03 for o, e in zip(shares, normalize_ratios(ratio, 4))),
    str([round(s, 3) for s in shares]),
)

# 稳定性：salt/桶数/配置不变，重复运行逐用户落桶一致
users = [f"stable-{i}" for i in range(2000)]
a = run_srm_check(users, "fixed-salt", 5, [3, 1, 2, 1, 3])
b = run_srm_check(users, "fixed-salt", 5, [3, 1, 2, 1, 3])
check("同配置稳定落桶", a["assignments"] == b["assignments"], "逐用户映射一致")
check("等比例与历史原语一致",
      all(arm_bucket(u, "s", 3, [1, 1, 1]) == arm_bucket(u, "s", 3) for u in users[:50]),
      "退化路径一致")

# 偏离发生在后段臂（前两组完全平衡），旧的两组 z 检验必然漏报
tampered = {0: 2500, 1: 2500, 2: 3200, 3: 2500, 4: 2500}
report = srm_report(tampered)
check("多臂失衡可检出", report["status"] == "imbalance", report["alert"])
check("定位到具体偏离臂", report["deviating_arms"] == [2], str(report["deviating_arms"]))
legacy_p = srm_p_value({0: 2500, 1: 2500})
check("旧两组口径会漏报", legacy_p > 0.05, f"前两组 p={legacy_p:.3f}")

# 8 臂污染：用户从 6 号臂挪到 3 号臂，应同时定位两个臂且残差符号正确
salt = "eight-arm"
assignments = {u: arm_bucket(u, salt, 8) for u in [f"u{i}" for i in range(16000)]}
counts8 = {i: 0 for i in range(8)}
moved = 0
for uid, arm in assignments.items():
    if arm == 6 and moved < 300:
        arm, moved = 3, moved + 1
    counts8[arm] += 1
report8 = srm_report(counts8)
res = {e["arm"]: e["adjusted_residual"] for e in report8["per_arm"]}
check("8 臂污染定位", set(report8["deviating_arms"]) == {3, 6}, str(report8["deviating_arms"]))
check("残差方向可解释", res[3] > 0 > res[6], f"r3={res[3]:.2f}, r6={res[6]:.2f}")

# 三类结论区分
check("样本不足识别", srm_report({0: 50, 1: 0, 2: 50})["status"] == "insufficient_sample",
      "零样本桶")
check("稀疏期望识别",
      srm_report({0: 99, 1: 1}, [0.99, 0.01])["status"] == "insufficient_sample",
      "期望频数 < 5")
check("配置错误-比例长度",
      run_srm_check(users, "s", 3, [1, 1])["status"] == "config_error", "长度不匹配")
check("配置错误-零概率臂进样",
      srm_report({0: 100, 1: 3, 2: 97}, [0.5, 0.0, 0.5])["status"] == "config_error",
      "零概率臂出现样本")
check("配置错误-全零比例",
      srm_report({0: 5, 1: 5}, [0, 0])["status"] == "config_error", "全零比例")

# 重复用户 ID 去重
dup = run_srm_check([f"u{i}" for i in range(1000)] + [f"u{i}" for i in range(200)], "d", 2)
check("重复 ID 去重", dup["duplicates"] == 200 and dup["unique_users"] == 1000,
      f"dup={dup['duplicates']}")

# 极端桶数
check("极端桶数-样本不足", run_srm_check([f"u{i}" for i in range(20)], "s", 20)["status"]
      == "insufficient_sample", "20 桶 20 人")
check("极端桶数-可运行", run_srm_check([f"u{i}" for i in range(20000)], "s", 100)["buckets"] == 100,
      "100 桶 20000 人")
try:
    arm_bucket("u", "s", 2 ** 32 + 1)
    check("极端桶数-上限拦截", False, "未抛异常")
except ValueError:
    check("极端桶数-上限拦截", True, f"上限 2^32")

failed = sum(not ok for _, ok, _ in checks)
print("== 断言 ==")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print(f"\n结果: {len(checks)-failed}/{len(checks)} 通过")
raise SystemExit(1 if failed else 0)
PY

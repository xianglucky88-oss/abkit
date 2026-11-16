"""合成实验与序贯策略的批量校准。

模拟器在已知真值下重复生成实验，用来验证早停策略是否维持预期的一类
错误（零效应）、检出率（正/负效应）以及小样本下的稳健性。每次重复的
种子由 (主种子, 场景名, 重复序号) 确定性地派生，因此重复运行、调整
场景顺序都不会改变结果；空重复或空样本返回定义好的零值，不产生 NaN。
"""

from __future__ import annotations

import random

from .sequential import GroupSequentialDesign, monitor
from .stats import wilson_interval

# 内置评估场景：零效应（一类错误）、正/负效应（检出率与方向）、小样本（稳健性）
SCENARIOS: dict[str, dict[str, float | int]] = {
    "null": {"effect": 0.0},
    "positive": {"effect": 0.2},
    "negative": {"effect": -0.2},
    "small_sample": {"effect": 0.2, "n_per_arm": 40},
}


def simulate(n_per_arm: int = 200, effect: float = 0.2,
             seed: int | str = 7) -> tuple[list[float], list[float]]:
    """生成一次实验的两臂样本；相同 seed 永远得到相同样本。"""
    rng = random.Random(seed)
    control = [rng.gauss(0.0, 1.0) for _ in range(max(0, n_per_arm))]
    treatment = [rng.gauss(effect, 1.0) for _ in range(max(0, n_per_arm))]
    return control, treatment


def run_scenario(name: str, effect: float, repetitions: int = 500,
                 n_per_arm: int = 400, design: GroupSequentialDesign | None = None,
                 seed: int | str = "abkit") -> dict:
    """在已知真值 effect 下重复跑序贯实验并汇总。

    返回拒绝率（含 Wilson 置信区间）、平均查看次数、早停率、停止原因
    分布、平均估计与显著时的方向正确率。repetitions<=0 或 n_per_arm<=0
    时返回零值汇总，便于批量流程安全运行。
    """
    design = design or GroupSequentialDesign()
    reps = max(0, int(repetitions))
    hits = 0
    looks_used = 0
    estimate_sum = 0.0
    direction_correct = 0
    stop_reasons: dict[str, int] = {}
    for rep in range(reps):
        control, treatment = simulate(n_per_arm, effect, f"{seed}:{name}:{rep}")
        report = monitor(design, control, treatment)
        hits += report.significant
        looks_used += report.looks_used
        estimate_sum += report.estimate
        stop_reasons[report.stop_reason] = stop_reasons.get(report.stop_reason, 0) + 1
        if report.significant and effect != 0 and (report.z > 0) == (effect > 0):
            direction_correct += 1
    return {
        "scenario": name,
        "true_effect": effect,
        "repetitions": reps,
        "n_per_arm": n_per_arm,
        "looks": design.looks,
        "spending": design.spending,
        "rejections": hits,
        "rejection_rate": hits / reps if reps else 0.0,
        "rejection_ci": wilson_interval(hits, reps),
        "mean_looks_used": looks_used / reps if reps else 0.0,
        "early_stop_rate": stop_reasons.get("efficacy", 0) / reps if reps else 0.0,
        "stop_reasons": stop_reasons,
        "mean_estimate": estimate_sum / reps if reps else 0.0,
        "direction_correct_rate": direction_correct / hits if hits else 0.0,
    }


def batch_evaluate(scenarios: dict[str, dict] | None = None,
                   repetitions: int = 500, n_per_arm: int = 400,
                   design: GroupSequentialDesign | None = None,
                   seed: int | str = "abkit") -> dict[str, dict]:
    """对一组场景批量评估同一序贯设计，返回 {场景名: 汇总}。"""
    design = design or GroupSequentialDesign()
    scenarios = SCENARIOS if scenarios is None else scenarios
    return {
        name: run_scenario(name, spec["effect"], repetitions,
                           spec.get("n_per_arm", n_per_arm), design, seed)
        for name, spec in scenarios.items()
    }


def calibration(repetitions: int = 20, n_per_arm: int = 200, effect: float = 0.2,
                seed: int | str = "abkit:calibration") -> dict[str, float]:
    """固定 horizon 校准（向后兼容接口，等价于单次查看的序贯设计）。"""
    summary = run_scenario("fixed", effect, repetitions, n_per_arm,
                           GroupSequentialDesign(looks=1), seed)
    return {
        "rejection_rate": summary["rejection_rate"],
        "mean_estimate": summary["mean_estimate"],
        "rejection_ci": summary["rejection_ci"],
        "repetitions": summary["repetitions"],
    }

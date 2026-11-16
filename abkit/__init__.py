"""abkit —— A/B 实验的统计推断工具。

只依赖标准库。模块划分：
    assignment  分流与样本比例失衡（SRM）
    variance    方差缩减（CUPED、delta method）
    sequential  序贯检验与早停
    multiple    多重比较校正
    segment     分层与异质性分析
    power       功效与样本量
    report      结论报告
    simulator   合成实验（已知真值，用于校准）
"""

__version__ = "0.5.0"

from .multiple import bh, correct, holm
from .report import markdown, summarize, summarize_metrics, to_json

__all__ = [
    "bh",
    "holm",
    "correct",
    "summarize",
    "summarize_metrics",
    "markdown",
    "to_json",
]


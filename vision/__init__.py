
"""
Vision-Grounded Embodied Task Agent 的视觉模块。

这个包负责运行时视觉感知相关功能。

重要设计原则：
运行时感知模块不能读取仿真器中的物体 Ground Truth。
Ground Truth 只允许用于后续 evaluation 评测。
"""

from .perception import ObjectObservation, PerceptionSystem

__all__ = [
    "ObjectObservation",
    "PerceptionSystem",
]
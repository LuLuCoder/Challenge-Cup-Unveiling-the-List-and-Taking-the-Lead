"""工艺约束。"""

from satc import config


def thermal_constraint(x):
    """喷嘴温度必须在 [T_min, T_max] 内，越界量作为违规值。"""
    temperature = float(x[2])
    violation = 0.0
    if temperature < config.TEMPERATURE_MIN:
        violation += config.TEMPERATURE_MIN - temperature
    if temperature > config.TEMPERATURE_MAX:
        violation += temperature - config.TEMPERATURE_MAX
    return violation

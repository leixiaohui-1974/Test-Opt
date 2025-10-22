"""
水网优化模型包
"""

# 导入异常类（不依赖外部库）
from .exceptions import (
    WaterNetworkError,
    ConfigurationError,
    ValidationError,
    TopologyError,
    TimeSeriesError,
    SolverError,
    DataError,
)

# 条件导入（依赖pyomo等外部库）
__all__ = [
    "WaterNetworkError",
    "ConfigurationError",
    "ValidationError",
    "TopologyError",
    "TimeSeriesError",
    "SolverError",
    "DataError",
]

try:
    from .water_network_generic import build_water_network_model
    from .validation import validate_network_config
    __all__.extend(["build_water_network_model", "validate_network_config"])
except ImportError:
    # pyomo等依赖未安装时，不导入这些模块
    pass

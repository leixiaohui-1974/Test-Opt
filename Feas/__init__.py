"""
水网优化模型包
"""
from .water_network_generic import build_water_network_model
from .exceptions import (
    WaterNetworkError,
    ConfigurationError,
    ValidationError,
    TopologyError,
    TimeSeriesError,
    SolverError,
    DataError,
)
from .validation import validate_network_config

__all__ = [
    "build_water_network_model",
    "validate_network_config",
    "WaterNetworkError",
    "ConfigurationError",
    "ValidationError",
    "TopologyError",
    "TimeSeriesError",
    "SolverError",
    "DataError",
]

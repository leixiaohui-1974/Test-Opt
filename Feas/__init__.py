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
from .feasibility import (
    FeasibilityStatus,
    FeasibilityResult,
    check_solver_results,
    check_constraint_violations,
    check_problem_feasibility,
)
from .defaults import (
    OPTIMIZATION_DEFAULTS,
    MPC_DEFAULTS,
    CONTROL_EVALUATION_DEFAULTS,
    CANAL_IDZ_DEFAULTS,
    VISUALIZATION_DEFAULTS,
    get_default,
    update_defaults,
)
from .utils import (
    TimeSeriesGenerator,
    ResultExtractor,
    SolverManager,
)

__all__ = [
    "build_water_network_model",
    "validate_network_config",
    # 异常
    "WaterNetworkError",
    "ConfigurationError",
    "ValidationError",
    "TopologyError",
    "TimeSeriesError",
    "SolverError",
    "DataError",
    # 可行性检查
    "FeasibilityStatus",
    "FeasibilityResult",
    "check_solver_results",
    "check_constraint_violations",
    "check_problem_feasibility",
    # 默认配置
    "OPTIMIZATION_DEFAULTS",
    "MPC_DEFAULTS",
    "CONTROL_EVALUATION_DEFAULTS",
    "CANAL_IDZ_DEFAULTS",
    "VISUALIZATION_DEFAULTS",
    "get_default",
    "update_defaults",
    # 工具
    "TimeSeriesGenerator",
    "ResultExtractor",
    "SolverManager",
]

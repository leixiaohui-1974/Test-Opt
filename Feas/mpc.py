"""
模型预测控制（MPC）：滚动时域优化实现
"""
from typing import Dict, List, Optional, Tuple, Any
import copy
from pyomo.environ import SolverFactory, value

# 尝试相对导入，失败则使用绝对导入
try:
    from .water_network_generic import build_water_network_model
    from .exceptions import SolverError
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from water_network_generic import build_water_network_model
    from exceptions import SolverError


class MPCController:
    """
    滚动时域模型预测控制器

    实现滚动时域优化策略：
    1. 在每个时间步，使用当前状态作为初始条件
    2. 优化未来N步（预测窗口）
    3. 仅执行第一步控制动作
    4. 更新状态，滚动到下一时间步
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        prediction_horizon: int = 24,
        control_horizon: Optional[int] = None,
        solver_name: str = "glpk",
        solver_options: Optional[Dict] = None,
    ):
        """
        初始化MPC控制器

        Args:
            base_config: 基础网络配置
            prediction_horizon: 预测窗口长度（时间步数）
            control_horizon: 控制窗口长度（None则等于prediction_horizon）
            solver_name: 求解器名称
            solver_options: 求解器选项
        """
        self.base_config = copy.deepcopy(base_config)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon or prediction_horizon
        self.solver_name = solver_name
        self.solver_options = solver_options or {}

        # 初始化求解器
        self.solver = SolverFactory(solver_name)
        if not self.solver.available(exception_flag=False):
            raise SolverError(f"求解器 {solver_name} 不可用")

        # 存储历史
        self.state_history = []
        self.control_history = []
        self.current_step = 0

    def _create_mpc_config(
        self, current_states: Dict[Tuple[str, str], float], start_time_idx: int
    ) -> Dict[str, Any]:
        """
        创建MPC优化问题的配置

        Args:
            current_states: 当前状态 {(node_id, state_name): value}
            start_time_idx: 起始时间索引

        Returns:
            优化问题配置
        """
        config = copy.deepcopy(self.base_config)

        # 获取完整时间序列
        full_periods = config["horizon"]["periods"]

        # 提取预测窗口
        end_idx = min(start_time_idx + self.prediction_horizon, len(full_periods))
        mpc_periods = full_periods[start_time_idx:end_idx]

        # 更新时间范围
        config["horizon"]["periods"] = mpc_periods

        # 更新时间序列（截取对应窗口）
        for series_id, series_spec in config.get("series", {}).items():
            if "values" in series_spec:
                values = series_spec["values"]
                # 截取对应窗口的值
                mpc_values = values[start_time_idx:end_idx]
                # 如果不够，用default填充
                default = series_spec.get("default", 0.0)
                while len(mpc_values) < len(mpc_periods):
                    mpc_values.append(default)
                series_spec["values"] = mpc_values

        # 更新初始状态
        for node in config["nodes"]:
            states = node.get("states", {})
            for state_name, state_spec in states.items():
                key = (node["id"], state_name)
                if key in current_states:
                    state_spec["initial"] = current_states[key]

        return config

    def _extract_solution(self, model, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从求解的模型中提取解

        Args:
            model: 求解后的Pyomo模型
            config: 模型配置

        Returns:
            包含状态和控制的字典
        """
        periods = list(model.T)
        first_period = periods[0]

        # 提取第一步的控制（流量）
        controls = {}
        for edge_id in model.E:
            controls[edge_id] = value(model.flow[edge_id, first_period])

        # 提取状态
        states = {}
        if hasattr(model, "state_index"):
            for node_id, state_name in model.state_index:
                key = (node_id, state_name)
                states[key] = value(model.state[key, first_period])

        # 提取目标函数值
        obj_value = value(model.obj)

        return {
            "controls": controls,
            "states": states,
            "objective": obj_value,
            "period": first_period,
        }

    def step(
        self, current_states: Dict[Tuple[str, str], float]
    ) -> Dict[str, Any]:
        """
        执行一步MPC优化

        Args:
            current_states: 当前状态 {(node_id, state_name): value}

        Returns:
            优化结果字典，包含controls和states
        """
        # 创建MPC配置
        mpc_config = self._create_mpc_config(current_states, self.current_step)

        # 构建并求解模型
        model = build_water_network_model(mpc_config, validate=False)

        try:
            results = self.solver.solve(model, tee=False, options=self.solver_options)

            from pyomo.opt import TerminationCondition

            if results.solver.termination_condition != TerminationCondition.optimal:
                raise SolverError(
                    f"MPC求解失败: {results.solver.termination_condition}"
                )

        except Exception as e:
            raise SolverError(f"MPC求解错误: {str(e)}") from e

        # 提取解
        solution = self._extract_solution(model, mpc_config)

        # 更新历史
        self.state_history.append(solution["states"])
        self.control_history.append(solution["controls"])
        self.current_step += 1

        return solution

    def run(
        self,
        initial_states: Dict[Tuple[str, str], float],
        num_steps: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> Dict[str, List]:
        """
        运行MPC仿真

        Args:
            initial_states: 初始状态
            num_steps: 运行步数（None则运行到时间序列结束）
            callback: 每步后调用的回调函数 callback(step, solution)

        Returns:
            历史记录字典 {'states': [...], 'controls': [...]}
        """
        # 重置历史
        self.state_history = []
        self.control_history = []
        self.current_step = 0

        # 确定运行步数
        total_periods = len(self.base_config["horizon"]["periods"])
        if num_steps is None:
            num_steps = total_periods - self.prediction_horizon + 1

        current_states = copy.copy(initial_states)

        for step in range(num_steps):
            # 执行MPC步骤
            solution = self.step(current_states)

            # 更新当前状态为下一时刻的状态
            current_states = solution["states"]

            # 调用回调
            if callback:
                callback(step, solution)

            # 检查是否到达终点
            if self.current_step >= total_periods:
                break

        return {
            "states": self.state_history,
            "controls": self.control_history,
        }

    def get_full_trajectory(self) -> Dict[str, Any]:
        """
        获取完整轨迹

        Returns:
            包含所有状态和控制历史的字典
        """
        return {
            "states": self.state_history,
            "controls": self.control_history,
            "num_steps": len(self.state_history),
        }


def create_mpc_controller(
    config: Dict[str, Any],
    prediction_horizon: int = 24,
    solver_name: str = "glpk",
) -> MPCController:
    """
    创建MPC控制器的便捷函数

    Args:
        config: 网络配置
        prediction_horizon: 预测窗口
        solver_name: 求解器名称

    Returns:
        MPCController实例
    """
    return MPCController(
        base_config=config,
        prediction_horizon=prediction_horizon,
        solver_name=solver_name,
    )


if __name__ == "__main__":
    # 示例：简单的水库-泵站系统MPC
    print("MPC滚动时域优化示例...")

    config = {
        "horizon": {"periods": [f"t{i}" for i in range(48)]},  # 48小时
        "nodes": [
            {
                "id": "reservoir",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 1000.0,
                        "bounds": (500.0, 2000.0),
                        "role": "storage",
                    }
                },
                "attributes": {"misc": {"inflow_series": "inflow"}},
            },
            {
                "id": "demand",
                "kind": "demand",
                "states": {},
                "attributes": {"demand_profile": "demand"},
            },
        ],
        "edges": [
            {
                "id": "flow",
                "kind": "pipeline",
                "from_node": "reservoir",
                "to_node": "demand",
                "attributes": {"capacity": 100.0, "energy_cost": 1.0},
            }
        ],
        "series": {
            "inflow": {
                "values": [60.0] * 48,
                "default": 60.0,
            },
            "demand": {
                "values": [50.0] * 48,
                "default": 50.0,
            },
        },
        "objective_weights": {
            "pumping_cost": 100.0,
            "shortage_penalty": 100000.0,
        },
    }

    # 创建MPC控制器
    mpc = create_mpc_controller(config, prediction_horizon=12)

    # 运行MPC
    initial_states = {("reservoir", "storage"): 1000.0}

    def print_callback(step, solution):
        storage = solution["states"][("reservoir", "storage")]
        flow = solution["controls"]["flow"]
        print(f"步骤 {step}: 库容={storage:.1f}, 流量={flow:.1f}")

    print("\n运行MPC优化...")
    results = mpc.run(initial_states, num_steps=10, callback=print_callback)

    print(f"\n完成! 共{len(results['states'])}步")

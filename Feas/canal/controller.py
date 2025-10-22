"""
渠道MPC控制器模块
"""

import numpy as np


class CanalMPCController:
    """渠道MPC控制器"""

    def __init__(
        self,
        canal_system,
        prediction_horizon=12,
        control_horizon=6,
        dt=5,
        depth_weight=100.0,
        control_weight=1.0,
        terminal_weight=200.0,
        first_gate_gain=1.5,
        middle_gate_gain=1.2,
        last_gate_gain=1.3,
        base_flow_ratio=0.6,
    ):
        """
        初始化MPC控制器

        Args:
            canal_system: 渠道系统对象
            prediction_horizon: 预测时域（步数），默认12（60分钟/5分钟）
            control_horizon: 控制时域（步数），默认6（30分钟/5分钟）
            dt: 采样周期（分钟），默认5
            depth_weight: 水深偏差权重，默认100.0
            control_weight: 控制变化权重，默认1.0
            terminal_weight: 终端状态权重，默认200.0
            first_gate_gain: 第一个闸门的控制增益，默认1.5
            middle_gate_gain: 中间闸门的控制增益，默认1.2
            last_gate_gain: 最后闸门的控制增益，默认1.3
            base_flow_ratio: 基础流量比例（相对于最大流量），默认0.6
        """
        self.system = canal_system
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.dt = dt

        # 权重参数
        self.depth_weight = depth_weight
        self.control_weight = control_weight
        self.terminal_weight = terminal_weight

        # 控制增益参数
        self.first_gate_gain = first_gate_gain
        self.middle_gate_gain = middle_gate_gain
        self.last_gate_gain = last_gate_gain
        self.base_flow_ratio = base_flow_ratio

    def optimize(self, current_depths, offtake_forecast):
        """
        MPC优化

        Args:
            current_depths: 当前水深
            offtake_forecast: 未来取水预测 [time][pool_id]

        Returns:
            optimal_gate_flows: 最优闸门流量序列
        """
        # 简化的MPC优化（基于启发式规则）
        # 在实际应用中应该使用优化求解器

        optimal_flows = []

        # 对每个控制步长进行优化
        for t in range(self.control_horizon):
            gate_flows = []

            for i, pool in enumerate(self.system.pools):
                # 目标：维持水深接近目标值
                depth_error = pool.current_depth - pool.target_depth

                # 计算期望的流量调整
                # 如果水深高于目标，增加出流
                # 如果水深低于目标，减少出流

                if i == 0:
                    # 第一个闸门：根据池段1的水深调整
                    base_flow = self.system.gates[0].max_flow * self.base_flow_ratio
                    adjustment = -depth_error * self.first_gate_gain
                    gate_flows.append(
                        np.clip(
                            base_flow + adjustment,
                            self.system.gates[0].min_flow,
                            self.system.gates[0].max_flow,
                        )
                    )
                else:
                    # 中间闸门：平衡上下游
                    base_flow = gate_flows[-1]  # 跟随上游
                    adjustment = -depth_error * self.middle_gate_gain
                    gate_flows.append(
                        np.clip(
                            base_flow + adjustment,
                            self.system.gates[i].min_flow,
                            self.system.gates[i].max_flow,
                        )
                    )

            # 最后一个闸门
            base_flow = gate_flows[-1]
            last_pool = self.system.pools[-1]
            depth_error = last_pool.current_depth - last_pool.target_depth
            adjustment = -depth_error * self.last_gate_gain
            gate_flows.append(
                np.clip(
                    base_flow + adjustment,
                    self.system.gates[-1].min_flow,
                    self.system.gates[-1].max_flow,
                )
            )

            optimal_flows.append(gate_flows)

        # 返回第一个控制动作
        if optimal_flows:
            return optimal_flows[0]
        else:
            # 返回默认流量（基于系统初始状态）
            num_gates = len(self.system.gates)
            default_flow = self.system.gates[0].max_flow * self.base_flow_ratio
            return [default_flow] * num_gates

    def set_weights(self, depth_weight=None, control_weight=None, terminal_weight=None):
        """
        更新权重参数

        Args:
            depth_weight: 水深偏差权重
            control_weight: 控制变化权重
            terminal_weight: 终端状态权重
        """
        if depth_weight is not None:
            self.depth_weight = depth_weight
        if control_weight is not None:
            self.control_weight = control_weight
        if terminal_weight is not None:
            self.terminal_weight = terminal_weight

    def set_gains(
        self, first_gate_gain=None, middle_gate_gain=None, last_gate_gain=None
    ):
        """
        更新控制增益

        Args:
            first_gate_gain: 第一个闸门的控制增益
            middle_gate_gain: 中间闸门的控制增益
            last_gate_gain: 最后闸门的控制增益
        """
        if first_gate_gain is not None:
            self.first_gate_gain = first_gate_gain
        if middle_gate_gain is not None:
            self.middle_gate_gain = middle_gate_gain
        if last_gate_gain is not None:
            self.last_gate_gain = last_gate_gain

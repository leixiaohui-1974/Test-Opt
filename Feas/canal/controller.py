"""
渠道MPC控制器模块
"""

import numpy as np
import cvxpy as cp


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

    def optimize_with_mass_balance(self, current_depths, offtake_forecast, current_gate_flows=None):
        """
        基于优化求解的MPC，带质量平衡约束

        质量平衡约束: Gate[i+1] + Offtake[i] = Gate[i]
        这将产生级联流量递减效果

        Args:
            current_depths: 当前水深 [pool1, pool2, pool3, pool4]
            offtake_forecast: 未来取水预测 [time][pool_id]
            current_gate_flows: 当前闸门流量 [gate0, gate1, gate2, gate3, gate4] (可选)

        Returns:
            optimal_gate_flows: 最优闸门流量 [gate0, gate1, gate2, gate3, gate4]
        """
        n_pools = len(self.system.pools)
        n_gates = len(self.system.gates)
        N = self.prediction_horizon

        # 如果没有提供当前闸门流量，使用默认值
        if current_gate_flows is None:
            current_gate_flows = [self.system.gates[i].max_flow * 0.6 for i in range(n_gates)]

        # 定义优化变量
        gate_flows = cp.Variable((N, n_gates))  # 闸门流量决策变量
        depths = cp.Variable((N+1, n_pools))     # 水深状态变量

        # 目标函数
        cost = 0

        # 深度跟踪误差成本
        for t in range(N):
            for i in range(n_pools):
                target = self.system.pools[i].target_depth
                cost += self.depth_weight * cp.square(depths[t+1, i] - target)

        # 终端成本
        for i in range(n_pools):
            target = self.system.pools[i].target_depth
            cost += self.terminal_weight * cp.square(depths[N, i] - target)

        # 控制平滑成本
        for t in range(N-1):
            for i in range(n_gates):
                cost += self.control_weight * cp.square(gate_flows[t+1, i] - gate_flows[t, i])

        # 约束条件
        constraints = []

        # 初始状态约束
        for i in range(n_pools):
            constraints.append(depths[0, i] == current_depths[i])

        # 动态约束 (简化的IDZ模型)
        for t in range(N):
            # 获取offtake预测
            if t < len(offtake_forecast):
                offtakes = offtake_forecast[t]
            else:
                offtakes = offtake_forecast[-1] if offtake_forecast else [0] * (n_pools - 1)

            # 为每个池段建立动态方程
            for i in range(n_pools):
                # 流入 = gate[i]
                # 流出 = gate[i+1] + offtake[i] (如果有offtake的话)
                inflow = gate_flows[t, i]

                if i < n_pools - 1:
                    outflow = gate_flows[t, i+1]
                    if i < len(offtakes):
                        outflow += offtakes[i]
                else:
                    # 最后一个池段
                    outflow = gate_flows[t, i+1]

                # 简化的水深变化模型: dh/dt ≈ (inflow - outflow) / area
                area = self.system.pools[i].length * self.system.pools[i].width
                depth_change = (inflow - outflow) * self.dt / area

                constraints.append(depths[t+1, i] == depths[t, i] + depth_change)

        # 质量平衡约束 (核心约束!)
        # Gate[i+1] + Offtake[i] = Gate[i] (or Gate[i+1] <= Gate[i] - Offtake[i])
        # 这将确保流量从上游到下游递减
        # 使用软约束（放宽容差）以提高可行性
        mass_balance_penalty = 0
        for t in range(N):
            if t < len(offtake_forecast):
                offtakes = offtake_forecast[t]
            else:
                offtakes = offtake_forecast[-1] if offtake_forecast else [0] * (n_pools - 1)

            # 对每个池段应用质量平衡
            for i in range(n_gates - 1):
                if i < len(offtakes):
                    # 有offtake的池段: Gate[i+1] = Gate[i] - Offtake[i]
                    expected_downstream = gate_flows[t, i] - offtakes[i]
                else:
                    # 没有offtake的池段（如Pool 4）: Gate[i+1] ≈ Gate[i]
                    expected_downstream = gate_flows[t, i]

                # 松弛变量方法：允许偏差，但在目标函数中惩罚
                mass_balance_penalty += 10.0 * cp.square(gate_flows[t, i+1] - expected_downstream)

                # 硬约束：确保物理可行性（下游流量不能凭空增加）
                if i < len(offtakes):
                    # 有offtake: 下游必须小于等于上游（因为有分水）
                    constraints.append(gate_flows[t, i+1] <= gate_flows[t, i])
                else:
                    # 没有offtake: 下游应该约等于上游（允许小幅调节）
                    constraints.append(gate_flows[t, i+1] <= gate_flows[t, i] + 1.0)
                    constraints.append(gate_flows[t, i+1] >= gate_flows[t, i] - 1.0)

        # 将质量平衡惩罚加入目标函数
        cost += mass_balance_penalty

        # 闸门流量上下限约束
        for t in range(N):
            for i in range(n_gates):
                constraints.append(gate_flows[t, i] >= self.system.gates[i].min_flow)
                constraints.append(gate_flows[t, i] <= self.system.gates[i].max_flow)

        # 求解优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                # 返回第一个时间步的最优控制动作
                return gate_flows[0, :].value.tolist()
            else:
                print(f"Warning: MPC optimization failed with status: {problem.status}")
                # 返回启发式解作为后备
                return self.optimize(current_depths, offtake_forecast)

        except Exception as e:
            print(f"Warning: MPC optimization error: {e}")
            # 返回启发式解作为后备
            return self.optimize(current_depths, offtake_forecast)

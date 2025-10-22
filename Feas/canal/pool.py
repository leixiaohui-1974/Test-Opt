"""
渠道池段模型

IDZ (Integrator Delay Zero) 模型：
- 延迟效应：水流传播需要时间
- 顶托效应：下游水位影响上游流量
- 回水区：非均匀流条件下的水位分布
- 动态响应：考虑渠道惯性和扰动
"""

import numpy as np
from collections import deque


class IDZCanalPool:
    """IDZ模型的渠道池段（改进版：Muskingum蓄量关系）"""

    def __init__(
        self,
        pool_id,
        length,
        width,
        bottom_slope,
        roughness,
        target_depth,
        delay_time,
        backwater_coeff=0.1,
        side_slope=1.5,
        initial_inflow=None,
        initial_outflow=None,
        muskingum_K_factor=2.5,
        muskingum_X=0.20,
    ):
        """
        初始化渠道池段

        Args:
            pool_id: 池段编号
            length: 池段长度 (m)
            width: 渠底宽度 (m)
            bottom_slope: 底坡
            roughness: 糙率 (Manning n)
            target_depth: 目标水深 (m)
            delay_time: 延迟时间 (分钟)
            backwater_coeff: 顶托系数，默认0.1
            side_slope: 边坡系数 (m=1.5表示1:1.5的边坡)，默认1.5
            initial_inflow: 初始入流 (m³/min)，默认为None（自动计算）
            initial_outflow: 初始出流 (m³/min)，默认为None（自动计算）
            muskingum_K_factor: Muskingum K参数的放大系数，默认2.5（增加系统惯性）
            muskingum_X: Muskingum X参数（权重），默认0.20
        """
        self.pool_id = pool_id
        self.length = length
        self.width = width  # 底宽
        self.bottom_slope = bottom_slope
        self.roughness = roughness
        self.target_depth = target_depth
        self.delay_time = delay_time
        self.backwater_coeff = backwater_coeff
        self.side_slope = side_slope  # 梯形断面边坡

        # Muskingum参数（蓄量=f(流量,水位)）
        # S = K[X*I + (1-X)*O]，其中K是蓄量常数，X是权重
        wave_celerity = self._calculate_wave_celerity()
        self.muskingum_K = (length / (wave_celerity * 60)) * muskingum_K_factor
        self.muskingum_X = muskingum_X

        # 延迟队列（存储历史入流）
        self.inflow_history = deque(maxlen=int(delay_time) + 1)

        # 计算初始流量（如果未指定）
        if initial_inflow is None:
            initial_inflow = self._depth_to_flow(target_depth)
        if initial_outflow is None:
            initial_outflow = initial_inflow

        # 当前状态
        self.current_depth = target_depth
        self.current_inflow = initial_inflow
        self.current_outflow = initial_outflow
        # 初始化时，流量平衡，使用目标水深
        self.current_storage = self._calculate_muskingum_storage(
            target_depth, initial_inflow, initial_outflow
        )

    def _calculate_wave_celerity(self):
        """计算波速 (m/s) - 用于Muskingum K参数"""
        # 简化：c = dQ/dA ≈ 5/3 * v
        # 估算平均流速
        normal_depth = self.target_depth
        area = self._get_wetted_area(normal_depth)
        perimeter = self._get_wetted_perimeter(normal_depth)
        hydraulic_radius = area / perimeter

        # Manning公式估算流速
        velocity = (
            (1 / self.roughness)
            * (hydraulic_radius ** (2 / 3))
            * (self.bottom_slope**0.5)
        )
        celerity = (5 / 3) * velocity
        return max(celerity, 0.5)  # 最小0.5 m/s

    def _get_wetted_area(self, depth):
        """计算过流断面面积（梯形断面）"""
        # A = (b + m*h)*h
        return (self.width + self.side_slope * depth) * depth

    def _get_wetted_perimeter(self, depth):
        """计算湿周（梯形断面）"""
        # P = b + 2*h*sqrt(1+m^2)
        return self.width + 2 * depth * np.sqrt(1 + self.side_slope**2)

    def _flow_to_depth(self, flow):
        """根据流量计算水深（Manning公式反算）"""
        # 迭代求解：Q = (1/n) * A * R^(2/3) * S^(1/2)
        # 简化：使用目标水深附近的线性近似
        target_flow = self._depth_to_flow(self.target_depth)
        if abs(target_flow) < 0.1:
            return self.target_depth

        depth_ratio = (flow / target_flow) ** 0.6  # 简化关系
        estimated_depth = self.target_depth * depth_ratio

        # 限制在合理范围
        return np.clip(
            estimated_depth, self.target_depth * 0.2, self.target_depth * 2.5
        )

    def _depth_to_flow(self, depth):
        """根据水深计算流量（Manning公式）"""
        area = self._get_wetted_area(depth)
        perimeter = self._get_wetted_perimeter(depth)
        hydraulic_radius = area / perimeter if perimeter > 0 else 0

        # Manning公式: Q = (1/n) * A * R^(2/3) * S^(1/2)
        flow = (
            (1 / self.roughness)
            * area
            * (hydraulic_radius ** (2 / 3))
            * (self.bottom_slope**0.5)
        )
        return flow * 60  # 转换为 m³/min

    def _calculate_muskingum_storage(self, depth, inflow, outflow):
        """
        计算Muskingum蓄量
        S = K[X*I + (1-X)*O] + 棱柱蓄量

        棱柱蓄量基于水深和断面
        楔形蓄量基于入流出流差异
        """
        # 棱柱蓄量（基于平均水深）
        prism_storage = self._get_wetted_area(depth) * self.length

        # Muskingum楔形蓄量修正
        wedge_storage = self.muskingum_K * (
            self.muskingum_X * inflow + (1 - self.muskingum_X) * outflow
        )

        total_storage = prism_storage + wedge_storage
        return max(total_storage, 0.0)

    def depth_to_storage(self, depth, inflow, outflow):
        """水深+流量转换为蓄水量（考虑Muskingum关系）"""
        return self._calculate_muskingum_storage(depth, inflow, outflow)

    def storage_to_depth(self, storage, inflow, outflow, max_iterations=5, damping=0.5):
        """
        蓄水量转换为水深（考虑Muskingum关系，迭代求解）

        Args:
            storage: 蓄水量
            inflow: 入流
            outflow: 出流
            max_iterations: 最大迭代次数，默认5
            damping: 阻尼系数，默认0.5

        Returns:
            depth: 水深
        """
        # 迭代求解：给定蓄量和流量，反算水深
        depth_guess = self.current_depth

        for _ in range(max_iterations):
            calc_storage = self._calculate_muskingum_storage(
                depth_guess, inflow, outflow
            )
            error = storage - calc_storage

            # 简单调整
            area = self._get_wetted_area(depth_guess)
            if area > 0:
                depth_correction = error / (
                    self.length * (self.width + 2 * self.side_slope * depth_guess)
                )
                depth_guess += depth_correction * damping

            depth_guess = np.clip(
                depth_guess, self.target_depth * 0.3, self.target_depth * 2.0
            )

        return depth_guess

    def update_state(self, inflow, outflow, downstream_depth, dt):
        """
        更新池段状态（使用Muskingum方法）

        Args:
            inflow: 上游入流 (m³/min)
            outflow: 下游出流 (m³/min)
            downstream_depth: 下游水深 (m)
            dt: 时间步长 (分钟)

        Returns:
            new_depth: 更新后的水深 (m)
        """
        # 顶托效应：下游水深影响出流
        backwater_effect = self.backwater_coeff * (
            downstream_depth - self.target_depth
        )
        adjusted_outflow = outflow * (1 - backwater_effect)

        # 保存当前入流出流用于Muskingum计算
        self.current_inflow = inflow
        self.current_outflow = adjusted_outflow

        # 水量平衡：dS/dt = I - O
        dStorage = (inflow - adjusted_outflow) * dt
        new_storage = self.current_storage + dStorage

        # 使用新的蓄量和流量反算水深（考虑Muskingum关系）
        new_depth = self.storage_to_depth(new_storage, inflow, adjusted_outflow)

        # 限制在合理范围内
        new_depth = np.clip(
            new_depth, self.target_depth * 0.3, self.target_depth * 2.0
        )

        # 重新计算蓄量（保证一致性）
        new_storage = self._calculate_muskingum_storage(
            new_depth, inflow, adjusted_outflow
        )

        # 更新状态
        self.current_storage = new_storage
        self.current_depth = new_depth

        # 更新延迟队列
        self.inflow_history.append(inflow)

        return self.current_depth

    def get_delayed_inflow(self):
        """获取延迟后的入流（用于下游池段）"""
        if len(self.inflow_history) > 0:
            return self.inflow_history[0]
        return 0.0

"""
闸门控制器模块
"""

import numpy as np


class Gate:
    """闸门控制器"""

    def __init__(self, gate_id, max_flow, min_flow=0, initial_opening=0.5):
        """
        初始化闸门

        Args:
            gate_id: 闸门编号
            max_flow: 最大流量 (m³/min)
            min_flow: 最小流量 (m³/min)，默认0
            initial_opening: 初始开度（0-1之间），默认0.5
        """
        self.gate_id = gate_id
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.current_opening = np.clip(initial_opening, 0.0, 1.0)

    def set_flow(self, target_flow):
        """
        设置目标流量

        Args:
            target_flow: 目标流量 (m³/min)

        Returns:
            flow: 实际设置的流量（限制在min_flow和max_flow之间）
        """
        flow = np.clip(target_flow, self.min_flow, self.max_flow)
        self.current_opening = (flow - self.min_flow) / (self.max_flow - self.min_flow) if self.max_flow > self.min_flow else 0.0
        return flow

    def get_flow(self):
        """
        获取当前流量

        Returns:
            flow: 当前流量 (m³/min)
        """
        return self.min_flow + self.current_opening * (self.max_flow - self.min_flow)

    def set_opening(self, opening):
        """
        直接设置闸门开度

        Args:
            opening: 开度（0-1之间）
        """
        self.current_opening = np.clip(opening, 0.0, 1.0)

    def get_opening(self):
        """
        获取当前开度

        Returns:
            opening: 当前开度（0-1之间）
        """
        return self.current_opening

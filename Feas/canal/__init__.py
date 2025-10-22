"""
渠道控制基础类库

提供渠道系统建模和MPC控制的通用组件：
- IDZCanalPool: 基于IDZ模型的渠道池段
- Gate: 闸门控制器
- CanalMPCController: MPC控制器

示例:
    from Feas.canal import IDZCanalPool, Gate, CanalMPCController

    # 创建池段
    pool = IDZCanalPool(
        pool_id=1,
        length=2000,
        width=10,
        bottom_slope=0.0002,
        roughness=0.025,
        target_depth=2.0,
        delay_time=5
    )

    # 创建闸门
    gate = Gate(gate_id=0, max_flow=40, min_flow=10)

    # 创建控制器
    controller = CanalMPCController(
        canal_system,
        prediction_horizon=12,
        control_horizon=6
    )
"""

from .pool import IDZCanalPool
from .gate import Gate
from .controller import CanalMPCController

__all__ = ["IDZCanalPool", "Gate", "CanalMPCController"]

__version__ = "1.0.0"

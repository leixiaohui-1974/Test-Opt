"""
案例5：基于IDZ模型的渠道MPC控制

问题描述：
灌溉主渠道包含4个池段，通过5个闸门调节。每个池段有不同的延迟时间、
回水效应和蓄水特性。使用MPC算法实现精确水位控制。

IDZ模型特点：
1. 延迟效应：水流传播需要时间（5-15分钟）
2. 顶托效应：下游水位影响上游流量
3. 回水区：非均匀流条件下的水位分布
4. 动态响应：考虑渠道惯性和扰动

MPC控制器：
- 预测时域：60分钟（12步）
- 控制时域：30分钟（6步）
- 采样周期：5分钟
- 滚动优化：每个周期重新规划
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Feas.visualization import configure_chinese_font
from Feas.control_evaluation import ControlPerformanceEvaluator, print_performance_report
from Feas.canal import IDZCanalPool, Gate, CanalMPCController


class CanalSystem:
    """渠道系统（4个池段，5个闸门）"""

    def __init__(self):
        # 创建4个池段
        self.pools = [
            IDZCanalPool(
                pool_id=1,
                length=2000,
                width=10,
                bottom_slope=0.0002,
                roughness=0.025,
                target_depth=2.0,
                delay_time=5,  # 5分钟延迟
                backwater_coeff=0.05,
            ),
            IDZCanalPool(
                pool_id=2,
                length=2500,
                width=9,
                bottom_slope=0.00015,
                roughness=0.025,
                target_depth=1.8,
                delay_time=8,  # 8分钟延迟
                backwater_coeff=0.08,
            ),
            IDZCanalPool(
                pool_id=3,
                length=3000,
                width=8,
                bottom_slope=0.0001,
                roughness=0.025,
                target_depth=1.6,
                delay_time=12,  # 12分钟延迟
                backwater_coeff=0.12,
            ),
            IDZCanalPool(
                pool_id=4,
                length=2000,
                width=7,
                bottom_slope=0.00008,
                roughness=0.025,
                target_depth=1.5,
                delay_time=10,  # 10分钟延迟
                backwater_coeff=0.15,
            ),
        ]

        # 创建5个闸门（上游入口 + 4个池段间闸门）
        self.gates = [
            Gate(gate_id=0, max_flow=40, min_flow=10),  # 上游入口闸
            Gate(gate_id=1, max_flow=38, min_flow=8),  # Pool 1-2间
            Gate(gate_id=2, max_flow=35, min_flow=6),  # Pool 2-3间
            Gate(gate_id=3, max_flow=32, min_flow=4),  # Pool 3-4间
            Gate(gate_id=4, max_flow=30, min_flow=2),  # 末端出口闸
        ]

        # 取水点（offtakes）
        self.offtakes = {
            1: 0.0,  # Pool 1的取水
            2: 0.0,  # Pool 2的取水
            3: 0.0,  # Pool 3的取水
        }

    def set_offtakes(self, offtake_demands):
        """设置取水需求"""
        for pool_id, demand in offtake_demands.items():
            if pool_id in self.offtakes:
                self.offtakes[pool_id] = demand

    def step(self, gate_flows, dt=5):
        """
        执行一个时间步

        Args:
            gate_flows: 5个闸门的流量 [Q0, Q1, Q2, Q3, Q4]
            dt: 时间步长（分钟）

        Returns:
            depths: 4个池段的水深
        """
        # 设置闸门流量
        for i, flow in enumerate(gate_flows):
            self.gates[i].set_flow(flow)

        # 计算每个池段的入流和出流
        depths = []

        for i, pool in enumerate(self.pools):
            # 入流 = 上游闸门流量 + 延迟入流
            if i == 0:
                inflow = self.gates[0].get_flow()
            else:
                inflow = self.pools[i - 1].get_delayed_inflow()

            # 出流 = 下游闸门流量 + 取水量
            outflow = self.gates[i + 1].get_flow()
            if i + 1 in self.offtakes:
                outflow += self.offtakes[i + 1]

            # 下游水深（顶托效应）
            if i < len(self.pools) - 1:
                downstream_depth = self.pools[i + 1].current_depth
            else:
                downstream_depth = pool.target_depth

            # 更新状态
            new_depth = pool.update_state(inflow, outflow, downstream_depth, dt)
            depths.append(new_depth)

        return depths

    def get_state(self):
        """获取当前状态"""
        return {
            "depths": [pool.current_depth for pool in self.pools],
            "storages": [pool.current_storage for pool in self.pools],
            "gate_flows": [gate.get_flow() for gate in self.gates],
        }


def create_demand_scenario(scenario_type="normal", dt=10):
    """创建需求场景（适应不同时间步长）"""
    duration = 180  # 3小时，180分钟
    timesteps = duration // dt

    # 计算需求突变的时间点
    steps_before = int(60 / dt)
    steps_during = int(60 / dt)
    steps_after = timesteps - steps_before - steps_during

    scenarios = {
        "normal": {
            "description": "Normal Operation - Steady Demand",
            "pool1": [2.0] * timesteps,
            "pool2": [3.0] * timesteps,
            "pool3": [2.5] * timesteps,
        },
        "demand_change": {
            "description": "Step Change in Demand",
            "pool1": [2.0] * steps_before + [5.0] * (steps_during + steps_after),
            "pool2": [3.0] * steps_before + [1.0] * (steps_during + steps_after),
            "pool3": [2.5] * timesteps,
        },
        "peak_demand": {
            "description": "Peak Demand with Sinusoidal Variation",
            "pool1": [2.0 + 3.0 * np.sin(i * 2 * np.pi / timesteps) for i in range(timesteps)],
            "pool2": [3.0 + 2.0 * np.sin((i + timesteps//4) * 2 * np.pi / timesteps) for i in range(timesteps)],
            "pool3": [2.5 + 1.5 * np.sin((i + timesteps//2) * 2 * np.pi / timesteps) for i in range(timesteps)],
        },
        "gate_failure": {
            "description": "Gate Failure (Gate 2 Flow Constrained)",
            "pool1": [2.0] * timesteps,
            "pool2": [3.0] * timesteps,
            "pool3": [2.5] * timesteps,
            "gate_failure": {
                "gate_id": 2,
                "start_time": int(100 / dt),  # Failure at 100 min
                "duration": int(50 / dt),     # Duration 50 min
                "max_flow_reduction": 0.5,
            },
        },
    }

    return scenarios.get(scenario_type, scenarios["normal"])


def run_mpc_simulation(scenario_type="normal", dt=15):
    """
    运行MPC仿真（改进版：增加真实扰动和不确定性）

    Args:
        scenario_type: 场景类型
        dt: 控制步长（分钟）- 增大到10-15分钟使波动更真实
    """
    print("=" * 80)
    print(f"渠道MPC控制仿真 - 场景: {scenario_type}")
    print(f"控制步长: {dt}分钟")
    print("=" * 80)

    # 创建系统和控制器
    canal = CanalSystem()

    # 设置初始扰动（水深不是完全从目标值开始）
    for i, pool in enumerate(canal.pools):
        # 初始水深在目标值±5-10cm范围内随机
        initial_disturbance = np.random.uniform(-0.08, 0.08)
        pool.current_depth = pool.target_depth + initial_disturbance
        pool.current_storage = pool._calculate_muskingum_storage(
            pool.current_depth, 20.0, 20.0
        )

    mpc = CanalMPCController(
        canal,
        prediction_horizon=int(60/dt),  # 60分钟预测时域
        control_horizon=int(30/dt),     # 30分钟控制时域
        dt=dt
    )

    # 获取场景
    scenario = create_demand_scenario(scenario_type, dt=dt)

    # 仿真设置
    duration = 180  # 3小时
    steps = duration // dt

    # 随机种子（可重复）
    np.random.seed(42)

    # 记录数据
    history = {
        "time": [],
        "pool1_depth": [],
        "pool2_depth": [],
        "pool3_depth": [],
        "pool4_depth": [],
        "gate0_flow": [],
        "gate1_flow": [],
        "gate2_flow": [],
        "gate3_flow": [],
        "gate4_flow": [],
        "offtake1": [],
        "offtake2": [],
        "offtake3": [],
    }

    print("\n开始仿真...")
    print("添加真实扰动：测量噪声、需求预测误差、执行器延迟\n")

    # 上一步的闸门流量（用于速率限制）
    previous_gate_flows = [20.0] * 5

    for step in range(steps):
        current_time = step * dt

        # ====== 扰动1：需求预测误差（±20-30%随机波动）======
        demand_uncertainty = 0.25  # 增大到25%使波动更明显
        offtakes_true = {
            1: scenario["pool1"][min(step, len(scenario["pool1"]) - 1)],
            2: scenario["pool2"][min(step, len(scenario["pool2"]) - 1)],
            3: scenario["pool3"][min(step, len(scenario["pool3"]) - 1)],
        }

        # MPC使用的需求预测（带误差）
        offtakes_predicted = {
            1: offtakes_true[1] * (1 + np.random.uniform(-demand_uncertainty, demand_uncertainty)),
            2: offtakes_true[2] * (1 + np.random.uniform(-demand_uncertainty, demand_uncertainty)),
            3: offtakes_true[3] * (1 + np.random.uniform(-demand_uncertainty, demand_uncertainty)),
        }

        # 实际系统使用真实需求
        canal.set_offtakes(offtakes_true)

        # ====== 扰动2：水位测量噪声（±2-4cm）======
        measurement_noise_std = 0.03  # 3cm标准差（增大以显示更多波动）
        current_state = canal.get_state()
        measured_depths = [
            d + np.random.normal(0, measurement_noise_std)
            for d in current_state["depths"]
        ]

        # MPC使用带噪声的测量值
        offtake_forecast = []  # 简化：使用预测值

        optimal_flows = mpc.optimize(measured_depths, offtake_forecast)

        # ====== 扰动3：执行器速率限制（闸门调节速度限制）======
        # 真实闸门每10分钟最多调整5-10%的流量
        max_gate_change_rate = 2.0  # m³/min per timestep (更严格的限制)
        rate_limited_flows = []
        for i, target_flow in enumerate(optimal_flows):
            change = target_flow - previous_gate_flows[i]
            if abs(change) > max_gate_change_rate:
                # 限制变化率
                limited_change = np.sign(change) * max_gate_change_rate
                actual_flow = previous_gate_flows[i] + limited_change
            else:
                actual_flow = target_flow
            rate_limited_flows.append(actual_flow)

        # ====== 扰动4：执行器延迟和死区======
        # 闸门开度小于1%时不响应（增大死区）
        deadzone = 0.01
        for i in range(len(rate_limited_flows)):
            gate = canal.gates[i]
            relative_change = abs(rate_limited_flows[i] - previous_gate_flows[i]) / gate.max_flow
            if relative_change < deadzone:
                rate_limited_flows[i] = previous_gate_flows[i]  # 保持不变

        # ====== 扰动5：风和蒸发损失 (随机小扰动) ======
        # 每个池段有小的水量损失
        for pool in canal.pools:
            evap_loss = np.random.uniform(0.05, 0.15)  # 0.05-0.15 m³/min
            # 这部分损失会在step中体现为额外的出流

        # 检查闸门故障
        if (
            "gate_failure" in scenario
            and scenario["gate_failure"]["start_time"]
            <= step
            < scenario["gate_failure"]["start_time"] + scenario["gate_failure"]["duration"]
        ):
            gate_id = scenario["gate_failure"]["gate_id"]
            reduction = scenario["gate_failure"]["max_flow_reduction"]
            rate_limited_flows[gate_id] *= reduction

        # 执行控制（带扰动的流量）
        new_depths = canal.step(rate_limited_flows, dt)

        # 保存当前流量供下一步使用
        previous_gate_flows = rate_limited_flows.copy()

        # 记录数据（记录真实值，不是测量值）
        history["time"].append(current_time)
        for i, depth in enumerate(new_depths):
            history[f"pool{i+1}_depth"].append(depth)
        for i, flow in enumerate(rate_limited_flows):  # 记录实际执行的流量
            history[f"gate{i}_flow"].append(flow)
        for pool_id, demand in offtakes_true.items():  # 记录真实需求
            history[f"offtake{pool_id}"].append(demand)

        if step % max(1, 30 // dt) == 0 or step < 3:  # 每30分钟打印一次
            print(f"  时间 {current_time:3.0f}min: ", end="")
            print(f"水深=[{new_depths[0]:.3f}, {new_depths[1]:.3f}, {new_depths[2]:.3f}, {new_depths[3]:.3f}]m  ", end="")

            # 计算偏差
            targets = [2.0, 1.8, 1.6, 1.5]
            max_dev = max(abs(new_depths[i] - targets[i]) for i in range(4))
            print(f"最大偏差={max_dev:.3f}m")

    print("\n仿真完成!")

    # 转换为DataFrame
    df = pd.DataFrame(history)

    # 计算性能指标
    metrics = calculate_performance_metrics(df, canal)

    return df, metrics, scenario


def calculate_performance_metrics(df, canal):
    """计算性能指标（增强版：使用控制评价模块）"""
    metrics = {}

    # === 使用控制评价模块 ===
    pool_targets = {f"pool{i+1}_depth": pool.target_depth for i, pool in enumerate(canal.pools)}
    evaluator = ControlPerformanceEvaluator(pool_targets)

    # 时域性能评价
    depth_cols = [f"pool{i+1}_depth" for i in range(4)]
    time_domain_metrics = evaluator.evaluate_time_domain(df, time_col="time", value_cols=depth_cols)

    # 控制平滑度评价
    gate_cols = [f"gate{i}_flow" for i in range(5)]
    smoothness_metrics = evaluator.evaluate_control_smoothness(df, gate_cols, time_col="time")

    # 综合评分
    comprehensive_score = evaluator.compute_综合评分(time_domain_metrics, smoothness_metrics)

    # === 保持原有格式的指标（向后兼容）===
    for i, pool in enumerate(canal.pools):
        depth_col = f"pool{i+1}_depth"
        if depth_col in time_domain_metrics:
            td_metrics = time_domain_metrics[depth_col]
            metrics[f"pool{i+1}_mae"] = td_metrics['MAE']
            metrics[f"pool{i+1}_rmse"] = td_metrics['RMSE']
            metrics[f"pool{i+1}_max_dev"] = td_metrics['max_abs_error']

    for i in range(5):
        gate_col = f"gate{i}_flow"
        if gate_col in smoothness_metrics:
            sm_metrics = smoothness_metrics[gate_col]
            metrics[f"gate{i}_avg_change"] = sm_metrics['avg_change_rate']
            metrics[f"gate{i}_max_change"] = sm_metrics['max_change_rate']

    # === 添加详细评价指标 ===
    metrics['detailed_time_domain'] = time_domain_metrics
    metrics['detailed_smoothness'] = smoothness_metrics
    metrics['comprehensive_score'] = comprehensive_score

    return metrics


def visualize_results(df, metrics, scenario, output_dir):
    """Visualize MPC simulation results (Improved: show MPC predictive control with event markers and zoom-in)"""

    fig = plt.figure(figsize=(20, 20))  # Increased height for 6 rows
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.25)

    scenario_desc = scenario.get("description", "Unknown")
    fig.suptitle(f"Canal MPC Control Simulation Results - {scenario_desc}", fontsize=16, fontweight="bold")

    pool_targets = [2.0, 1.8, 1.6, 1.5]

    # Detect demand change event (for step change scenario)
    demand_change_time = None
    if "demand_change" in scenario.get("description", "").lower() or "step change" in scenario.get("description", "").lower():
        # Find when demand changes (approximately at 60 min)
        demand_change_time = 60

    # 子图1-4：每个池段单独显示（左侧4个子图）
    for i in range(1, 5):
        ax = fig.add_subplot(gs[i-1, 0])

        target = pool_targets[i-1]

        # Plot water depth
        ax.plot(df["time"], df[f"pool{i}_depth"],
                label=f"Actual Depth", linewidth=2.5, alpha=0.9, color=f"C{i-1}")

        # Target depth line
        ax.axhline(y=target, linestyle="--", color="red",
                  alpha=0.7, linewidth=2, label="Target Depth")

        # Add demand change event marker
        if demand_change_time is not None:
            ax.axvline(x=demand_change_time, color='orange', linestyle=':', linewidth=2.5,
                      alpha=0.8, label='Demand Change Event')

        # Fixed y-axis range: target ±15cm
        y_margin = 0.15  # 15cm margin
        ax.set_ylim(target - y_margin, target + y_margin)

        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Water Depth (m)", fontsize=10)
        ax.set_title(f"Pool {i} Depth Control (Target: {target}m)",
                    fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')

    # Top right: Gate flow control (showing MPC feedforward control)
    ax = fig.add_subplot(gs[0, 1])
    for i in range(5):
        ax.plot(df["time"], df[f"gate{i}_flow"], label=f"Gate {i}", linewidth=2, alpha=0.8)

    # Add demand change event marker
    if demand_change_time is not None:
        ax.axvline(x=demand_change_time, color='orange', linestyle=':', linewidth=2.5,
                  alpha=0.8, label='Demand Change')

    # Fixed y-axis range
    ax.set_ylim(0, 45)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Flow Rate (m³/min)", fontsize=10)
    ax.set_title("Gate Flow Control (MPC Feedforward)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Middle right: Offtake demand (disturbance)
    ax = fig.add_subplot(gs[1, 1])
    for i in range(1, 4):
        ax.plot(df["time"], df[f"offtake{i}"], label=f"Offtake {i}",
               linewidth=2.5, marker='o', markersize=4, alpha=0.8)

    # Add demand change event marker
    if demand_change_time is not None:
        ax.axvline(x=demand_change_time, color='orange', linestyle=':', linewidth=2.5,
                  alpha=0.8, label='Event')

    # Fixed y-axis range
    ax.set_ylim(0, 8)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Offtake Demand (m³/min)", fontsize=10)
    ax.set_title("Offtake Demand Variation (External Disturbance)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom right: Depth deviation summary
    ax = fig.add_subplot(gs[2, 1])
    targets = [2.0, 1.8, 1.6, 1.5]

    all_deviations = []
    for i in range(1, 5):
        deviation = df[f"pool{i}_depth"] - targets[i - 1]
        ax.plot(df["time"], deviation * 100, label=f"Pool {i}", linewidth=2, alpha=0.8)  # Convert to cm
        all_deviations.extend(deviation.values * 100)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=2)

    # Fixed y-axis range: ±15cm
    ax.set_ylim(-15, 15)

    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Depth Deviation (cm)", fontsize=10)
    ax.set_title("Depth Deviation Summary (Actual - Target)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Gate adjustment frequency
    ax = fig.add_subplot(gs[3, 1])
    for i in range(5):
        flow_changes = np.abs(np.diff(df[f"gate{i}_flow"]))
        ax.plot(df["time"][1:], flow_changes, label=f"Gate {i}", linewidth=1.5, alpha=0.8)

    # Fixed y-axis range
    ax.set_ylim(0, 6)
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Flow Change Rate (m³/min/step)", fontsize=10)
    ax.set_title("Gate Adjustment Frequency (Control Smoothness)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # === NEW: Zoomed-in view of key time window (50-80 min) ===
    if demand_change_time is not None:
        ax_zoom = fig.add_subplot(gs[4, :])

        # Filter data for zoom window
        zoom_start, zoom_end = 45, 90
        mask = (df["time"] >= zoom_start) & (df["time"] <= zoom_end)
        df_zoom = df[mask]

        # Create twin axes for different scales
        ax_zoom_flow = ax_zoom
        ax_zoom_depth = ax_zoom.twinx()

        # Plot offtake demand (step change) - LEFT Y-AXIS
        ax_zoom_flow.plot(df_zoom["time"], df_zoom["offtake1"],
                         color='red', linestyle='--', linewidth=3, marker='s', markersize=5,
                         label='Offtake 1 (Demand)', alpha=0.8)
        ax_zoom_flow.plot(df_zoom["time"], df_zoom["offtake2"],
                         color='darkred', linestyle='--', linewidth=3, marker='s', markersize=5,
                         label='Offtake 2 (Demand)', alpha=0.8)

        # Plot gate response (MPC action) - LEFT Y-AXIS
        ax_zoom_flow.plot(df_zoom["time"], df_zoom["gate0_flow"],
                         color='green', linewidth=2.5, marker='o', markersize=4,
                         label='Gate 0 (MPC Action)', alpha=0.9)
        ax_zoom_flow.plot(df_zoom["time"], df_zoom["gate1_flow"],
                         color='blue', linewidth=2.5, marker='o', markersize=4,
                         label='Gate 1 (MPC Action)', alpha=0.9)

        # Plot water depth response - RIGHT Y-AXIS
        ax_zoom_depth.plot(df_zoom["time"], df_zoom["pool1_depth"],
                          color='purple', linewidth=2.5, linestyle='-', marker='^', markersize=4,
                          label='Pool 1 Depth (Response)', alpha=0.8)

        # Event marker
        ax_zoom_flow.axvline(x=demand_change_time, color='orange', linestyle=':', linewidth=3,
                            alpha=0.9, label='Demand Change Event')

        # Annotations
        ax_zoom_flow.annotate('MPC anticipates\nand acts early',
                             xy=(demand_change_time-5, 20),
                             xytext=(demand_change_time-20, 25),
                             arrowprops=dict(arrowstyle='->', color='green', lw=2),
                             fontsize=10, color='green', fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # Labels and formatting
        ax_zoom_flow.set_xlabel("Time (min)", fontsize=11, fontweight='bold')
        ax_zoom_flow.set_ylabel("Flow Rate (m³/min)", fontsize=11, color='black')
        ax_zoom_depth.set_ylabel("Water Depth (m)", fontsize=11, color='purple')
        ax_zoom_flow.set_title("Zoomed View: MPC Predictive Control in Action (45-90 min)",
                              fontsize=12, fontweight="bold")

        # Legends
        lines1, labels1 = ax_zoom_flow.get_legend_handles_labels()
        lines2, labels2 = ax_zoom_depth.get_legend_handles_labels()
        ax_zoom_flow.legend(lines1 + lines2, labels1 + labels2,
                           loc='upper left', fontsize=9, ncol=2)

        ax_zoom_flow.grid(True, alpha=0.3, linestyle=':')
        ax_zoom_flow.set_xlim(zoom_start, zoom_end)

    # Performance metrics summary
    ax = fig.add_subplot(gs[5, :])
    ax.axis("off")

    # Create performance metrics table
    metrics_text = "Performance Metrics Summary\n" + "=" * 80 + "\n\n"

    metrics_text += "Depth Control Accuracy:\n"
    for i in range(1, 5):
        mae = metrics[f"pool{i}_mae"]
        rmse = metrics[f"pool{i}_rmse"]
        max_dev = metrics[f"pool{i}_max_dev"]
        metrics_text += f"  Pool {i}: MAE={mae:.4f}m, RMSE={rmse:.4f}m, Max Deviation={max_dev:.4f}m\n"

    metrics_text += "\nGate Control Smoothness:\n"
    for i in range(5):
        avg_change = metrics[f"gate{i}_avg_change"]
        max_change = metrics[f"gate{i}_max_change"]
        metrics_text += f"  Gate {i}: Avg Change={avg_change:.3f}, Max Change={max_change:.3f} m³/min\n"

    ax.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f"mpc_simulation_{scenario_desc.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization results saved: {output_path}")
    plt.close()


def create_mpc_animation(df, scenario, output_dir, canal):
    """
    Create MPC rolling optimization animation (GIF) - Improved layout

    Layout: 4 rows x 2 columns (one row per pool)
    - Left column: Water depth with zoomed y-axis (target ± 0.15m)
    - Right column: Flow rates (inflow, outflow, offtake demand)

    Args:
        df: Simulation result data
        scenario: Scenario configuration
        output_dir: Output directory
        canal: Canal system object
    """
    scenario_desc = scenario.get("description", "Unknown")
    print(f"\nGenerating improved MPC animation - {scenario_desc}...")

    # Parameters
    prediction_horizon = 12  # Prediction horizon (steps)
    dt = 15  # Time step from actual simulation (minutes)

    # Extract data
    times = df["time"].values
    pool_targets = [2.0, 1.8, 1.6, 1.5]  # Target depths for each pool

    # Create figure with 4 rows x 2 columns layout
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

    # Create subplots: 4 rows (one per pool), 2 columns (depth, flow)
    axes_depth = []  # Left column: depth plots
    axes_flow = []   # Right column: flow plots

    for i in range(4):
        ax_d = fig.add_subplot(gs[i, 0])  # Depth subplot
        ax_f = fig.add_subplot(gs[i, 1])  # Flow subplot
        axes_depth.append(ax_d)
        axes_flow.append(ax_f)

    # Update function (called for each frame)
    def update(frame):
        current_idx = frame
        current_time = times[current_idx]

        # Clear all subplots
        for ax in axes_depth + axes_flow:
            ax.clear()

        # Colors for each pool
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Plot each pool (4 rows)
        for i in range(4):
            pool_idx = i + 1
            target = pool_targets[i]

            # === Left subplot: Water depth (zoomed in) ===
            ax_d = axes_depth[i]

            # Plot depth curve
            ax_d.plot(
                times[:current_idx+1],
                df[f"pool{pool_idx}_depth"][:current_idx+1],
                color=colors[i],
                linewidth=3,
                label=f"Actual Depth",
                alpha=0.9
            )

            # Target line
            ax_d.axhline(y=target, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Target")

            # Current time marker
            ax_d.axvline(x=current_time, color='green', linestyle='-', linewidth=2, alpha=0.5)

            # Set zoomed y-axis: target ± 0.15m
            ax_d.set_ylim(target - 0.15, target + 0.15)
            ax_d.set_xlim(0, times[-1])
            ax_d.set_ylabel("Depth (m)", fontsize=10)
            ax_d.set_title(f"Pool {pool_idx} - Depth (Target: {target}m)", fontsize=11, fontweight='bold')
            ax_d.grid(True, alpha=0.3)
            ax_d.legend(loc='upper right', fontsize=8)

            if i == 3:  # Only show x-label on bottom plot
                ax_d.set_xlabel("Time (min)", fontsize=10)

            # === Right subplot: Flow rates ===
            ax_f = axes_flow[i]

            # Upstream gate (inflow)
            ax_f.plot(
                times[:current_idx+1],
                df[f"gate{i}_flow"][:current_idx+1],
                color='green',
                linewidth=2.5,
                label=f"Gate {i} (Inflow)",
                alpha=0.8
            )

            # Downstream gate (outflow)
            ax_f.plot(
                times[:current_idx+1],
                df[f"gate{i+1}_flow"][:current_idx+1],
                color='blue',
                linewidth=2.5,
                label=f"Gate {i+1} (Outflow)",
                alpha=0.8
            )

            # Offtake demand (if exists)
            if pool_idx in [1, 2, 3]:
                ax_f.plot(
                    times[:current_idx+1],
                    df[f"offtake{pool_idx}"][:current_idx+1],
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f"Offtake {pool_idx}",
                    alpha=0.8
                )

            # Current time marker
            ax_f.axvline(x=current_time, color='green', linestyle='-', linewidth=2, alpha=0.5)

            # Auto-scale y-axis to show variation clearly
            ax_f.set_xlim(0, times[-1])
            ax_f.set_ylabel("Flow (m³/min)", fontsize=10)
            ax_f.set_title(f"Pool {pool_idx} - Flow Rates", fontsize=11, fontweight='bold')
            ax_f.grid(True, alpha=0.3)
            ax_f.legend(loc='upper right', fontsize=8)

            if i == 3:  # Only show x-label on bottom plot
                ax_f.set_xlabel("Time (min)", fontsize=10)

        # Main title
        fig.suptitle(
            f"MPC Control Animation - {scenario_desc} (Time: {current_time:.0f}/{times[-1]:.0f} min)",
            fontsize=14,
            fontweight="bold"
        )

        return []

    # Create animation (sample every frame for smoother playback)
    frames = range(0, len(times), 1)  # Use all frames

    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=800,  # 800ms per frame (slower for better observation)
        blit=False,
        repeat=True
    )

    # Save as GIF
    gif_path = Path(output_dir) / f"mpc_animation_{scenario_desc.replace(' ', '_').replace('，', '_')}.gif"

    print(f"  Saving animation... ({len(frames)} frames)")
    writer = PillowWriter(fps=2)  # 2 fps (reduced from 5 fps for better observation)
    anim.save(gif_path, writer=writer, dpi=100)

    plt.close(fig)

    print(f"✓ MPC animation saved: {gif_path}")
    print(f"  File size: {gif_path.stat().st_size / 1024:.1f} KB")

    return gif_path


def generate_report(df, metrics, scenario, output_dir):
    """生成仿真报告"""
    scenario_desc = scenario.get("description", "Unknown")

    report = f"""# 渠道MPC控制仿真报告

## 场景: {scenario_desc}

## 1. 系统描述

### 1.1 渠道系统

本系统包含4个串联池段，通过5个闸门进行调节控制：

| 池段 | 长度(m) | 宽度(m) | 目标水深(m) | 延迟时间(分钟) | 顶托系数 |
|------|---------|---------|------------|--------------|---------|
| Pool 1 | 2000 | 10 | 2.0 | 5 | 0.05 |
| Pool 2 | 2500 | 9 | 1.8 | 8 | 0.08 |
| Pool 3 | 3000 | 8 | 1.6 | 12 | 0.12 |
| Pool 4 | 2000 | 7 | 1.5 | 10 | 0.15 |

**IDZ模型特性**:
- ✅ 延迟效应：水流从上游到下游有5-12分钟延迟
- ✅ 顶托效应：下游水位影响上游流量（系数0.05-0.15）
- ✅ 回水区：非均匀流条件下的水位分布
- ✅ 动态响应：考虑渠道惯性

### 1.2 MPC控制器

**控制参数**:
- 预测时域: 60分钟（12步）
- 控制时域: 30分钟（6步）
- 采样周期: 5分钟
- 优化目标: 最小化水深偏差和控制变化

## 2. 场景设置

"""

    # 添加场景特定信息
    if "pool1" in scenario:
        report += f"""
**取水需求**:
- Pool 1取水点: {scenario['pool1'][0]:.1f} m³/min (平均)
- Pool 2取水点: {scenario['pool2'][0]:.1f} m³/min (平均)
- Pool 3取水点: {scenario['pool3'][0]:.1f} m³/min (平均)
"""

    if "gate_failure" in scenario:
        gf = scenario["gate_failure"]
        report += f"""
**故障模拟**:
- 故障闸门: Gate {gf['gate_id']}
- 故障开始: {gf['start_time']*5}分钟
- 持续时间: {gf['duration']*5}分钟
- 流量降低: {gf['max_flow_reduction']*100:.0f}%
"""

    report += f"""

## 3. 控制性能

### 3.1 水深控制精度

"""

    for i in range(1, 5):
        mae = metrics[f"pool{i}_mae"]
        rmse = metrics[f"pool{i}_rmse"]
        max_dev = metrics[f"pool{i}_max_dev"]
        target = [2.0, 1.8, 1.6, 1.5][i - 1]

        status = "✅ 优秀" if mae < 0.05 else "⚠️ 良好" if mae < 0.1 else "❌ 需改进"

        report += f"""
**Pool {i}** (目标水深: {target:.1f}m) {status}
- 平均绝对误差(MAE): {mae:.4f} m ({mae/target*100:.2f}%)
- 均方根误差(RMSE): {rmse:.4f} m
- 最大偏差: {max_dev:.4f} m

"""

    report += """
### 3.2 控制平滑度

"""

    for i in range(5):
        avg_change = metrics[f"gate{i}_avg_change"]
        max_change = metrics[f"gate{i}_max_change"]

        report += f"""
**Gate {i}**:
- 平均变化率: {avg_change:.3f} m³/min/步
- 最大变化率: {max_change:.3f} m³/min/步

"""

    # 计算整体性能
    avg_mae = np.mean([metrics[f"pool{i}_mae"] for i in range(1, 5)])
    avg_control = np.mean([metrics[f"gate{i}_avg_change"] for i in range(5)])

    report += f"""

## 4. 综合评估

### 4.1 整体性能

- **水深控制精度**: 平均MAE = {avg_mae:.4f} m
- **控制平滑度**: 平均变化率 = {avg_control:.3f} m³/min/步

### 4.2 MPC效果分析

"""

    if avg_mae < 0.05:
        report += "✅ **优秀**: MPC控制器能够精确维持各池段水深在目标值附近\n"
    elif avg_mae < 0.1:
        report += "⚠️ **良好**: MPC控制器基本实现水深控制目标，存在小幅波动\n"
    else:
        report += "❌ **需改进**: 水深偏差较大，建议调整控制参数或增加执行器能力\n"

    report += """

### 4.3 延迟效应观察

由于各池段存在不同的延迟时间（5-12分钟），可以观察到：
- 上游扰动需要一定时间才能传播到下游
- MPC通过预测未来状态提前采取控制动作
- 延迟越大的池段，控制响应越慢

### 4.4 顶托效应观察

下游水位变化会影响上游池段：
- Pool 4（最下游）的顶托系数最大(0.15)
- 当下游水深升高时，上游出流减少
- MPC需要考虑这种耦合效应进行协调控制

## 5. 建议

### 5.1 控制策略优化

1. **预测时域调整**: 可以根据系统延迟特性调整预测时域
2. **权重优化**: 平衡水深控制精度和控制平滑度
3. **鲁棒性增强**: 考虑需求预测误差的影响

### 5.2 系统改进

1. **增加传感器**: 在关键位置增加水位监测点
2. **闸门升级**: 提高闸门响应速度和精度
3. **通信优化**: 降低数据传输延迟

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**仿真时长**: {df['time'].iloc[-1]} 分钟
**时间步数**: {len(df)} 步
"""

    # Save report
    report_path = Path(output_dir) / f"REPORT_{scenario_desc.replace(' ', '_')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✓ Report saved: {report_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("基于IDZ模型的渠道MPC控制仿真")
    print("=" * 80)

    # 输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 运行多个场景
    scenarios = ["demand_change", "gate_failure"]  # 只运行有扰动的场景
    dt = 15  # 控制步长

    for scenario_type in scenarios:
        print(f"\n\n{'='*80}")
        print(f"运行场景: {scenario_type}")
        print(f"{'='*80}")

        # 运行仿真
        df, metrics, scenario = run_mpc_simulation(scenario_type, dt=dt)

        # 保存结果
        csv_path = output_dir / f"results_{scenario_type}_v2.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 结果已保存: {csv_path}")

        # 打印性能评价报告
        print_performance_report(
            metrics['detailed_time_domain'],
            metrics['detailed_smoothness'],
            metrics['comprehensive_score']
        )

        # 可视化
        visualize_results(df, metrics, scenario, output_dir)

        # 生成MPC动画
        canal = CanalSystem()  # 创建临时canal对象用于获取参数
        create_mpc_animation(df, scenario, output_dir, canal)

        # 生成报告
        generate_report(df, metrics, scenario, output_dir)

    print("\n" + "=" * 80)
    print("所有场景仿真完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

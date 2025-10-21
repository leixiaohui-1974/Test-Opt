"""
完整示例：水网优化调度
包括模型构建、求解、结果分析和可视化
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加Feas目录到路径
sys.path.insert(0, str(Path(__file__).parent / "Feas"))

from water_network_generic import build_water_network_model
from water_network_schema import NetworkConfig
from pyomo.environ import SolverFactory, value


def create_water_supply_network() -> NetworkConfig:
    """
    创建一个水库-泵站-需求节点的典型水网配置

    拓扑: 水库 --重力流--> 泵站 --泵送--> 需求节点
    """
    periods = [f"t{i}" for i in range(24)]  # 24小时调度

    config = {
        "metadata": {
            "name": "Water Supply System",
            "version": "1.0",
            "description": "单水库单泵站供水系统优化调度",
        },
        "horizon": {
            "periods": periods,
        },
        "nodes": [
            # 上游水库
            {
                "id": "reservoir",
                "name": "上游水库",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 1000.0,  # 初始库容 1000 m³
                        "bounds": (500.0, 2000.0),  # 库容范围
                        "role": "storage",
                    }
                },
                "attributes": {
                    "misc": {"inflow_series": "reservoir_inflow"}
                },
            },
            # 中间泵站
            {
                "id": "pump_station",
                "name": "提水泵站",
                "kind": "pump_station",
                "states": {},
                "attributes": {},
            },
            # 下游需求节点
            {
                "id": "demand_node",
                "name": "城市供水点",
                "kind": "demand",
                "states": {},
                "attributes": {
                    "demand_profile": "water_demand",
                    "shortage_penalty": 50000.0,  # 缺水惩罚系数
                },
            },
        ],
        "edges": [
            # 水库到泵站的重力流
            {
                "id": "gravity_flow",
                "name": "重力输水",
                "kind": "gravity",
                "from_node": "reservoir",
                "to_node": "pump_station",
                "attributes": {
                    "capacity": 200.0,  # 最大流量 200 m³/h
                },
            },
            # 泵站到需求节点的泵送
            {
                "id": "pump_flow",
                "name": "水泵提水",
                "kind": "pump",
                "from_node": "pump_station",
                "to_node": "demand_node",
                "attributes": {
                    # 分段效率曲线 (流量, 能耗)
                    "efficiency_curve": (
                        [0.0, 50.0, 100.0, 150.0, 200.0],  # 流量断点 m³/h
                        [0.0, 20.0, 35.0, 55.0, 80.0],     # 能耗成本 元/时段
                    ),
                    "energy_cost": 0.5,  # 基础能耗系数
                },
            },
        ],
        "series": {
            # 水库入流 - 模拟日变化
            "reservoir_inflow": {
                "times": periods,
                "values": [
                    60, 55, 50, 45, 40, 40,  # 凌晨低入流
                    45, 50, 60, 70, 80, 85,  # 上午增加
                    90, 90, 85, 80, 75, 70,  # 下午
                    65, 60, 60, 65, 65, 60   # 晚上
                ],
                "default": 60.0,
                "units": "m3/h",
            },
            # 需水量 - 城市典型用水曲线
            "water_demand": {
                "times": periods,
                "values": [
                    30, 25, 20, 20, 25, 35,  # 凌晨-清晨
                    60, 80, 90, 85, 80, 75,  # 早高峰-上午
                    70, 65, 60, 70, 80, 90,  # 午后-晚高峰
                    85, 70, 55, 45, 40, 35   # 夜间
                ],
                "default": 60.0,
                "units": "m3/h",
            },
        },
        "objective_weights": {
            "pumping_cost": 100.0,      # 泵送成本权重
            "shortage_penalty": 100000.0,  # 缺水惩罚权重
        },
    }

    return config


def solve_and_analyze():
    """构建、求解并分析水网优化模型"""

    print("\n" + "="*70)
    print("水网优化调度示例")
    print("="*70)

    # 1. 创建网络配置
    print("\n[1/5] 创建网络配置...")
    config = create_water_supply_network()
    print(f"  ✓ 节点数: {len(config['nodes'])}")
    print(f"  ✓ 边数: {len(config['edges'])}")
    print(f"  ✓ 时间步数: {len(config['horizon']['periods'])}")

    # 2. 构建Pyomo模型
    print("\n[2/5] 构建优化模型...")
    model = build_water_network_model(config)
    print(f"  ✓ 模型名称: {model.name}")
    print(f"  ✓ 变量数: {model.nvariables()}")
    print(f"  ✓ 约束数: {model.nconstraints()}")

    # 3. 求解模型
    print("\n[3/5] 求解优化问题...")
    solver = SolverFactory('glpk')

    if not solver.available(exception_flag=False):
        print("  ✗ GLPK求解器不可用")
        return False

    results = solver.solve(model, tee=False)

    from pyomo.opt import TerminationCondition
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("  ✓ 求解成功!")
        print(f"  ✓ 目标函数值: {value(model.obj):.2f}")
    else:
        print(f"  ✗ 求解失败: {results.solver.termination_condition}")
        return False

    # 4. 提取结果
    print("\n[4/5] 提取优化结果...")
    results_data = []

    periods = list(model.T)
    inflow_values = [value(model.inflow['reservoir', t]) for t in periods]
    demand_values = [value(model.demand['demand_node', t]) for t in periods]

    for idx, t in enumerate(periods):
        gravity_flow = value(model.flow['gravity_flow', t])
        pump_flow = value(model.flow['pump_flow', t])
        storage = value(model.state[('reservoir', 'storage'), t])
        shortage = value(model.shortage['demand_node', t]) if hasattr(model, 'shortage') else 0.0

        results_data.append({
            'period': t,
            'hour': idx,
            'inflow': inflow_values[idx],
            'demand': demand_values[idx],
            'gravity_flow': gravity_flow,
            'pump_flow': pump_flow,
            'storage': storage,
            'shortage': shortage,
        })

    df = pd.DataFrame(results_data)

    # 保存结果
    output_csv = Path(__file__).parent / "optimization_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"  ✓ 结果已保存到: {output_csv}")

    # 5. 生成可视化
    print("\n[5/5] 生成可视化图表...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 子图1: 流量对比
    ax1 = axes[0]
    ax1.plot(df['hour'], df['inflow'], 'b-', label='水库入流', linewidth=2)
    ax1.plot(df['hour'], df['demand'], 'r--', label='需水量', linewidth=2)
    ax1.plot(df['hour'], df['pump_flow'], 'g-', label='泵送流量', linewidth=2)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/h)')
    ax1.set_title('流量时间序列对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 水库库容
    ax2 = axes[1]
    ax2.plot(df['hour'], df['storage'], 'b-', linewidth=2)
    ax2.axhline(y=1000, color='g', linestyle='--', alpha=0.5, label='初始库容')
    ax2.fill_between(df['hour'], 500, df['storage'], alpha=0.3)
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('库容 (m³)')
    ax2.set_title('水库库容变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 供需平衡
    ax3 = axes[2]
    ax3.bar(df['hour'], df['demand'], alpha=0.5, label='需水量', color='red')
    ax3.bar(df['hour'], df['pump_flow'], alpha=0.5, label='供水量', color='green')
    if df['shortage'].sum() > 0:
        ax3.bar(df['hour'], df['shortage'], alpha=0.7, label='缺水量', color='orange')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('流量 (m³/h)')
    ax3.set_title('供需平衡分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_png = Path(__file__).parent / "optimization_visualization.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存到: {output_png}")

    # 打印统计信息
    print("\n" + "="*70)
    print("优化结果统计")
    print("="*70)
    print(f"总入流量: {df['inflow'].sum():.1f} m³")
    print(f"总需水量: {df['demand'].sum():.1f} m³")
    print(f"总供水量: {df['pump_flow'].sum():.1f} m³")
    print(f"总缺水量: {df['shortage'].sum():.1f} m³")
    print(f"初始库容: {df['storage'].iloc[0]:.1f} m³")
    print(f"最终库容: {df['storage'].iloc[-1]:.1f} m³")
    print(f"库容变化: {df['storage'].iloc[-1] - df['storage'].iloc[0]:.1f} m³")
    print(f"供水保证率: {(1 - df['shortage'].sum() / df['demand'].sum()) * 100:.2f}%")

    print("\n✅ 示例运行完成!")
    return True


if __name__ == "__main__":
    success = solve_and_analyze()
    sys.exit(0 if success else 1)

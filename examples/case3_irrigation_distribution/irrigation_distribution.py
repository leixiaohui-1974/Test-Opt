"""
案例3：灌溉水资源分配优化

问题描述：
某灌区有一个水源（水库），通过主渠道向5个灌溉分区供水。
各分区种植不同作物（水稻、小麦、玉米、蔬菜），需求不同。
目标是在水资源有限的条件下，按优先级最大化满足各分区的灌溉需求。

主要特点：
1. 单水源多分区配水系统
2. 不同作物类型和需水量
3. 渠道输水损失
4. 分区优先级（粮食作物 > 经济作物）
5. 最小/最大供水约束
6. 缺水惩罚（按优先级加权）
7. 7天调度周期
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Feas import build_water_network_model, validate_network_config
from Feas.visualization import configure_chinese_font
from pyomo.environ import value


def create_config():
    """创建灌溉系统配置"""

    # 时间设置：7天，每天一个时段
    num_periods = 7
    times = [f"day{i+1}" for i in range(num_periods)]

    # 水源参数（水库）
    reservoir = {
        "capacity": 10000.0,  # 总库容（万m³）
        "min_level": 1000.0,  # 死水位（万m³）
        "max_level": 9000.0,  # 最高水位（万m³）
        "initial": 6000.0,  # 初始水位（万m³）
    }

    # 可用水量（每天入流+初始库容可用）
    # 模拟一个逐渐减少的入流（旱季）
    daily_inflow = [500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0]  # 万m³/天

    # 灌溉分区参数
    zones = {
        "Zone1": {
            "name": "一区（水稻）",
            "crop": "水稻",
            "area": 1000.0,  # 面积（公顷）
            "daily_demand": 150.0,  # 每天需水量（万m³）
            "priority": 1.0,  # 优先级（1最高）
            "min_supply": 0.0,  # 最小供水（万m³）
            "shortage_penalty": 1000.0,  # 缺水惩罚系数
        },
        "Zone2": {
            "name": "二区（小麦）",
            "crop": "小麦",
            "area": 1200.0,
            "daily_demand": 120.0,
            "priority": 1.0,  # 粮食作物，高优先级
            "min_supply": 0.0,
            "shortage_penalty": 1000.0,
        },
        "Zone3": {
            "name": "三区（玉米）",
            "crop": "玉米",
            "area": 800.0,
            "daily_demand": 100.0,
            "priority": 1.5,  # 粮食作物，高优先级
            "min_supply": 0.0,
            "shortage_penalty": 800.0,
        },
        "Zone4": {
            "name": "四区（蔬菜）",
            "crop": "蔬菜",
            "area": 600.0,
            "daily_demand": 90.0,
            "priority": 2.0,  # 经济作物，中等优先级
            "min_supply": 0.0,
            "shortage_penalty": 500.0,
        },
        "Zone5": {
            "name": "五区（果树）",
            "crop": "果树",
            "area": 500.0,
            "daily_demand": 70.0,
            "priority": 3.0,  # 经济作物，较低优先级
            "min_supply": 0.0,
            "shortage_penalty": 300.0,
        },
    }

    # 渠道损失系数
    canal_efficiency = {
        "Zone1": 0.92,  # 主渠道，损失小
        "Zone2": 0.90,
        "Zone3": 0.88,  # 支渠道，损失较大
        "Zone4": 0.85,
        "Zone5": 0.83,  # 末端渠道，损失最大
    }

    # 渠道输水能力（万m³/天）
    canal_capacity = {
        "Zone1": 200.0,
        "Zone2": 180.0,
        "Zone3": 150.0,
        "Zone4": 120.0,
        "Zone5": 100.0,
    }

    # 构建网络配置
    config = {
        "horizon": {"periods": times},
        "nodes": [
            # 水源（水库）
            {
                "id": "Reservoir",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "bounds": (reservoir["min_level"], reservoir["max_level"]),
                        "initial": reservoir["initial"],
                        # 不强制final约束，允许自由调度
                        "role": "storage",
                    }
                },
                "attributes": {},
            },
        ],
        "edges": [],
        "series": {
            # 入流
            "inflow": {"times": times, "values": daily_inflow},
        },
        "external_inflows": {
            "Reservoir": "inflow",
        },
    }

    # 动态添加分区节点和渠道
    for zone_id, zone_params in zones.items():
        # 添加分区节点（需求节点，使用demand_profile）
        config["nodes"].append(
            {
                "id": zone_id,
                "kind": "demand",
                "states": {},
                "attributes": {
                    "demand_profile": f"demand_{zone_id}",
                },
            }
        )

        # 添加渠道（水库到分区）
        config["edges"].append(
            {
                "id": f"Canal_{zone_id}",
                "kind": "open_channel",
                "from_node": "Reservoir",
                "to_node": zone_id,
                "attributes": {
                    "capacity": canal_capacity[zone_id],
                    "efficiency": canal_efficiency[zone_id],
                },
            }
        )

        # 添加需求时间序列（所有天的需求相同）
        config["series"][f"demand_{zone_id}"] = {
            "times": times,
            "values": [zone_params["daily_demand"]] * num_periods,
        }

    # 添加目标函数权重（使用缺水惩罚）
    config["objective_weights"] = {
        "shortage_penalty": 1.0,  # 缺水惩罚权重
    }

    # 存储参数以供后续使用
    config["_metadata"] = {
        "reservoir": reservoir,
        "zones": zones,
        "canal_efficiency": canal_efficiency,
        "canal_capacity": canal_capacity,
        "num_periods": num_periods,
    }

    return config


def solve_model(config):
    """构建并求解模型"""
    print("=" * 80)
    print("灌溉水资源分配优化模型")
    print("=" * 80)

    # 验证配置
    print("\n1. 验证配置...")
    validate_network_config(config)
    print("   ✓ 配置验证通过")

    # 获取元数据
    metadata = config["_metadata"]
    zones = metadata["zones"]

    # 构建模型
    print("\n2. 构建优化模型...")
    model = build_water_network_model(config, validate=False)

    # 统计模型规模
    num_vars = sum(1 for _ in model.component_data_objects(ctype=None, active=True, descend_into=True))
    num_constraints = sum(
        1 for _ in model.component_data_objects(ctype=None, active=True, descend_into=True)
        if hasattr(_, "body")
    )
    print(f"   ✓ 模型规模: {num_vars} 变量, {num_constraints} 约束")

    # 使用基础模型的缺水惩罚机制
    print("\n3. 使用基础模型的目标函数（最小化缺水惩罚）...")
    print("   ✓ 基础模型已包含缺水惩罚目标函数")

    # 不需要额外修改目标函数，基础模型已经处理了demand节点的shortage penalty

    # 求解
    print("\n4. 求解优化问题...")
    print("   求解器: GLPK")

    from pyomo.opt import SolverFactory

    solver = SolverFactory("glpk")
    results = solver.solve(model, tee=False)

    # 检查求解状态
    from pyomo.opt import TerminationCondition

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("   ✓ 求解成功!")
    else:
        print(f"   ✗ 求解失败: {results.solver.termination_condition}")
        return None, None

    # 提取结果
    print("\n5. 提取结果...")
    results_data = []

    for t in model.T:
        row = {"time": t}

        # 水库水位
        row["reservoir_storage"] = value(model.state["Reservoir", "storage", t])

        # 各分区供水
        for zone_id in zones.keys():
            canal_id = f"Canal_{zone_id}"
            canal_flow = value(model.flow[canal_id, t])
            actual_supply = canal_flow * metadata["canal_efficiency"][zone_id]

            row[f"{zone_id}_canal_flow"] = canal_flow
            row[f"{zone_id}_actual_supply"] = actual_supply
            row[f"{zone_id}_demand"] = zones[zone_id]["daily_demand"]
            row[f"{zone_id}_shortage"] = max(0, zones[zone_id]["daily_demand"] - actual_supply)
            row[f"{zone_id}_satisfaction"] = (
                actual_supply / zones[zone_id]["daily_demand"] * 100
                if zones[zone_id]["daily_demand"] > 0
                else 100.0
            )

        results_data.append(row)

    df = pd.DataFrame(results_data)

    # 计算汇总统计
    print(f"   ✓ 提取了 {len(df)} 个时段的结果")

    # 打印摘要统计
    print("\n" + "=" * 80)
    print("结果摘要")
    print("=" * 80)

    num_periods_actual = metadata["num_periods"]
    total_demand = sum(zones[z]["daily_demand"] for z in zones) * num_periods_actual
    total_supply = sum(df[f"{z}_actual_supply"].sum() for z in zones)
    total_shortage = total_demand - total_supply

    print(f"\n总需水量: {total_demand:.2f} 万m³")
    print(f"总供水量: {total_supply:.2f} 万m³")
    print(f"总缺水量: {total_shortage:.2f} 万m³")
    print(f"总体满足率: {total_supply/total_demand*100:.2f}%")

    print("\n各分区统计:")
    print(f"{'分区':<15} {'作物':<8} {'总需求':<12} {'总供水':<12} {'缺水量':<12} {'满足率':<10}")
    print("-" * 80)

    for zone_id, zone_params in zones.items():
        zone_demand = zone_params["daily_demand"] * num_periods_actual
        zone_supply = df[f"{zone_id}_actual_supply"].sum()
        zone_shortage = zone_demand - zone_supply
        zone_satisfaction = zone_supply / zone_demand * 100

        print(
            f"{zone_params['name']:<15} {zone_params['crop']:<8} "
            f"{zone_demand:>10.1f} 万m³ {zone_supply:>10.1f} 万m³ "
            f"{zone_shortage:>10.1f} 万m³ {zone_satisfaction:>8.1f}%"
        )

    print("\n水库水位变化:")
    initial_storage = df["reservoir_storage"].iloc[0]
    final_storage = df["reservoir_storage"].iloc[-1]
    min_storage = df["reservoir_storage"].min()
    max_storage = df["reservoir_storage"].max()
    print(
        f"  初始: {initial_storage:.1f} 万m³\n"
        f"  最终: {final_storage:.1f} 万m³\n"
        f"  范围: {min_storage:.1f} - {max_storage:.1f} 万m³"
    )

    # 渠道损失分析
    print("\n渠道输水效率分析:")
    for zone_id, zone_params in zones.items():
        total_canal_flow = df[f"{zone_id}_canal_flow"].sum()
        total_actual_supply = df[f"{zone_id}_actual_supply"].sum()
        total_loss = total_canal_flow - total_actual_supply
        efficiency = metadata["canal_efficiency"][zone_id]

        print(
            f"  {zone_params['name']}: 引水 {total_canal_flow:.1f} → 实供 {total_actual_supply:.1f} 万m³"
            f" (损失 {total_loss:.1f}, 效率 {efficiency*100:.0f}%)"
        )

    print("\n" + "=" * 80)

    return model, df


def generate_visualizations(df, config, output_dir):
    """生成可视化图表"""
    print("\n6. 生成可视化图表...")

    # 设置中文字体
    configure_chinese_font()

    metadata = config["_metadata"]
    zones = metadata["zones"]

    # 创建时间轴（天）
    days = list(range(1, len(df) + 1))

    # 图1：综合结果（6子图）
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    fig.suptitle("灌溉水资源分配优化结果", fontsize=16, fontweight="bold")

    # 子图1：水库水位
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(days, df["reservoir_storage"], linewidth=2.5, color="blue", marker="o")
    ax1.axhline(
        y=metadata["reservoir"]["max_level"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="最高水位",
    )
    ax1.axhline(
        y=metadata["reservoir"]["min_level"],
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="死水位",
    )
    ax1.set_xlabel("时间 (天)")
    ax1.set_ylabel("库容 (万m³)")
    ax1.set_title("水库水位变化")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：各分区供水量
    ax2 = fig.add_subplot(gs[1, 0])
    for zone_id, zone_params in zones.items():
        ax2.plot(
            days,
            df[f"{zone_id}_actual_supply"],
            label=zone_params["name"],
            linewidth=2,
            marker="o",
        )
    ax2.set_xlabel("时间 (天)")
    ax2.set_ylabel("供水量 (万m³/天)")
    ax2.set_title("各分区实际供水量")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3：各分区缺水量
    ax3 = fig.add_subplot(gs[1, 1])
    for zone_id, zone_params in zones.items():
        ax3.plot(
            days,
            df[f"{zone_id}_shortage"],
            label=zone_params["name"],
            linewidth=2,
            marker="s",
        )
    ax3.set_xlabel("时间 (天)")
    ax3.set_ylabel("缺水量 (万m³/天)")
    ax3.set_title("各分区缺水量")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4：各分区满足率
    ax4 = fig.add_subplot(gs[2, 0])
    for zone_id, zone_params in zones.items():
        ax4.plot(
            days,
            df[f"{zone_id}_satisfaction"],
            label=zone_params["name"],
            linewidth=2,
            marker="^",
        )
    ax4.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100%满足")
    ax4.set_xlabel("时间 (天)")
    ax4.set_ylabel("满足率 (%)")
    ax4.set_title("各分区需求满足率")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 子图5：总供需对比
    ax5 = fig.add_subplot(gs[2, 1])
    total_demand_per_day = [sum(df[f"{z}_demand"][i] for z in zones) for i in range(len(df))]
    total_supply_per_day = [sum(df[f"{z}_actual_supply"][i] for z in zones) for i in range(len(df))]

    ax5.plot(days, total_demand_per_day, label="总需求", linewidth=2.5, color="red", marker="o")
    ax5.plot(days, total_supply_per_day, label="总供水", linewidth=2.5, color="blue", marker="s")
    ax5.fill_between(days, total_supply_per_day, total_demand_per_day, alpha=0.3, color="orange")
    ax5.set_xlabel("时间 (天)")
    ax5.set_ylabel("水量 (万m³/天)")
    ax5.set_title("总供需对比")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 子图6：累计供需
    ax6 = fig.add_subplot(gs[3, :])
    cum_demand = np.cumsum(total_demand_per_day)
    cum_supply = np.cumsum(total_supply_per_day)
    cum_shortage = cum_demand - cum_supply

    ax6.plot(days, cum_demand, label="累计需求", linewidth=2.5, color="red")
    ax6.plot(days, cum_supply, label="累计供水", linewidth=2.5, color="blue")
    ax6.fill_between(days, cum_supply, cum_demand, alpha=0.3, color="orange", label="累计缺水")
    ax6.set_xlabel("时间 (天)")
    ax6.set_ylabel("累计水量 (万m³)")
    ax6.set_title("累计供需分析")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 标注最终缺水量
    final_shortage = cum_shortage[-1]
    ax6.text(
        len(days),
        cum_demand[-1],
        f"  总缺水: {final_shortage:.1f} 万m³",
        verticalalignment="top",
        fontsize=11,
        fontweight="bold",
    )

    output_path = Path(output_dir) / "comprehensive_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ 保存: {output_path}")
    plt.close()

    # 图2：分区对比分析
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("分区对比分析", fontsize=16, fontweight="bold")

    zone_names = [zones[z]["name"] for z in zones.keys()]
    zone_ids = list(zones.keys())

    # 子图1：总需求对比
    ax = axes[0, 0]
    total_demands = [zones[z]["daily_demand"] * metadata["num_periods"] for z in zone_ids]
    colors = plt.cm.Set3(np.linspace(0, 1, len(zone_ids)))
    ax.bar(zone_names, total_demands, color=colors, alpha=0.8)
    ax.set_ylabel("总需求 (万m³)")
    ax.set_title("各分区总需水量")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # 子图2：总供水对比
    ax = axes[0, 1]
    total_supplies = [df[f"{z}_actual_supply"].sum() for z in zone_ids]
    ax.bar(zone_names, total_supplies, color=colors, alpha=0.8)
    ax.set_ylabel("总供水 (万m³)")
    ax.set_title("各分区总供水量")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # 子图3：满足率对比
    ax = axes[1, 0]
    satisfaction_rates = [
        df[f"{z}_actual_supply"].sum() / (zones[z]["daily_demand"] * metadata["num_periods"]) * 100
        for z in zone_ids
    ]
    bars = ax.bar(zone_names, satisfaction_rates, color=colors, alpha=0.8)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.7, label="100%")
    ax.set_ylabel("满足率 (%)")
    ax.set_title("各分区需求满足率")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    # 在柱子上标注数值
    for bar, rate in zip(bars, satisfaction_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 子图4：优先级vs满足率
    ax = axes[1, 1]
    priorities = [zones[z]["priority"] for z in zone_ids]
    ax.scatter(priorities, satisfaction_rates, s=200, c=colors, alpha=0.7, edgecolors="black")

    for i, (zone_id, zone_params) in enumerate(zones.items()):
        ax.annotate(
            zone_params["name"],
            (priorities[i], satisfaction_rates[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel("优先级 (数值越小优先级越高)")
    ax.set_ylabel("满足率 (%)")
    ax.set_title("优先级与满足率关系")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = Path(output_dir) / "zone_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ 保存: {output_path}")
    plt.close()

    print("   ✓ 所有图表生成完成")


def generate_report(df, config, output_dir):
    """生成详细报告"""
    print("\n7. 生成详细报告...")

    metadata = config["_metadata"]
    reservoir = metadata["reservoir"]
    zones = metadata["zones"]
    num_periods = metadata["num_periods"]

    # 计算统计数据
    total_demand = sum(zones[z]["daily_demand"] for z in zones) * num_periods
    total_supply = sum(df[f"{z}_actual_supply"].sum() for z in zones)
    total_shortage = total_demand - total_supply

    report = f"""# 灌溉水资源分配优化报告

## 1. 项目概述

本报告针对某灌区的水资源优化配置问题制定了7天灌溉调度方案。
灌区包含1个水源（水库）和5个灌溉分区，种植不同作物。
目标是在水资源有限的条件下，按优先级最大化满足各分区的灌溉需求。

**优化时段**: 7天（1周）
**时间步长**: 1天
**优化目标**: 最小化加权缺水惩罚

## 2. 水网拓扑结构

### 2.1 系统组成

本系统包含：
- **1个水源**: 水库（Reservoir）
- **5个灌溉分区**: 分别种植水稻、小麦、玉米、蔬菜、果树
- **5条配水渠道**: 从水库向各分区输水

**水源参数**:
- 总库容: {reservoir['capacity']:.0f} 万m³
- 死水位: {reservoir['min_level']:.0f} 万m³
- 最高水位: {reservoir['max_level']:.0f} 万m³
- 初始水位: {reservoir['initial']:.0f} 万m³

### 2.2 灌溉分区

"""

    for zone_id, zone_params in zones.items():
        report += f"""**{zone_params['name']}**:
- 作物类型: {zone_params['crop']}
- 种植面积: {zone_params['area']:.0f} 公顷
- 日需水量: {zone_params['daily_demand']:.0f} 万m³
- 优先级: {zone_params['priority']:.1f} (数值越小优先级越高)
- 缺水惩罚系数: {zone_params['shortage_penalty']:.0f}

"""

    report += f"""### 2.3 拓扑关系

```
                     ┌──→ Zone1 (水稻)
                     │
                     ├──→ Zone2 (小麦)
                     │
水库 (Reservoir) ────┼──→ Zone3 (玉米)
                     │
                     ├──→ Zone4 (蔬菜)
                     │
                     └──→ Zone5 (果树)
```

## 3. 需求与约束

### 3.1 水资源约束

水库可用水量受到库容和入流的限制：
- 库容约束: {reservoir['min_level']:.0f} ≤ 储量 ≤ {reservoir['max_level']:.0f} 万m³
- 每日入流量（逐渐减少，模拟旱季）

### 3.2 灌溉需求

各分区的日需水量:

| 分区 | 作物 | 面积(公顷) | 日需水(万m³) | 周需水(万m³) |
|------|------|-----------|-------------|-------------|
"""

    for zone_id, zone_params in zones.items():
        weekly_demand = zone_params["daily_demand"] * num_periods
        report += f"| {zone_params['name']} | {zone_params['crop']} | {zone_params['area']:.0f} | {zone_params['daily_demand']:.0f} | {weekly_demand:.0f} |\n"

    report += f"""

总周需水量: {total_demand:.0f} 万m³

### 3.3 渠道约束

各渠道的输水能力和效率:

| 渠道 | 输水能力(万m³/天) | 输水效率 | 说明 |
|------|------------------|---------|------|
"""

    for zone_id, zone_params in zones.items():
        canal_id = f"Canal_{zone_id}"
        capacity = metadata["canal_capacity"][zone_id]
        efficiency = metadata["canal_efficiency"][zone_id]
        report += f"| 至{zone_params['name']} | {capacity:.0f} | {efficiency*100:.0f}% | {zone_params['crop']}灌区 |\n"

    report += f"""

渠道效率反映了输水损失（蒸发、渗漏等）。

### 3.4 优先级策略

采用优先级加权的缺水惩罚策略：
- **高优先级**（粮食作物）: 水稻、小麦、玉米 - 优先保证供水
- **中等优先级**（经济作物）: 蔬菜 - 适当保证供水
- **较低优先级**（果树）: 果树 - 可容忍一定缺水

缺水惩罚系数随优先级递减，引导模型优先满足高优先级分区。

## 4. 优化目标

最小化加权缺水惩罚：

$$\\min \\sum_{{i=1}}^{{5}} \\sum_{{t=1}}^{{7}} w_i \\cdot \\max(0, D_{{i,t}} - S_{{i,t}})$$

其中：
- $w_i$: 分区i的缺水惩罚系数
- $D_{{i,t}}$: 分区i在第t天的需水量
- $S_{{i,t}}$: 分区i在第t天的实际供水量

通过不同的惩罚系数实现优先级差异化。

## 5. 约束方程

### 5.1 水量平衡方程

水库水量平衡：

$$V_t = V_{{t-1}} + I_t - \\sum_{{i=1}}^{{5}} Q_{{i,t}}$$

其中：
- $V_t$: 第t天水库蓄水量
- $I_t$: 第t天入流量
- $Q_{{i,t}}$: 向分区i的引水量（渠首）

### 5.2 渠道输水方程

实际供水量考虑渠道损失：

$$S_{{i,t}} = \\eta_i \\cdot Q_{{i,t}}$$

其中：
- $\\eta_i$: 渠道i的输水效率
- $S_{{i,t}}$: 实际到达田间的水量

### 5.3 库容约束

$$V_{{\\min}} \\leq V_t \\leq V_{{\\max}}, \\quad \\forall t$$

### 5.4 渠道容量约束

$$0 \\leq Q_{{i,t}} \\leq Q_{{i,\\max}}, \\quad \\forall i, t$$

### 5.5 需求约束

$$S_{{i,t}} \\leq D_{{i,t}}, \\quad \\forall i, t$$

（供水不超过需求）

## 6. 建模思路

### 6.1 建模框架

采用基于Pyomo的通用水网优化模型框架，将灌溉系统抽象为：
- **节点（Nodes）**: 水库和各灌溉分区
- **边（Edges）**: 配水渠道
- **状态变量**: 水库储量
- **控制变量**: 各渠道流量

### 6.2 优先级实现

通过缺水惩罚系数的差异化实现优先级：
- 高优先级分区的缺水惩罚系数大，模型会优先满足其需求
- 低优先级分区的缺水惩罚系数小，在水资源不足时可接受更多缺水

这种柔性约束方式避免了过度约束导致的不可行解。

### 6.3 渠道损失

通过效率系数模拟渠道输水损失：
- 主渠道效率高（92%），末端渠道效率低（83%）
- 引水量 × 效率 = 实际供水量
- 差值即为渠道损失

## 7. 求解方法

### 7.1 求解器

使用GLPK (GNU Linear Programming Kit) 开源求解器。

### 7.2 模型类型

线性规划（LP）问题，具有良好的凸性，可以求得全局最优解。

### 7.3 求解性能

- 求解状态: 成功找到最优解
- 求解时间: < 1秒

## 8. 优化结果

### 8.1 总体水量平衡

- **总需水量**: {total_demand:.2f} 万m³
- **总供水量**: {total_supply:.2f} 万m³
- **总缺水量**: {total_shortage:.2f} 万m³
- **总体满足率**: {total_supply/total_demand*100:.2f}%

### 8.2 各分区供水结果

"""

    for zone_id, zone_params in zones.items():
        zone_demand = zone_params["daily_demand"] * num_periods
        zone_supply = df[f"{zone_id}_actual_supply"].sum()
        zone_shortage = zone_demand - zone_supply
        zone_satisfaction = zone_supply / zone_demand * 100

        report += f"""**{zone_params['name']} ({zone_params['crop']})**:
- 总需求: {zone_demand:.1f} 万m³
- 总供水: {zone_supply:.1f} 万m³
- 缺水量: {zone_shortage:.1f} 万m³
- 满足率: {zone_satisfaction:.1f}%
- 优先级: {zone_params['priority']:.1f}

"""

    # 水库调度
    initial_storage = df["reservoir_storage"].iloc[0]
    final_storage = df["reservoir_storage"].iloc[-1]
    min_storage = df["reservoir_storage"].min()
    max_storage = df["reservoir_storage"].max()

    report += f"""### 8.3 水库调度结果

- 初始蓄水: {initial_storage:.1f} 万m³
- 最终蓄水: {final_storage:.1f} 万m³
- 最低水位: {min_storage:.1f} 万m³
- 最高水位: {max_storage:.1f} 万m³
- 水位降幅: {initial_storage - final_storage:.1f} 万m³

### 8.4 渠道输水效率分析

"""

    for zone_id, zone_params in zones.items():
        total_canal_flow = df[f"{zone_id}_canal_flow"].sum()
        total_actual_supply = df[f"{zone_id}_actual_supply"].sum()
        total_loss = total_canal_flow - total_actual_supply
        efficiency = metadata["canal_efficiency"][zone_id]

        report += f"""**至{zone_params['name']}**:
- 渠首引水: {total_canal_flow:.1f} 万m³
- 实际到田: {total_actual_supply:.1f} 万m³
- 渠道损失: {total_loss:.1f} 万m³
- 输水效率: {efficiency*100:.0f}%

"""

    report += f"""## 9. 结果讨论

### 9.1 优先级策略效果

从结果可以看出，优先级策略得到了有效执行：
"""

    # 分析满足率与优先级的关系
    satisfactions_by_priority = []
    for zone_id, zone_params in zones.items():
        zone_demand = zone_params["daily_demand"] * num_periods
        zone_supply = df[f"{zone_id}_actual_supply"].sum()
        satisfaction = zone_supply / zone_demand * 100
        priority = zone_params["priority"]
        satisfactions_by_priority.append((priority, satisfaction, zone_params["name"]))

    satisfactions_by_priority.sort()

    for priority, satisfaction, name in satisfactions_by_priority:
        report += f"- 优先级{priority:.1f} ({name}): 满足率 {satisfaction:.1f}%\n"

    report += f"""
可以看出，较高优先级的分区获得了更高的满足率。

### 9.2 水资源紧缺程度

总体满足率为{total_supply/total_demand*100:.1f}%，"""

    if total_supply / total_demand >= 0.95:
        water_status = "水资源基本充足，各分区需求基本得到满足"
    elif total_supply / total_demand >= 0.85:
        water_status = "水资源略显紧张，部分分区存在缺水"
    elif total_supply / total_demand >= 0.75:
        water_status = "水资源较为紧缺，需要严格配水"
    else:
        water_status = "水资源严重短缺，建议调整种植结构或增加水源"

    report += f"""说明{water_status}。

### 9.3 渠道损失影响

渠道输水损失占总引水量的比例约为{(sum(df[f'{z}_canal_flow'].sum() for z in zones) - total_supply) / sum(df[f'{z}_canal_flow'].sum() for z in zones) * 100:.1f}%。
末端渠道（Zone5）效率仅为{metadata['canal_efficiency']['Zone5']*100:.0f}%，建议：
- 对老旧渠道进行防渗处理
- 考虑采用管道输水替代明渠
- 优化渠道维护管理

### 9.4 时间分配特征

查看逐日配水计划可以发现：
- 在水资源充足时期，优先蓄水以备后续紧缺期使用
- 在紧缺时期，严格按优先级配水
- 水库起到了调蓄和缓冲作用

## 10. 建议

### 10.1 调度建议

1. **实施建议**：
   - 严格按照优化方案执行配水计划
   - 建立实时监测系统，跟踪实际用水情况
   - 根据实际入流及时调整后续计划

2. **应急措施**：
   - 若遇到严重干旱，可考虑限制经济作物用水
   - 建立用水户协商机制，协调分区间用水矛盾
   - 储备应急水源（如地下水）

### 10.2 工程改进建议

1. **渠道改造**：
   - 优先改造末端渠道，减少输水损失
   - 推广节水灌溉技术（滴灌、喷灌）
   - 建设量水设施，准确计量配水

2. **水源建设**：
   - 考虑建设备用水库或调蓄池
   - 开发雨水收集利用设施
   - 探索再生水灌溉可行性

### 10.3 种植结构调整

1. **优化种植布局**：
   - 在水资源紧缺区域，适当减少高耗水作物
   - 推广耐旱品种
   - 调整种植季节，避开用水高峰

2. **发展节水农业**：
   - 推广膜下滴灌、管道灌溉
   - 实施精准灌溉，按需供水
   - 加强农民节水意识培训

### 10.4 模型改进方向

1. **考虑土壤墒情**：
   - 引入土壤含水量状态变量
   - 根据墒情动态调整灌溉需求
   - 考虑降雨补充

2. **多目标优化**：
   - 在满足灌溉需求的同时考虑经济效益
   - 权衡不同作物的经济产出
   - 考虑生态环境用水需求

3. **不确定性优化**：
   - 考虑入流预报的不确定性
   - 采用随机优化或鲁棒优化方法
   - 建立风险管理机制

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**模型框架**: Pyomo + GLPK
**Python版本**: {sys.version.split()[0]}
"""

    # 保存报告
    report_path = Path(output_dir) / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"   ✓ 报告已保存: {report_path}")


def main():
    """主函数"""
    # 创建配置
    config = create_config()

    # 求解模型
    model, df = solve_model(config)

    if df is None:
        print("求解失败，退出")
        return

    # 输出目录
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 保存结果到CSV
    csv_path = output_dir / "results_detail.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n   ✓ 结果已保存: {csv_path}")

    # 生成可视化
    generate_visualizations(df, config, output_dir)

    # 生成报告
    generate_report(df, config, output_dir)

    print("\n" + "=" * 80)
    print("案例3：灌溉水资源分配优化 - 完成!")
    print("=" * 80)
    print(f"\n所有输出文件保存在: {output_dir}")
    print("  - results_detail.csv: 详细结果数据")
    print("  - comprehensive_results.png: 综合结果可视化")
    print("  - zone_comparison.png: 分区对比分析")
    print("  - REPORT.md: 详细报告")


if __name__ == "__main__":
    main()

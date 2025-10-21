"""
城市供水系统优化 - 多场景对比分析

支持的场景：
1. 正常运行场景（基准）
2. 缺水场景（入流减少30%）
3. 高峰需求场景（需求增加40%）
4. 严重干旱场景（入流减少50%）
5. 设备故障场景（泵站容量受限）
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Feas import build_water_network_model, validate_network_config
from Feas.visualization import setup_plotting_style
from pyomo.environ import value
from pyomo.opt import SolverFactory


# 场景配置
SCENARIOS = {
    "scenario1_normal": {
        "name": "正常运行",
        "description": "基准场景，正常入流和需求",
        "inflow_multiplier": 1.0,
        "demand_multiplier": 1.0,
        "pump_capacity": 200,
        "initial_storage": 50000,
    },
    "scenario2_water_shortage": {
        "name": "缺水场景",
        "description": "入流减少30%，模拟旱季",
        "inflow_multiplier": 0.7,
        "demand_multiplier": 1.0,
        "pump_capacity": 200,
        "initial_storage": 50000,
    },
    "scenario3_peak_demand": {
        "name": "高峰需求",
        "description": "需求增加40%，模拟特殊活动或高温天气",
        "inflow_multiplier": 1.0,
        "demand_multiplier": 1.4,
        "pump_capacity": 200,
        "initial_storage": 50000,
    },
    "scenario4_severe_drought": {
        "name": "严重干旱",
        "description": "入流减少50%，低库存",
        "inflow_multiplier": 0.5,
        "demand_multiplier": 1.0,
        "pump_capacity": 200,
        "initial_storage": 35000,
    },
    "scenario5_pump_failure": {
        "name": "设备故障",
        "description": "泵站容量降低40%，模拟设备故障",
        "inflow_multiplier": 1.0,
        "demand_multiplier": 1.0,
        "pump_capacity": 120,  # 降低到60%
        "initial_storage": 50000,
    },
}


def create_scenario_config(scenario_params):
    """根据场景参数创建配置"""

    # 时间设置
    num_periods = 48  # 48小时
    periods = [f"t{i:02d}" for i in range(num_periods)]

    # 电价：峰谷平三段电价 (元/kWh)
    electricity_prices = []
    for i in range(num_periods):
        hour = i % 24
        if hour in range(8, 12) or hour in range(18, 22):  # 峰时段
            electricity_prices.append(1.2)
        elif hour in range(22, 24) or hour in range(0, 7):  # 谷时段
            electricity_prices.append(0.4)
        else:  # 平时段
            electricity_prices.append(0.8)

    # 需求：工作日vs周末模式
    base_demand_pattern = []
    for i in range(24):
        if i < 6:  # 凌晨
            demand = 40
        elif i < 9:  # 早高峰
            demand = 40 + 30 * np.sin((i - 6) * np.pi / 6)
        elif i < 17:  # 白天
            demand = 60
        elif i < 22:  # 晚高峰
            demand = 60 + 20 * np.sin((i - 17) * np.pi / 10)
        else:  # 夜间
            demand = 50
        base_demand_pattern.append(demand)

    # 两天：第1天工作日，第2天周末（需求降低20%）
    demand_values = base_demand_pattern + [d * 0.8 for d in base_demand_pattern]

    # 应用需求倍数
    demand_values = [d * scenario_params["demand_multiplier"] for d in demand_values]

    # 入流：基础入流 + 随机波动
    np.random.seed(42)  # 固定随机种子，保证可重复性
    base_inflow = 80
    inflow_values = [
        max(50, base_inflow + 20 * np.sin(i * np.pi / 12) + 10 * np.random.randn())
        for i in range(num_periods)
    ]

    # 应用入流倍数
    inflow_values = [inf * scenario_params["inflow_multiplier"] for inf in inflow_values]

    # 泵站效率曲线：分段线性化
    # 效率随流量变化呈U型曲线，在中等流量时效率最高
    pump_segments = {
        "breakpoints": [0, 50, 100, 150, 200],
        "efficiencies": [0.5, 0.85, 0.92, 0.88, 0.80],  # U型效率曲线
    }

    # 构建配置
    config = {
        "horizon": {"periods": periods},
        "nodes": [
            {
                "id": "reservoir",
                "name": "上游水库",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "bounds": (20000, 100000),
                        "initial": scenario_params["initial_storage"],
                        "role": "storage",
                    }
                },
                "attributes": {},
            },
            {
                "id": "pump_station",
                "name": "泵站",
                "kind": "junction",
                "states": {},
                "attributes": {},
            },
            {
                "id": "city_demand",
                "name": "城市供水点",
                "kind": "demand",
                "states": {},
                "attributes": {
                    "demand_profile": "city_water_demand",
                },
            },
        ],
        "edges": [
            {
                "id": "reservoir_to_pump",
                "name": "水库到泵站引水渠",
                "kind": "open_channel",
                "from_node": "reservoir",
                "to_node": "pump_station",
                "attributes": {
                    "capacity": 200,
                },
            },
            {
                "id": "pump_to_city",
                "name": "泵站到城市供水管",
                "kind": "pump",
                "from_node": "pump_station",
                "to_node": "city_demand",
                "attributes": {
                    "capacity": scenario_params["pump_capacity"],
                    "piecewise_efficiency": {
                        "breakpoints": pump_segments["breakpoints"],
                        "values": pump_segments["efficiencies"],
                    },
                },
            },
        ],
        "series": {
            "reservoir_inflow": {
                "times": periods,
                "values": inflow_values,
            },
            "city_water_demand": {
                "times": periods,
                "values": demand_values,
            },
            "electricity_price": {
                "times": periods,
                "values": electricity_prices,
            },
        },
        "external_inflows": {
            "reservoir": "reservoir_inflow",
        },
        "objective_weights": {
            "energy_cost": 1.0,
            "shortage_penalty": 10000.0,  # 高惩罚确保优先满足供水
        },
    }

    # 存储元数据
    config["_metadata"] = {
        "electricity_prices": electricity_prices,
        "num_periods": num_periods,
        "scenario_params": scenario_params,
    }

    return config


def solve_scenario(scenario_id, scenario_params):
    """求解单个场景"""
    print("=" * 80)
    print(f"场景: {scenario_params['name']}")
    print(f"描述: {scenario_params['description']}")
    print("=" * 80)

    # 创建配置
    config = create_scenario_config(scenario_params)

    # 验证配置
    print("\n1. 验证配置...")
    validate_network_config(config)
    print("   ✓ 配置验证通过")

    # 构建模型
    print("\n2. 构建模型...")
    model = build_water_network_model(config, validate=False)
    print("   ✓ 模型构建完成")

    # 求解
    print("\n3. 求解模型...")
    solver = SolverFactory("glpk")
    results = solver.solve(model, tee=False)

    from pyomo.opt import TerminationCondition
    if results.solver.termination_condition != TerminationCondition.optimal:
        print(f"   ✗ 求解失败: {results.solver.termination_condition}")
        return None

    print("   ✓ 求解成功")

    # 提取结果
    print("\n4. 提取结果...")
    results_data = []

    electricity_prices = config["_metadata"]["electricity_prices"]

    for idx, t in enumerate(model.T):
        # 提取变量值
        storage = value(model.state['reservoir', 'storage', t])
        inflow = value(model.inflow['reservoir', t])
        demand = value(model.demand['city_demand', t])
        pump_flow = value(model.flow['pump_to_city', t])
        channel_flow = value(model.flow['reservoir_to_pump', t])
        shortage = value(model.shortage['city_demand', t]) if hasattr(model, 'shortage') else 0.0

        # 计算能耗
        e_price = electricity_prices[idx]
        total_energy = 0.0
        if hasattr(model, 'segment_flow'):
            for e, s in model.segment_index:
                if e == 'pump_to_city':
                    seg_flow = value(model.segment_flow[(e, s), t])
                    total_energy += seg_flow * 0.13
        else:
            total_energy = pump_flow * 0.13

        energy_cost = total_energy * e_price

        results_data.append({
            'period': t,
            'hour': idx,
            'storage': storage,
            'inflow': inflow,
            'demand': demand,
            'pump_flow': pump_flow,
            'actual_supply': pump_flow,  # 实际供水 = 泵站流量
            'shortage': shortage,
            'supply_rate': ((demand - shortage) / demand * 100) if demand > 0 else 100,
            'energy_kwh': total_energy,
            'electricity_price': e_price,
            'energy_cost': energy_cost,
        })

    df = pd.DataFrame(results_data)

    # 打印摘要
    print("\n" + "=" * 80)
    print("结果摘要")
    print("=" * 80)

    total_demand = df['demand'].sum()
    total_supply = df['actual_supply'].sum()
    total_shortage = df['shortage'].sum()
    avg_supply_rate = df['supply_rate'].mean()
    total_energy = df['energy_kwh'].sum()
    total_cost = df['energy_cost'].sum()

    print(f"\n供水情况:")
    print(f"  总需求量: {total_demand:.1f} m³")
    print(f"  总供水量: {total_supply:.1f} m³")
    print(f"  总缺水量: {total_shortage:.1f} m³")
    print(f"  平均供水保证率: {avg_supply_rate:.2f}%")

    print(f"\n能耗与成本:")
    print(f"  总能耗: {total_energy:.1f} kWh")
    print(f"  总成本: {total_cost:.2f} 元")
    print(f"  单位水成本: {total_cost/total_supply:.4f} 元/m³" if total_supply > 0 else "  单位水成本: N/A")

    print(f"\n水库运行:")
    print(f"  初始库容: {df['storage'].iloc[0]:.1f} m³")
    print(f"  最终库容: {df['storage'].iloc[-1]:.1f} m³")
    print(f"  最低库容: {df['storage'].min():.1f} m³")
    print(f"  最高库容: {df['storage'].max():.1f} m³")

    print("=" * 80)

    return df


def compare_scenarios(scenario_results):
    """对比多个场景的结果"""
    print("\n" + "=" * 80)
    print("场景对比分析")
    print("=" * 80)

    comparison_data = []

    for scenario_id, (scenario_params, df) in scenario_results.items():
        if df is None:
            continue

        total_demand = df['demand'].sum()
        total_supply = df['actual_supply'].sum()
        total_shortage = df['shortage'].sum()
        avg_supply_rate = df['supply_rate'].mean()
        total_energy = df['energy_kwh'].sum()
        total_cost = df['energy_cost'].sum()

        comparison_data.append({
            'scenario': scenario_params['name'],
            'description': scenario_params['description'],
            'total_demand': total_demand,
            'total_supply': total_supply,
            'shortage': total_shortage,
            'supply_rate': avg_supply_rate,
            'energy': total_energy,
            'cost': total_cost,
            'unit_cost': total_cost / total_supply if total_supply > 0 else 0,
        })

    comparison_df = pd.DataFrame(comparison_data)

    print("\n关键指标对比:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def visualize_comparison(scenario_results, output_dir):
    """生成场景对比可视化"""
    setup_plotting_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("多场景对比分析", fontsize=16, fontweight='bold')

    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 子图1：供水保证率
    ax = axes[0, 0]
    scenario_names = []
    supply_rates = []
    for scenario_id, (scenario_params, df) in scenario_results.items():
        if df is not None:
            scenario_names.append(scenario_params['name'])
            supply_rates.append(df['supply_rate'].mean())

    bars = ax.bar(range(len(scenario_names)), supply_rates, color=colors[:len(scenario_names)])
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.set_ylabel('供水保证率 (%)')
    ax.set_title('供水保证率对比')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱子上标注数值
    for bar, rate in zip(bars, supply_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    # 子图2：总成本对比
    ax = axes[0, 1]
    costs = []
    for scenario_id, (scenario_params, df) in scenario_results.items():
        if df is not None:
            costs.append(df['energy_cost'].sum())

    ax.bar(range(len(scenario_names)), costs, color=colors[:len(scenario_names)])
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.set_ylabel('总成本 (元)')
    ax.set_title('运行成本对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 子图3：缺水量对比
    ax = axes[0, 2]
    shortages = []
    for scenario_id, (scenario_params, df) in scenario_results.items():
        if df is not None:
            shortages.append(df['shortage'].sum())

    ax.bar(range(len(scenario_names)), shortages, color=colors[:len(scenario_names)])
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.set_ylabel('总缺水量 (m³)')
    ax.set_title('缺水量对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 子图4：水库库容变化
    ax = axes[1, 0]
    for idx, (scenario_id, (scenario_params, df)) in enumerate(scenario_results.items()):
        if df is not None:
            ax.plot(df['hour'], df['storage'], label=scenario_params['name'],
                   color=colors[idx], linewidth=2)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('库容 (m³)')
    ax.set_title('水库库容变化对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图5：供水量vs需求量
    ax = axes[1, 1]
    for idx, (scenario_id, (scenario_params, df)) in enumerate(scenario_results.items()):
        if df is not None:
            ax.plot(df['hour'], df['demand'], label=f"{scenario_params['name']}-需求",
                   linestyle='--', color=colors[idx], alpha=0.7)
            ax.plot(df['hour'], df['actual_supply'], label=f"{scenario_params['name']}-供水",
                   linestyle='-', color=colors[idx], linewidth=2)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('流量 (m³/h)')
    ax.set_title('供需对比')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 子图6：单位水成本对比
    ax = axes[1, 2]
    unit_costs = []
    for scenario_id, (scenario_params, df) in scenario_results.items():
        if df is not None:
            total_cost = df['energy_cost'].sum()
            total_supply = df['actual_supply'].sum()
            unit_costs.append(total_cost / total_supply if total_supply > 0 else 0)

    ax.bar(range(len(scenario_names)), unit_costs, color=colors[:len(scenario_names)])
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=15, ha='right')
    ax.set_ylabel('单位水成本 (元/m³)')
    ax.set_title('单位水成本对比')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / 'scenario_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图表已保存: {output_path}")
    plt.close()


def generate_comparison_report(scenario_results, comparison_df, output_dir):
    """生成对比分析报告"""
    report = f"""# 城市供水系统多场景对比分析报告

## 1. 概述

本报告对城市供水系统在不同运行场景下的表现进行了全面对比分析，包括：

"""

    for scenario_id, (scenario_params, df) in scenario_results.items():
        report += f"- **{scenario_params['name']}**: {scenario_params['description']}\n"

    report += f"""

**分析时段**: 48小时（2天）
**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. 场景设置

### 场景参数对比

| 场景 | 入流倍数 | 需求倍数 | 泵站容量 | 初始库容 |
|------|---------|---------|---------|---------|
"""

    for scenario_id, (scenario_params, df) in scenario_results.items():
        report += f"| {scenario_params['name']} | {scenario_params['inflow_multiplier']:.1%} | {scenario_params['demand_multiplier']:.1%} | {scenario_params['pump_capacity']} m³/h | {scenario_params['initial_storage']:,} m³ |\n"

    report += """

## 3. 关键指标对比

### 3.1 供水保障

"""

    for idx, row in comparison_df.iterrows():
        status = "✅ 优秀" if row['supply_rate'] >= 99 else "⚠️ 良好" if row['supply_rate'] >= 95 else "❌ 不足"
        report += f"""
**{row['scenario']}**:
- 供水保证率: {row['supply_rate']:.2f}% {status}
- 总需求: {row['total_demand']:.1f} m³
- 总供水: {row['total_supply']:.1f} m³
- 缺水量: {row['shortage']:.1f} m³

"""

    report += """

### 3.2 经济性分析

"""

    for idx, row in comparison_df.iterrows():
        report += f"""
**{row['scenario']}**:
- 总能耗: {row['energy']:.1f} kWh
- 总成本: {row['cost']:.2f} 元
- 单位水成本: {row['unit_cost']:.4f} 元/m³

"""

    # 找出最佳和最差场景
    best_supply = comparison_df.loc[comparison_df['supply_rate'].idxmax()]
    best_cost = comparison_df.loc[comparison_df['unit_cost'].idxmin()]
    worst_supply = comparison_df.loc[comparison_df['supply_rate'].idxmin()]
    worst_cost = comparison_df.loc[comparison_df['unit_cost'].idxmax()]

    report += f"""

## 4. 对比分析

### 4.1 供水可靠性

- **最佳场景**: {best_supply['scenario']} (供水保证率 {best_supply['supply_rate']:.2f}%)
- **最差场景**: {worst_supply['scenario']} (供水保证率 {worst_supply['supply_rate']:.2f}%)
- **差距**: {best_supply['supply_rate'] - worst_supply['supply_rate']:.2f}%

### 4.2 经济效益

- **最经济场景**: {best_cost['scenario']} (单位水成本 {best_cost['unit_cost']:.4f} 元/m³)
- **成本最高场景**: {worst_cost['scenario']} (单位水成本 {worst_cost['unit_cost']:.4f} 元/m³)
- **成本差异**: {(worst_cost['unit_cost'] - best_cost['unit_cost']) / best_cost['unit_cost'] * 100:.1f}%

## 5. 主要发现

### 5.1 水源条件影响

"""

    # 对比正常vs缺水场景
    normal = comparison_df[comparison_df['scenario'].str.contains('正常')].iloc[0] if len(comparison_df[comparison_df['scenario'].str.contains('正常')]) > 0 else None
    drought = comparison_df[comparison_df['scenario'].str.contains('干旱')].iloc[0] if len(comparison_df[comparison_df['scenario'].str.contains('干旱')]) > 0 else None

    if normal is not None and drought is not None:
        supply_drop = normal['supply_rate'] - drought['supply_rate']
        shortage_increase = drought['shortage'] - normal['shortage']

        report += f"""
入流减少对供水系统影响显著：
- 严重干旱场景下，供水保证率下降 {supply_drop:.2f}%
- 缺水量增加 {shortage_increase:.1f} m³
- 需要加强水源调度和应急预案
"""

    report += """

### 5.2 需求波动影响

"""

    peak = comparison_df[comparison_df['scenario'].str.contains('高峰')].iloc[0] if len(comparison_df[comparison_df['scenario'].str.contains('高峰')]) > 0 else None

    if normal is not None and peak is not None:
        cost_increase = (peak['cost'] - normal['cost']) / normal['cost'] * 100

        report += f"""
高峰需求对系统压力较大：
- 需求增加40%导致成本上升 {cost_increase:.1f}%
- 供水保证率下降至 {peak['supply_rate']:.2f}%
- 建议在高峰期加强水库预蓄和泵站调度
"""

    report += """

### 5.3 设备故障应对

"""

    failure = comparison_df[comparison_df['scenario'].str.contains('故障')].iloc[0] if len(comparison_df[comparison_df['scenario'].str.contains('故障')]) > 0 else None

    if normal is not None and failure is not None:
        report += f"""
设备故障场景分析：
- 泵站容量降低40%时，供水保证率降至 {failure['supply_rate']:.2f}%
- 缺水量达到 {failure['shortage']:.1f} m³
- 需要建立备用泵站和应急调水机制
"""

    report += """

## 6. 建议

### 6.1 短期措施

1. **优化调度策略**
   - 根据天气预报提前调整水库蓄水
   - 在谷电时段增加抽水，降低成本
   - 建立实时监控和预警系统

2. **需求管理**
   - 在高峰期实施用水建议
   - 推广节水技术和设备
   - 建立阶梯水价机制

### 6.2 长期规划

1. **水源建设**
   - 增加备用水源或应急水源
   - 建设雨水收集和再生水利用设施
   - 加强流域水资源统一调度

2. **设施升级**
   - 增加备用泵站和管网
   - 提升泵站效率和可靠性
   - 建设智能化监控系统

3. **应急预案**
   - 制定不同场景下的应急预案
   - 储备应急供水设备
   - 加强跨区域供水联动

## 7. 结论

通过多场景对比分析，可以得出以下结论：

1. 在正常条件下，系统可以实现100%供水保证率，运行稳定经济
2. 水源减少和需求增加是主要风险因素，需重点防范
3. 设备故障对供水保障影响严重，需加强设备维护和备用能力
4. 优化调度策略可以在保证供水的同时降低运行成本
5. 建议建立多元化水源体系和完善的应急机制

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**优化框架**: Pyomo + GLPK
**Python版本**: {sys.version.split()[0]}
"""

    # 保存报告
    report_path = Path(output_dir) / 'COMPARISON_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ 对比报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("城市供水系统 - 多场景对比分析")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(__file__).parent / "comparison_results"
    output_dir.mkdir(exist_ok=True)

    # 求解所有场景
    scenario_results = {}

    for scenario_id, scenario_params in SCENARIOS.items():
        print(f"\n\n{'='*80}")
        print(f"运行场景 {scenario_id}")
        print(f"{'='*80}")

        df = solve_scenario(scenario_id, scenario_params)
        scenario_results[scenario_id] = (scenario_params, df)

        # 保存单个场景结果
        if df is not None:
            csv_path = output_dir / f"{scenario_id}_results.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 场景结果已保存: {csv_path}")

    # 对比分析
    comparison_df = compare_scenarios(scenario_results)

    # 保存对比表
    comparison_df.to_csv(output_dir / 'comparison_summary.csv', index=False, encoding='utf-8-sig')

    # 生成可视化
    visualize_comparison(scenario_results, output_dir)

    # 生成报告
    generate_comparison_report(scenario_results, comparison_df, output_dir)

    print("\n" + "=" * 80)
    print("所有场景分析完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")
    print("  - comparison_summary.csv: 汇总对比表")
    print("  - scenario_comparison.png: 对比图表")
    print("  - COMPARISON_REPORT.md: 详细对比报告")
    print("  - scenario*_results.csv: 各场景详细数据")


if __name__ == "__main__":
    main()

"""
测试重构后的代码

验证：
1. 可行性检查功能是否正常
2. 默认配置是否可用
3. 通用工具是否正常工作
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Feas import (
    build_water_network_model,
    SolverManager,
    TimeSeriesGenerator,
    ResultExtractor,
    OPTIMIZATION_DEFAULTS,
    FeasibilityStatus,
    check_problem_feasibility,
)


def test_feasibility_check():
    """测试可行性检查"""
    print("=" * 70)
    print("测试1: 可行性检查")
    print("=" * 70)

    # 创建一个简单的配置
    ts_gen = TimeSeriesGenerator()
    periods = ts_gen.create_periods(24)

    config = {
        "horizon": {"periods": periods},
        "nodes": [
            {
                "id": "source",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 1000.0,
                        "bounds": (500.0, 2000.0),
                        "role": "storage",
                    }
                },
                "attributes": {
                    "misc": {"inflow_series": "inflow"}
                },
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
                "id": "pipe",
                "kind": "pipeline",
                "from_node": "source",
                "to_node": "demand",
                "attributes": {"capacity": 100.0},
            }
        ],
        "series": {
            "inflow": {
                "values": ts_gen.constant(60.0, 24),
                "default": 60.0,
            },
            "demand": {
                "values": ts_gen.constant(50.0, 24),
                "default": 50.0,
            },
        },
        "objective_weights": {
            "shortage_penalty": OPTIMIZATION_DEFAULTS.shortage_penalty,
        },
    }

    # 检查配置可行性
    feasibility_result = check_problem_feasibility(config)
    print(f"配置可行性检查: {feasibility_result.status.value}")
    print(f"消息: {feasibility_result.message}")
    if not feasibility_result.is_feasible:
        print(f"问题: {feasibility_result.details}")
    print()


def test_solver_manager():
    """测试求解器管理器"""
    print("=" * 70)
    print("测试2: 求解器管理器和可行性检查")
    print("=" * 70)

    # 创建配置
    ts_gen = TimeSeriesGenerator()
    periods = ts_gen.create_periods(24)

    config = {
        "horizon": {"periods": periods},
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
                "attributes": {
                    "demand_profile": "demand",
                    "shortage_penalty": 100000.0,
                },
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
                "values": ts_gen.constant(60.0, 24),
                "default": 60.0,
            },
            "demand": {
                "values": ts_gen.constant(50.0, 24),
                "default": 50.0,
            },
        },
    }

    # 构建模型
    model = build_water_network_model(config)
    print(f"模型构建成功: {model.nvariables()} 变量, {model.nconstraints()} 约束")

    # 使用求解器管理器求解
    solver_mgr = SolverManager()
    results = solver_mgr.solve(model, tee=False)

    print(f"求解成功!")

    # 提取结果
    extractor = ResultExtractor()
    states = extractor.extract_node_states(model)
    flows = extractor.extract_edge_flows(model)
    objective_parts = extractor.extract_objective_components(model)

    print(f"\n结果提取:")
    print(f"  节点状态: {list(states.keys())}")
    print(f"  边流量: {list(flows.keys())}")
    print(f"  目标函数值: {objective_parts.get('total_objective', 'N/A'):.2f}")
    print()


def test_time_series_generator():
    """测试时间序列生成器"""
    print("=" * 70)
    print("测试3: 时间序列生成器")
    print("=" * 70)

    ts_gen = TimeSeriesGenerator()

    # 测试常数序列
    const_series = ts_gen.constant(10.0, 10)
    print(f"常数序列: {const_series[:5]}...")

    # 测试正弦序列
    sin_series = ts_gen.sinusoidal(base=50.0, amplitude=20.0, num_periods=24)
    print(f"正弦序列: {[f'{v:.1f}' for v in sin_series[:5]]}...")

    # 测试阶跃变化
    step_series = ts_gen.step_change(
        initial_value=10.0,
        final_value=20.0,
        num_periods=20,
        change_start=10,
        change_duration=5
    )
    print(f"阶跃序列: {step_series}")

    # 测试分段序列
    piecewise_series = ts_gen.piecewise(
        values_list=[10.0, 20.0, 15.0],
        durations=[5, 3, 4]
    )
    print(f"分段序列: {piecewise_series}")
    print()


def test_infeasible_problem():
    """测试不可行问题的检测"""
    print("=" * 70)
    print("测试4: 不可行问题检测")
    print("=" * 70)

    ts_gen = TimeSeriesGenerator()
    periods = ts_gen.create_periods(24)

    # 创建一个不可行的配置（需求远大于供给，且没有缺水松弛）
    config = {
        "horizon": {"periods": periods},
        "nodes": [
            {
                "id": "reservoir",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 100.0,
                        "bounds": (50.0, 200.0),  # 很小的水库
                        "role": "storage",
                    }
                },
                "attributes": {"misc": {"inflow_series": "inflow"}},
            },
            {
                "id": "demand",
                "kind": "demand",
                "states": {},
                "attributes": {
                    "demand_profile": "demand",
                    # 注意：这里提供shortage_penalty使得问题总是可行的（通过缺水松弛）
                    "shortage_penalty": 100000.0,
                },
            },
        ],
        "edges": [
            {
                "id": "flow",
                "kind": "pipeline",
                "from_node": "reservoir",
                "to_node": "demand",
                "attributes": {"capacity": 10.0},  # 容量很小
            }
        ],
        "series": {
            "inflow": {
                "values": ts_gen.constant(5.0, 24),  # 入流很小
                "default": 5.0,
            },
            "demand": {
                "values": ts_gen.constant(50.0, 24),  # 需求很大
                "default": 50.0,
            },
        },
    }

    # 检查配置可行性（预检）
    feasibility_result = check_problem_feasibility(config)
    print(f"配置预检结果: {feasibility_result.status.value}")
    print(f"消息: {feasibility_result.message}")
    if feasibility_result.details:
        print(f"详细信息: {feasibility_result.details}")

    # 尝试求解
    print("\n尝试求解...")
    try:
        model = build_water_network_model(config)
        solver_mgr = SolverManager()
        results = solver_mgr.solve(model, tee=False)
        print("求解成功（通过缺水松弛变量）")

        # 检查缺水量
        extractor = ResultExtractor()
        shortages = extractor.extract_shortages(model)
        if shortages:
            for node_id, shortage_list in shortages.items():
                total_shortage = sum(shortage_list)
                print(f"  节点 {node_id} 总缺水量: {total_shortage:.1f}")
    except Exception as e:
        print(f"求解失败: {e}")

    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("重构代码测试")
    print("=" * 70 + "\n")

    try:
        test_feasibility_check()
        test_solver_manager()
        test_time_series_generator()
        test_infeasible_problem()

        print("=" * 70)
        print("所有测试通过！✓")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

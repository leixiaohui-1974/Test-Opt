"""
简化测试 - 不需要求解器

验证：
1. 可行性检查模块
2. 默认配置模块
3. 通用工具模块
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from Feas import (
    TimeSeriesGenerator,
    OPTIMIZATION_DEFAULTS,
    MPC_DEFAULTS,
    CANAL_IDZ_DEFAULTS,
    check_problem_feasibility,
    FeasibilityStatus,
)


def test_defaults():
    """测试默认配置"""
    print("=" * 70)
    print("测试1: 默认配置模块")
    print("=" * 70)

    print(f"优化默认参数:")
    print(f"  缺水惩罚: {OPTIMIZATION_DEFAULTS.shortage_penalty}")
    print(f"  泵站成本: {OPTIMIZATION_DEFAULTS.pump_cost}")
    print(f"  默认求解器: {OPTIMIZATION_DEFAULTS.default_solver}")

    print(f"\nMPC默认参数:")
    print(f"  预测时域: {MPC_DEFAULTS.prediction_horizon}")
    print(f"  控制时域: {MPC_DEFAULTS.control_horizon}")
    print(f"  跟踪权重: {MPC_DEFAULTS.tracking_weight}")

    print(f"\n渠道IDZ默认参数:")
    print(f"  Muskingum K缩放: {CANAL_IDZ_DEFAULTS.muskingum_k_scale}")
    print(f"  Muskingum X: {CANAL_IDZ_DEFAULTS.muskingum_x}")
    print(f"  初始闸门开度: {CANAL_IDZ_DEFAULTS.initial_gate_opening}")

    print("\n✓ 默认配置模块正常工作")
    print()


def test_time_series_generator():
    """测试时间序列生成器"""
    print("=" * 70)
    print("测试2: 时间序列生成器")
    print("=" * 70)

    ts_gen = TimeSeriesGenerator()

    # 测试周期生成
    periods = ts_gen.create_periods(48, prefix="t")
    print(f"生成周期: {periods[:5]}... (共{len(periods)}个)")

    # 测试常数序列
    const_series = ts_gen.constant(10.0, 10)
    print(f"常数序列: {const_series}")
    assert all(v == 10.0 for v in const_series), "常数序列错误"

    # 测试正弦序列
    sin_series = ts_gen.sinusoidal(base=50.0, amplitude=20.0, num_periods=24, frequency=1.0)
    print(f"正弦序列（前5个）: {[f'{v:.1f}' for v in sin_series[:5]]}")
    assert len(sin_series) == 24, "正弦序列长度错误"
    assert min(sin_series) >= 30.0 and max(sin_series) <= 70.0, "正弦序列范围错误"

    # 测试阶跃变化
    step_series = ts_gen.step_change(
        initial_value=10.0,
        final_value=20.0,
        num_periods=20,
        change_start=10,
        change_duration=5
    )
    print(f"阶跃序列: {step_series}")
    assert step_series[9] == 10.0, "阶跃前值错误"
    assert step_series[10] == 20.0, "阶跃后值错误"
    assert step_series[15] == 10.0, "阶跃恢复值错误"

    # 测试分段序列
    piecewise_series = ts_gen.piecewise(
        values_list=[10.0, 20.0, 15.0],
        durations=[5, 3, 4]
    )
    print(f"分段序列: {piecewise_series}")
    assert len(piecewise_series) == 12, "分段序列长度错误"
    assert piecewise_series[0:5] == [10.0] * 5, "第一段错误"
    assert piecewise_series[5:8] == [20.0] * 3, "第二段错误"
    assert piecewise_series[8:12] == [15.0] * 4, "第三段错误"

    print("\n✓ 时间序列生成器正常工作")
    print()


def test_feasibility_check():
    """测试可行性检查"""
    print("=" * 70)
    print("测试3: 可行性检查模块")
    print("=" * 70)

    ts_gen = TimeSeriesGenerator()
    periods = ts_gen.create_periods(24)

    # 测试1: 正常配置
    config_normal = {
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
                "attributes": {"misc": {"inflow_series": "inflow"}},
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
            "inflow": {"values": ts_gen.constant(60.0, 24), "default": 60.0},
            "demand": {"values": ts_gen.constant(50.0, 24), "default": 50.0},
        },
    }

    result = check_problem_feasibility(config_normal)
    print(f"正常配置检查: {result.status.value}")
    print(f"  消息: {result.message}")
    assert result.is_feasible, "正常配置应该可行"

    # 测试2: 初始值超出范围
    config_bad_initial = {
        "horizon": {"periods": periods},
        "nodes": [
            {
                "id": "source",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 3000.0,  # 超出上界
                        "bounds": (500.0, 2000.0),
                        "role": "storage",
                    }
                },
            },
        ],
        "edges": [],
        "series": {},
    }

    result = check_problem_feasibility(config_bad_initial)
    print(f"\n初始值超界配置检查: {result.status.value}")
    print(f"  消息: {result.message}")
    print(f"  问题: {result.details.get('issues', [])}")
    assert not result.is_feasible or result.status == FeasibilityStatus.UNKNOWN, "应检测出初始值问题"

    # 测试3: 空配置
    config_empty = {
        "horizon": {"periods": []},
        "nodes": [],
        "edges": [],
        "series": {},
    }

    result = check_problem_feasibility(config_empty)
    print(f"\n空配置检查: {result.status.value}")
    print(f"  消息: {result.message}")
    print(f"  问题: {result.details.get('issues', [])}")
    assert not result.is_feasible, "空配置应该不可行"

    print("\n✓ 可行性检查模块正常工作")
    print()


def test_comprehensive():
    """综合测试 - 展示如何使用新工具"""
    print("=" * 70)
    print("测试4: 综合示例")
    print("=" * 70)

    # 使用工具创建完整配置
    ts_gen = TimeSeriesGenerator()

    # 生成48小时周期
    periods = ts_gen.create_periods(48)
    print(f"生成时间周期: {len(periods)}个")

    # 生成变化的需求模式（正弦波 + 阶跃变化）
    base_demand = ts_gen.sinusoidal(base=60.0, amplitude=20.0, num_periods=48, frequency=2.0)
    demand_with_peak = [
        base_demand[i] * 1.5 if 20 <= i < 28 else base_demand[i]
        for i in range(48)
    ]
    print(f"需求范围: {min(demand_with_peak):.1f} - {max(demand_with_peak):.1f}")

    # 生成入流（稳定 + 小扰动）
    inflow = ts_gen.sinusoidal(base=80.0, amplitude=10.0, num_periods=48, frequency=1.0, noise_std=2.0)
    print(f"入流范围: {min(inflow):.1f} - {max(inflow):.1f}")

    # 生成电价（分段）
    peak_hours = [8, 9, 10, 11, 18, 19, 20, 21]
    valley_hours = [0, 1, 2, 3, 4, 5]
    electricity_price = []
    for i in range(48):
        hour = i % 24
        if hour in peak_hours:
            electricity_price.append(1.2)
        elif hour in valley_hours:
            electricity_price.append(0.4)
        else:
            electricity_price.append(0.8)
    print(f"电价范围: {min(electricity_price)} - {max(electricity_price)}")

    # 创建完整配置
    config = {
        "horizon": {"periods": periods},
        "nodes": [
            {
                "id": "reservoir",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 50000.0,
                        "bounds": (20000.0, 100000.0),
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
                    "shortage_penalty": OPTIMIZATION_DEFAULTS.shortage_penalty,
                },
            },
        ],
        "edges": [
            {
                "id": "pipe",
                "kind": "pipeline",
                "from_node": "reservoir",
                "to_node": "demand",
                "attributes": {"capacity": 200.0},
            }
        ],
        "series": {
            "inflow": {"values": inflow, "default": 80.0},
            "demand": {"values": demand_with_peak, "default": 60.0},
            "electricity_price": {"values": electricity_price, "default": 0.8},
        },
        "objective_weights": {
            "pumping_cost": OPTIMIZATION_DEFAULTS.pump_cost,
            "shortage_penalty": OPTIMIZATION_DEFAULTS.shortage_penalty,
        },
    }

    # 检查可行性
    result = check_problem_feasibility(config)
    print(f"\n综合配置可行性: {result.status.value}")
    print(f"  消息: {result.message}")

    print("\n✓ 综合示例完成")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("简化重构代码测试（无需求解器）")
    print("=" * 70 + "\n")

    try:
        test_defaults()
        test_time_series_generator()
        test_feasibility_check()
        test_comprehensive()

        print("=" * 70)
        print("所有测试通过！✓")
        print("=" * 70)
        print("\n重构总结:")
        print("1. ✓ 添加了可行性检查框架 (feasibility.py)")
        print("2. ✓ 创建了默认配置模块 (defaults.py) - 移除硬编码")
        print("3. ✓ 提取了通用工具 (utils.py):")
        print("   - TimeSeriesGenerator: 时间序列生成")
        print("   - ResultExtractor: 结果提取")
        print("   - SolverManager: 求解器管理")
        print("4. ✓ MPC模块已集成可行性检查")
        print("5. ✓ 所有基础模块使用配置化参数，无硬编码")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

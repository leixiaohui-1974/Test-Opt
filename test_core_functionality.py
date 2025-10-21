"""
独立测试脚本：验证水网优化模型核心功能
"""
import sys
from pathlib import Path

# 添加Feas目录到路径
sys.path.insert(0, str(Path(__file__).parent / "Feas"))

from water_network_generic import build_water_network_model
from water_network_schema import NetworkConfig

def create_simple_network_config() -> NetworkConfig:
    """创建一个简单的测试网络配置"""
    config = {
        "metadata": {
            "name": "Simple Test Network",
            "version": "1.0",
        },
        "horizon": {
            "periods": ["t0", "t1", "t2", "t3"],
        },
        "nodes": [
            {
                "id": "reservoir",
                "name": "上游水库",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "initial": 100.0,
                        "bounds": (50.0, 200.0),
                        "role": "storage",
                    }
                },
                "attributes": {},
            },
            {
                "id": "demand_node",
                "name": "需求节点",
                "kind": "demand",
                "states": {},
                "attributes": {
                    "demand_profile": "demand_series",
                },
            },
        ],
        "edges": [
            {
                "id": "flow_edge",
                "name": "流量边",
                "kind": "pipeline",
                "from_node": "reservoir",
                "to_node": "demand_node",
                "attributes": {
                    "capacity": 50.0,
                    "energy_cost": 1.0,
                },
            }
        ],
        "series": {
            "inflow_series": {
                "times": ["t0", "t1", "t2", "t3"],
                "values": [20.0, 25.0, 30.0, 20.0],
                "default": 20.0,
                "units": "m3/s",
            },
            "demand_series": {
                "times": ["t0", "t1", "t2", "t3"],
                "values": [15.0, 20.0, 25.0, 15.0],
                "default": 15.0,
                "units": "m3/s",
            },
        },
        "objective_weights": {
            "pumping_cost": 200.0,
            "shortage_penalty": 100000.0,
        },
    }

    # 添加入流序列到水库
    config["nodes"][0]["attributes"] = {
        "misc": {"inflow_series": "inflow_series"}
    }

    return config


def test_model_building():
    """测试1: 模型构建"""
    print("\n" + "="*60)
    print("测试1: 构建Pyomo优化模型")
    print("="*60)

    config = create_simple_network_config()
    model = build_water_network_model(config)

    print(f"✓ 模型名称: {model.name}")
    print(f"✓ 时间步数: {len(list(model.T))}")
    print(f"✓ 节点数: {len(list(model.N))}")
    print(f"✓ 边数: {len(list(model.E))}")
    print(f"✓ 约束数: {model.nconstraints()}")
    print(f"✓ 变量数: {model.nvariables()}")

    assert model is not None, "模型构建失败"
    assert model.nconstraints() > 0, "模型约束数为0"
    assert model.nvariables() > 0, "模型变量数为0"

    print("\n✅ 测试1通过：模型构建成功")
    return model


def test_model_structure(model):
    """测试2: 模型结构"""
    print("\n" + "="*60)
    print("测试2: 验证模型结构")
    print("="*60)

    # 检查集合
    assert hasattr(model, "T"), "缺少时间集合 T"
    assert hasattr(model, "N"), "缺少节点集合 N"
    assert hasattr(model, "E"), "缺少边集合 E"
    print("✓ 集合定义正确")

    # 检查变量
    assert hasattr(model, "flow"), "缺少流量变量 flow"
    assert hasattr(model, "state"), "缺少状态变量 state"
    print("✓ 变量定义正确")

    # 检查约束
    assert hasattr(model, "mass_balance"), "缺少质量守恒约束"
    print("✓ 约束定义正确")

    # 检查目标函数
    assert hasattr(model, "obj"), "缺少目标函数"
    print("✓ 目标函数定义正确")

    print("\n✅ 测试2通过：模型结构完整")


def test_complex_network():
    """测试3: 复杂网络配置（带分段效率曲线）"""
    print("\n" + "="*60)
    print("测试3: 复杂网络配置（泵站+分段效率）")
    print("="*60)

    config = {
        "horizon": {"periods": ["t0", "t1", "t2"]},
        "nodes": [
            {
                "id": "source",
                "kind": "reservoir",
                "states": {"storage": {"initial": 100.0, "role": "storage"}},
                "attributes": {"misc": {"inflow_series": "inflow"}},
            },
            {
                "id": "pump_station",
                "kind": "pump_station",
                "states": {},
                "attributes": {},
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
                "id": "gravity_flow",
                "kind": "gravity",
                "from_node": "source",
                "to_node": "pump_station",
                "attributes": {"capacity": 100.0},
            },
            {
                "id": "pump_edge",
                "kind": "pump",
                "from_node": "pump_station",
                "to_node": "demand",
                "attributes": {
                    "efficiency_curve": (
                        [0.0, 10.0, 20.0, 30.0],  # 流量断点
                        [0.0, 50.0, 80.0, 120.0],  # 能耗成本
                    ),
                    "energy_cost": 2.0,
                },
            },
        ],
        "series": {
            "inflow": {"values": [20.0, 25.0, 30.0], "default": 20.0},
            "demand": {"values": [15.0, 20.0, 25.0], "default": 15.0},
        },
        "objective_weights": {
            "pumping_cost": 100.0,
            "shortage_penalty": 1e5,
        },
    }

    model = build_water_network_model(config)

    print(f"✓ 模型构建成功")
    print(f"✓ 约束数: {model.nconstraints()}")
    print(f"✓ 变量数: {model.nvariables()}")

    # 检查分段效率相关组件
    if hasattr(model, "pw_edges"):
        print(f"✓ 分段效率边数: {len(list(model.pw_edges))}")

    if hasattr(model, "segment_flow"):
        print(f"✓ 分段流量变量已创建")

    print("\n✅ 测试3通过：复杂网络配置正常")
    return model


def test_solver_availability():
    """测试4: 求解器可用性"""
    print("\n" + "="*60)
    print("测试4: 检查优化求解器")
    print("="*60)

    try:
        from pyomo.environ import SolverFactory

        # 测试GLPK
        try:
            glpk = SolverFactory('glpk')
            glpk_available = glpk.available(exception_flag=False)
            print(f"GLPK求解器: {'✓ 可用' if glpk_available else '✗ 不可用'}")
        except Exception as e:
            print(f"GLPK求解器: ✗ 错误 ({e})")
            glpk_available = False

        # 测试HiGHS
        try:
            highs = SolverFactory('appsi_highs')
            highs_available = highs.available(exception_flag=False)
            print(f"HiGHS求解器: {'✓ 可用' if highs_available else '✗ 不可用'}")
        except Exception as e:
            print(f"HiGHS求解器: ✗ 错误 ({e})")
            highs_available = False

        if glpk_available or highs_available:
            print("\n✅ 测试4通过：至少一个求解器可用")
            return True
        else:
            print("\n⚠️  测试4警告：没有可用的求解器")
            return False

    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("水网优化模型核心功能测试")
    print("="*60)

    try:
        # 测试1: 基础模型构建
        model1 = test_model_building()

        # 测试2: 模型结构验证
        test_model_structure(model1)

        # 测试3: 复杂网络
        model2 = test_complex_network()

        # 测试4: 求解器
        solver_available = test_solver_availability()

        print("\n" + "="*60)
        print("所有测试完成")
        print("="*60)
        print("✅ 核心功能正常")

        if not solver_available:
            print("⚠️  建议安装GLPK或HiGHS求解器以运行优化")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

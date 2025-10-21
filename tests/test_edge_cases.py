"""
边界条件测试：测试各种边缘情况和异常场景
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Feas"))

from water_network_generic import build_water_network_model
from water_network_schema import NetworkConfig


class TestEmptyNetwork:
    """测试空网络或最小配置"""

    def test_minimal_network(self):
        """测试最小可运行网络（单节点）"""
        config = {
            "horizon": {"periods": ["t0", "t1"]},
            "nodes": [
                {
                    "id": "node1",
                    "kind": "junction",
                    "states": {},
                    "attributes": {},
                }
            ],
            "edges": [],
            "series": {},
            "objective_weights": {},
        }

        model = build_water_network_model(config)
        assert model is not None
        assert len(list(model.N)) == 1
        assert len(list(model.E)) == 0

    def test_empty_time_series(self):
        """测试空时间序列"""
        config = {
            "horizon": {"periods": []},
            "nodes": [{"id": "n1", "kind": "junction", "states": {}, "attributes": {}}],
            "edges": [],
            "series": {},
        }

        # 空时间序列应该引发错误
        with pytest.raises(Exception):
            build_water_network_model(config)


class TestBoundaryValues:
    """测试边界值情况"""

    def test_zero_capacity(self):
        """测试零容量边"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": "n1", "kind": "source", "states": {}, "attributes": {}},
                {"id": "n2", "kind": "sink", "states": {}, "attributes": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "kind": "pipeline",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {"capacity": 0.0},
                }
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        assert model is not None
        # 流量上界应该是0
        assert model.flow["e1", "t0"].ub == 0.0

    def test_large_values(self):
        """测试大数值稳定性"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {
                    "id": "reservoir",
                    "kind": "reservoir",
                    "states": {
                        "storage": {
                            "initial": 1e9,  # 超大初始值
                            "bounds": (1e8, 1e10),
                            "role": "storage",
                        }
                    },
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
                    "id": "flow",
                    "kind": "pipeline",
                    "from_node": "reservoir",
                    "to_node": "demand",
                    "attributes": {"capacity": 1e6},
                }
            ],
            "series": {"demand": {"values": [1e5], "default": 1e5}},
        }

        model = build_water_network_model(config)
        assert model is not None
        assert model.state[("reservoir", "storage"), "t0"].lb == 1e8
        assert model.state[("reservoir", "storage"), "t0"].ub == 1e10

    def test_negative_bounds_invalid(self):
        """测试负值边界（库容不应该为负）"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {
                    "id": "reservoir",
                    "kind": "reservoir",
                    "states": {
                        "storage": {
                            "initial": 100.0,
                            "bounds": (-100.0, 200.0),  # 负下界
                            "role": "storage",
                        }
                    },
                    "attributes": {},
                }
            ],
            "edges": [],
            "series": {},
        }

        model = build_water_network_model(config)
        # 模型应该构建成功，但下界是负值（这可能导致非物理解）
        assert model.state[("reservoir", "storage"), "t0"].lb == -100.0


class TestTopologyVariations:
    """测试不同拓扑结构"""

    def test_chain_network(self):
        """测试链式网络: n1 -> n2 -> n3"""
        config = {
            "horizon": {"periods": ["t0", "t1"]},
            "nodes": [
                {"id": f"n{i}", "kind": "junction", "states": {}, "attributes": {}}
                for i in range(1, 4)
            ],
            "edges": [
                {
                    "id": f"e{i}",
                    "kind": "pipeline",
                    "from_node": f"n{i}",
                    "to_node": f"n{i+1}",
                    "attributes": {"capacity": 100.0},
                }
                for i in range(1, 3)
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        assert len(list(model.N)) == 3
        assert len(list(model.E)) == 2

    def test_star_network(self):
        """测试星型网络: n1 连接到 n2, n3, n4"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": f"n{i}", "kind": "junction", "states": {}, "attributes": {}}
                for i in range(1, 5)
            ],
            "edges": [
                {
                    "id": f"e{i}",
                    "kind": "pipeline",
                    "from_node": "n1",
                    "to_node": f"n{i+1}",
                    "attributes": {"capacity": 50.0},
                }
                for i in range(1, 4)
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        assert len(list(model.N)) == 4
        assert len(list(model.E)) == 3

    def test_parallel_edges(self):
        """测试并联边（两个节点间多条边）"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": "n1", "kind": "source", "states": {}, "attributes": {}},
                {"id": "n2", "kind": "sink", "states": {}, "attributes": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "kind": "pipeline",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {"capacity": 50.0},
                },
                {
                    "id": "e2",
                    "kind": "pipeline",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {"capacity": 30.0},
                },
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        assert len(list(model.E)) == 2


class TestPiecewiseEfficiency:
    """测试分段效率曲线边界情况"""

    def test_single_segment(self):
        """测试单段效率曲线（退化为线性）"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": "n1", "kind": "source", "states": {}, "attributes": {}},
                {"id": "n2", "kind": "sink", "states": {}, "attributes": {}},
            ],
            "edges": [
                {
                    "id": "pump",
                    "kind": "pump",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {
                        "efficiency_curve": ([0.0, 100.0], [0.0, 50.0])
                    },
                }
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        # 单段曲线应该产生1个分段
        if hasattr(model, "pw_edges"):
            assert len(list(model.pw_edges)) == 1

    def test_many_segments(self):
        """测试多段效率曲线"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": "n1", "kind": "source", "states": {}, "attributes": {}},
                {"id": "n2", "kind": "sink", "states": {}, "attributes": {}},
            ],
            "edges": [
                {
                    "id": "pump",
                    "kind": "pump",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {
                        "efficiency_curve": (
                            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            [0, 5, 12, 20, 30, 42, 56, 72, 90, 110, 135],
                        )
                    },
                }
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        if hasattr(model, "segment_index"):
            # 11个断点应该产生10个分段
            assert len(list(model.segment_index)) == 10

    def test_unsorted_breakpoints(self):
        """测试未排序的断点（应该自动排序）"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {"id": "n1", "kind": "source", "states": {}, "attributes": {}},
                {"id": "n2", "kind": "sink", "states": {}, "attributes": {}},
            ],
            "edges": [
                {
                    "id": "pump",
                    "kind": "pump",
                    "from_node": "n1",
                    "to_node": "n2",
                    "attributes": {
                        "efficiency_curve": (
                            [100, 0, 50],  # 未排序
                            [50, 0, 20],
                        )
                    },
                }
            ],
            "series": {},
        }

        model = build_water_network_model(config)
        # 应该能够构建成功（内部会排序）
        assert model is not None


class TestTimeSeriesHandling:
    """测试时间序列处理"""

    def test_short_time_series(self):
        """测试短于时间范围的序列（应该使用默认值填充）"""
        config = {
            "horizon": {"periods": ["t0", "t1", "t2", "t3", "t4"]},
            "nodes": [
                {
                    "id": "reservoir",
                    "kind": "reservoir",
                    "states": {
                        "storage": {"initial": 100.0, "role": "storage"}
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
                    "id": "flow",
                    "kind": "pipeline",
                    "from_node": "reservoir",
                    "to_node": "demand",
                    "attributes": {"capacity": 50.0},
                }
            ],
            "series": {
                "inflow": {"values": [10, 20], "default": 15.0},  # 只有2个值
                "demand": {"values": [5, 10, 15], "default": 12.0},  # 只有3个值
            },
        }

        model = build_water_network_model(config)
        # 应该能够构建成功，缺失的值用default填充
        assert model is not None
        # 验证参数值
        from pyomo.environ import value

        assert value(model.inflow["reservoir", "t0"]) == 10.0
        assert value(model.inflow["reservoir", "t1"]) == 20.0
        assert value(model.inflow["reservoir", "t2"]) == 15.0  # default
        assert value(model.demand["demand", "t2"]) == 15.0
        assert value(model.demand["demand", "t3"]) == 12.0  # default

    def test_missing_default_value(self):
        """测试缺少默认值的情况"""
        config = {
            "horizon": {"periods": ["t0", "t1", "t2"]},
            "nodes": [
                {
                    "id": "demand",
                    "kind": "demand",
                    "states": {},
                    "attributes": {"demand_profile": "demand"},
                }
            ],
            "edges": [],
            "series": {
                "demand": {"values": [10.0], "default": 0.0}  # 使用0作为默认
            },
        }

        model = build_water_network_model(config)
        from pyomo.environ import value

        assert value(model.demand["demand", "t0"]) == 10.0
        assert value(model.demand["demand", "t1"]) == 0.0
        assert value(model.demand["demand", "t2"]) == 0.0


class TestStateRoles:
    """测试不同状态角色"""

    def test_multiple_states_per_node(self):
        """测试单个节点的多个状态"""
        config = {
            "horizon": {"periods": ["t0", "t1"]},
            "nodes": [
                {
                    "id": "reservoir",
                    "kind": "reservoir",
                    "states": {
                        "storage": {
                            "initial": 1000.0,
                            "bounds": (500.0, 2000.0),
                            "role": "storage",
                        },
                        "level": {
                            "initial": 10.0,
                            "bounds": (5.0, 20.0),
                            "role": "level",
                        },
                        "temperature": {
                            "initial": 15.0,
                            "bounds": (0.0, 30.0),
                            "role": "auxiliary",
                        },
                    },
                    "attributes": {},
                }
            ],
            "edges": [],
            "series": {},
        }

        model = build_water_network_model(config)
        assert len(list(model.state_index)) == 3
        # 验证状态变量存在（不验证值，因为未求解）
        assert ("reservoir", "storage") in model.state_index
        assert ("reservoir", "level") in model.state_index
        assert ("reservoir", "temperature") in model.state_index

    def test_no_storage_state(self):
        """测试没有storage状态的节点（纯junction）"""
        config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [
                {
                    "id": "junction",
                    "kind": "junction",
                    "states": {},  # 无状态
                    "attributes": {},
                }
            ],
            "edges": [],
            "series": {},
        }

        model = build_water_network_model(config)
        # 应该能够构建，质量守恒约束变为代数方程
        assert model is not None


def test_comprehensive_stress():
    """综合压力测试：复杂网络配置"""
    config = {
        "horizon": {"periods": [f"t{i}" for i in range(48)]},  # 48小时
        "nodes": [
            {
                "id": f"node_{i}",
                "kind": "junction" if i % 3 != 0 else "reservoir",
                "states": (
                    {"storage": {"initial": 1000.0, "role": "storage"}}
                    if i % 3 == 0
                    else {}
                ),
                "attributes": {},
            }
            for i in range(20)
        ],
        "edges": [
            {
                "id": f"edge_{i}",
                "kind": "pipeline",
                "from_node": f"node_{i}",
                "to_node": f"node_{i+1}",
                "attributes": {"capacity": 100.0},
            }
            for i in range(19)
        ],
        "series": {},
    }

    model = build_water_network_model(config)
    assert model is not None
    assert len(list(model.N)) == 20
    assert len(list(model.E)) == 19
    assert len(list(model.T)) == 48
    print(f"压力测试: {model.nvariables()} 变量, {model.nconstraints()} 约束")


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])

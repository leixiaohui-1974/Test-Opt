"""
测试配置验证和异常处理
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Feas.exceptions import (
    ConfigurationError,
    ValidationError,
    TopologyError,
    TimeSeriesError,
)
from Feas.validation import validate_network_config
from Feas.water_network_generic import build_water_network_model


class TestConfigurationValidation:
    """测试配置验证"""

    def test_missing_required_fields(self):
        """测试缺少必需字段"""
        # 缺少nodes
        with pytest.raises(ConfigurationError, match="缺少必需字段"):
            validate_network_config({"horizon": {}, "edges": []})

        # 缺少edges
        with pytest.raises(ConfigurationError, match="缺少必需字段"):
            validate_network_config({"horizon": {}, "nodes": []})

    def test_invalid_types(self):
        """测试无效类型"""
        # nodes不是列表
        with pytest.raises(ConfigurationError, match="必须是列表类型"):
            validate_network_config(
                {"horizon": {}, "nodes": "invalid", "edges": []}
            )

        # edges不是列表
        with pytest.raises(ConfigurationError, match="必须是列表类型"):
            validate_network_config(
                {"horizon": {}, "nodes": [], "edges": "invalid"}
            )

    def test_empty_time_periods(self):
        """测试空时间周期"""
        with pytest.raises(TimeSeriesError, match="不能为空"):
            validate_network_config(
                {"horizon": {"periods": []}, "nodes": [], "edges": []}
            )

    def test_duplicate_periods(self):
        """测试重复时间周期"""
        with pytest.raises(TimeSeriesError, match="包含重复值"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0", "t1", "t0"]},
                    "nodes": [],
                    "edges": [],
                }
            )


class TestNodeValidation:
    """测试节点验证"""

    def test_missing_node_id(self):
        """测试缺少节点ID"""
        with pytest.raises(ConfigurationError, match="缺少 'id' 字段"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"kind": "junction"}],
                    "edges": [],
                }
            )

    def test_duplicate_node_ids(self):
        """测试重复节点ID"""
        with pytest.raises(ValidationError, match="节点ID重复"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {"id": "n1", "kind": "junction"},
                        {"id": "n1", "kind": "source"},
                    ],
                    "edges": [],
                }
            )

    def test_invalid_node_kind(self):
        """测试无效节点类型"""
        with pytest.raises(ValidationError, match="类型.*无效"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"id": "n1", "kind": "invalid_type"}],
                    "edges": [],
                }
            )

    def test_invalid_state_role(self):
        """测试无效状态角色"""
        with pytest.raises(ValidationError, match="角色.*无效"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {
                            "id": "n1",
                            "kind": "reservoir",
                            "states": {
                                "storage": {"role": "invalid_role"}
                            },
                        }
                    ],
                    "edges": [],
                }
            )

    def test_invalid_bounds(self):
        """测试无效边界"""
        with pytest.raises(ValidationError, match="下界.*大于上界"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {
                            "id": "n1",
                            "kind": "reservoir",
                            "states": {
                                "storage": {
                                    "bounds": (100.0, 50.0),  # 下界>上界
                                    "role": "storage",
                                }
                            },
                        }
                    ],
                    "edges": [],
                }
            )


class TestEdgeValidation:
    """测试边验证"""

    def test_missing_edge_nodes(self):
        """测试缺少边节点"""
        with pytest.raises(ConfigurationError, match="缺少 'from_node'"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"id": "n1", "kind": "junction"}],
                    "edges": [{"id": "e1", "to_node": "n1"}],
                }
            )

    def test_duplicate_edge_ids(self):
        """测试重复边ID"""
        with pytest.raises(ValidationError, match="边ID重复"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {"id": "n1", "kind": "source"},
                        {"id": "n2", "kind": "sink"},
                    ],
                    "edges": [
                        {
                            "id": "e1",
                            "from_node": "n1",
                            "to_node": "n2",
                            "kind": "pipeline",
                        },
                        {
                            "id": "e1",
                            "from_node": "n1",
                            "to_node": "n2",
                            "kind": "pipeline",
                        },
                    ],
                }
            )

    def test_nonexistent_nodes(self):
        """测试不存在的节点"""
        with pytest.raises(TopologyError, match="不存在"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"id": "n1", "kind": "junction"}],
                    "edges": [
                        {
                            "id": "e1",
                            "from_node": "n1",
                            "to_node": "n2",  # n2不存在
                            "kind": "pipeline",
                        }
                    ],
                }
            )

    def test_negative_capacity(self):
        """测试负容量"""
        with pytest.raises(ValidationError, match="不能为负"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {"id": "n1", "kind": "source"},
                        {"id": "n2", "kind": "sink"},
                    ],
                    "edges": [
                        {
                            "id": "e1",
                            "from_node": "n1",
                            "to_node": "n2",
                            "kind": "pipeline",
                            "attributes": {"capacity": -100.0},
                        }
                    ],
                }
            )

    def test_invalid_efficiency_curve(self):
        """测试无效效率曲线"""
        # 断点数不足
        with pytest.raises(ValidationError, match="至少需要2个断点"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [
                        {"id": "n1", "kind": "source"},
                        {"id": "n2", "kind": "sink"},
                    ],
                    "edges": [
                        {
                            "id": "e1",
                            "from_node": "n1",
                            "to_node": "n2",
                            "kind": "pump",
                            "attributes": {
                                "efficiency_curve": ([0.0], [0.0])
                            },
                        }
                    ],
                }
            )


class TestTimeSeriesValidation:
    """测试时间序列验证"""

    def test_empty_values(self):
        """测试空值列表"""
        with pytest.raises(TimeSeriesError, match="不能为空"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"id": "n1", "kind": "junction", "states": {}, "attributes": {}}],
                    "edges": [],
                    "series": {"s1": {"values": []}},
                }
            )

    def test_values_exceed_times(self):
        """测试值超过时间长度"""
        with pytest.raises(TimeSeriesError, match="超过 times 长度"):
            validate_network_config(
                {
                    "horizon": {"periods": ["t0"]},
                    "nodes": [{"id": "n1", "kind": "junction", "states": {}, "attributes": {}}],
                    "edges": [],
                    "series": {
                        "s1": {
                            "times": ["t0", "t1"],
                            "values": [1, 2, 3, 4, 5],  # 超过times长度
                        }
                    },
                }
            )


class TestIntegrationWithModel:
    """测试与模型构建的集成"""

    def test_validation_enabled_by_default(self):
        """测试默认启用验证"""
        invalid_config = {
            "horizon": {"periods": []},  # 空周期
            "nodes": [],
            "edges": [],
        }

        with pytest.raises(TimeSeriesError):
            build_water_network_model(invalid_config)

    def test_validation_can_be_disabled(self):
        """测试可以禁用验证"""
        invalid_config = {
            "horizon": {"periods": ["t0"]},
            "nodes": [],
            "edges": [],
        }

        # 禁用验证应该能构建模型
        model = build_water_network_model(invalid_config, validate=False)
        assert model is not None

    def test_valid_config_passes(self):
        """测试有效配置通过验证"""
        valid_config = {
            "horizon": {"periods": ["t0", "t1"]},
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
                    "attributes": {"capacity": 100.0},
                }
            ],
            "series": {},
        }

        # 应该能成功构建
        model = build_water_network_model(valid_config)
        assert model is not None
        assert len(list(model.N)) == 2
        assert len(list(model.E)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

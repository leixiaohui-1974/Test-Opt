"""
案例4：简单水网调度示例

问题描述：
展示一个简单的三节点水网系统：水源 → 中间节点 → 需求点
演示基本的水量平衡、流量约束和需求满足。

主要特点：
1. 简单明了的网络拓扑
2. 基本约束展示
3. 适合学习和测试
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Feas import build_water_network_model
from Feas.visualization import configure_chinese_font
from pyomo.environ import value


def create_simple_config():
    """创建简单配置"""
    config = {
        "horizon": {"periods": ["t0", "t1", "t2", "t3", "t4"]},
        "nodes": [
            {
                "id": "source",
                "kind": "reservoir",
                "states": {
                    "storage": {
                        "bounds": (100, 1000),
                        "initial": 500,
                        "role": "storage",
                    }
                },
                "attributes": {},
            },
            {
                "id": "demand",
                "kind": "demand",
                "states": {},
                "attributes": {"demand_profile": "water_demand"},
            },
        ],
        "edges": [
            {
                "id": "pipe1",
                "kind": "pipeline",
                "from_node": "source",
                "to_node": "demand",
                "attributes": {"capacity": 100},
            }
        ],
        "series": {
            "water_demand": {
                "times": ["t0", "t1", "t2", "t3", "t4"],
                "values": [50, 60, 70, 80, 90],
            }
        },
    }
    return config


def main():
    print("=" * 60)
    print("简单水网调度示例")
    print("=" * 60)

    # 创建配置
    config = create_simple_config()

    # 构建并求解模型
    print("\n构建模型...")
    model = build_water_network_model(config)

    print("求解模型...")
    from pyomo.opt import SolverFactory

    solver = SolverFactory("glpk")
    results = solver.solve(model, tee=False)

    print("求解成功!\n")

    # 提取结果
    results_data = []
    for t in model.T:
        results_data.append(
            {
                "time": t,
                "storage": value(model.state["source", "storage", t]),
                "flow": value(model.flow["pipe1", t]),
            }
        )

    df = pd.DataFrame(results_data)
    print("结果:")
    print(df.to_string(index=False))

    # 保存结果
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "results.csv", index=False)

    # 简单可视化
    configure_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    times = range(len(df))
    ax1.plot(times, df["storage"], marker="o", linewidth=2)
    ax1.set_ylabel("库容 (m³)")
    ax1.set_title("水库水位变化")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, df["flow"], marker="s", linewidth=2, color="green")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("流量 (m³)")
    ax2.set_title("管道流量")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "results.png", dpi=150)
    print(f"\n结果已保存到: {output_dir}")
    print("  - results.csv")
    print("  - results.png")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

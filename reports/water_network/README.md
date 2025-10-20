# 串联闸群（2 闸 3 渠池）示例工作流

本目录记录串联闸群（库→闸 1→渠池 1→开渠→渠池 2→闸 2→渠池 3→出水）案例的脚本与数据输出，涵盖了建模、线性化、MPC 原型与可视化环节。

## 1. 构建线性化数据

```bash
python scripts/export_water_network_linearization.py \
    --output reports/water_network/gate_chain_linearization.json
```

输出结果包含：
- `state_piecewise`、`state_couplings`：节点状态的分段映射与线性耦合信息；
- `gate_tables`：闸门开度曲线；
- `sos2_cost_tables`：泵送或闸门的 SOS2 成本（若有）；
- `idz_dynamics`：渠池放流（IDZ）动力学参数（含响应时间、行波延迟、离散步长）；
- `disturbances`：节点/边扰动通道及对应的外生序列元数据；
- `time_step_seconds`：整个系统的离散时间步长。

## 2. MPC/LQR 原型演示

```bash
python scripts/mpc_gate_chain.py \
    --output reports/water_network/gate_chain_mpc_result.json
```

脚本主要操作：
1. 调用 `build_gate_chain_config()` 构造 2 闸 3 渠池串联系统；
2. 触发线性化导出，读取 IDZ 动态与扰动通道；
3. 构建简化状态空间（放流为状态、闸门放流为控制、计划分水为扰动）；
4. 基于离散 LQR （Ricatti 迭代）生成反馈增益并模拟闭环；
5. 输出轨迹（状态、控制、参考、扰动），供可视化分析。

> 注意：该原型仅用于演示，未考虑约束（如闸门开度上下限、放流约束、实际调度目标等），后续可利用该数据进一步集成完整 MPC 框架。

## 3. 可视化诊断

```bash
python scripts/plot_gate_chain_results.py \
    --input  reports/water_network/gate_chain_mpc_result.json \
    --output reports/water_network/gate_chain_mpc_diagnostics.png
```

生成图像包括：
- 渠池放流 vs 参考（确认追踪效果）；
- 闸门控制与扰动通道对比；
- 放流追踪误差。

若不提供 `--output`，脚本会直接展示 Matplotlib 图窗。

## 4. 相关脚本/文件

| 文件                                             | 说明                                                         |
|--------------------------------------------------|--------------------------------------------------------------|
| `scripts/run_water_network_gate_chain.py`        | 构建并求解串联闸群 Pyomo 模型                                |
| `scripts/export_water_network_linearization.py`  | 输出线性化所需的曲线、耦合、IDZ 动态、扰动通道等             |
| `scripts/mpc_gate_chain.py`                       | 简版 MPC / LQR 原型（输出 JSON 结果）                        |
| `scripts/plot_gate_chain_results.py`              | 对 MPC 结果进行多图诊断（水位、放流、开度比、扰动、误差）    |
| `reports/water_network/gate_chain_mpc_result.json` | MPC 输出样例（运行脚本后生成）                               |
| `reports/water_network/gate_chain_mpc_diagnostics.png` | 可视化示例（运行绘图脚本后生成）                             |

## 5. 后续拓展建议

- 将 MPC 原型拓展为带约束的预测控制器（可借助 `cvxpy`/`pyomo` 等库）；
- 在 IDZ 状态空间中引入更多渠段/水位状态，细化空间分辨率；
- 扩展可视化脚本，对各渠池水位/流量、开度限制等进行更多诊断；
- 补充 README 中的参数说明与调参建议。搬迁案例至文档系统时，可直接引用此文件。

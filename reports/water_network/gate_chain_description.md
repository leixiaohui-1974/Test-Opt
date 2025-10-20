## 系统背景

“最小串联闸群”案例用于演示水网平台在 **IDZ 渠道建模、扰动接口、闸门线性化** 与 **MPC 原型** 方面的组合能力。网络拓扑按照水流方向依次为：上游水库 → Gate1 → IDZ 渠段 1 → Gate2 → 合并后的下游 IDZ 渠段 → 下游需求点。渠道 2 与渠道 3 合并为单段，IDZ 参数保留行波时滞与回水效应。

### 核心目标

- Pyomo 优化：给定入流、需求与扰动计划，求解闸门放流与渠道流量，实现供需平衡并最小化能耗/缺水惩罚；
- MPC/LQR 原型：利用线性化得到的离散状态空间，提前调节闸门流量抵消已知扰动；
- 可视化诊断：输出水位、放流、闸门开度与扰动时间序列，生成纵剖面水位动画观察闸前/闸后台阶。

## 水网拓扑

| ID | 角色/类型 | 说明 |
| --- | ---------- | ---- |
| reservoir | boundary | 上游水库（带水位-库容映射） |
| reach1 | idz_channel | IDZ 渠段 1（Gate1 下游） |
| reach2 | idz_channel | 合并后的下游 IDZ 渠段（Gate2 下游至出水口） |
| outlet | demand | 下游需求/出水口 |

| 组件 ID | 类型 | 描述 |
| ------- | ---- | ---- |
| gate1 | gate_flow | reservoir → reach1，闸门 G1（含二维曲线与解析公式） |
| gate2 | gate_flow | reach1 → reach2，闸门 G2 |
| channel23 | open_channel | reach2 → outlet，下游渠道段 |

扰动通道：`reach1_diversion`、`reach2_diversion` 分别作用于两段渠道，均为阶跃计划分水。

## 关键参数概览

- **时间步长**：默认 1 小时，可用 `--dt-hours` 调整。
- **IDZ 参数**：
  | 渠段 | 长度 (m) | storage_time (s) | wave_speed (m/s) |
  | ---- | -------- | ---------------- | ---------------- |
  | reach1 | 450 | 14 400 | 0.05 |
  | reach2 | 600 | 18 000 | 0.04 |
  长时滞与较低波速确保存在回水区与传播延迟。
- **闸门曲线与解析公式**：`C_d = 0.64/0.60`，`head_reference ≈ 0.05m`，闸前后仅保留小跌水。
- **时间序列**：`reservoir_inflow`、`downstream_demand`、`reach1_diversion`、`reach2_diversion`。

## 模型与诊断

- Pyomo 模型通过 `run_water_network_gate_chain.py` 构建，目标函数 `pumping_cost * 总能耗 + shortage_penalty * 总缺水量`。
- MPC 原型 (`scripts/mpc_gate_chain.py`) 输出 `gate_chain_mpc_result.json`，配合 `plot_gate_chain_results.py`、`animate_gate_chain_profile.py` 生成诊断图与纵剖面动画。
- 纵剖面动画展示闸门小跌水、渠道沿程逐步下降，并同步显示扰动阶跃。

## 使用建议

1. 运行 `run_water_network_gate_chain.py` 验证基准优化可行性。
2. 依次执行线性化 → MPC → 绘图 → 动画脚本，观察扰动传播与闸门响应。
3. 调整 `reach*_diversion`、`storage_time`、`wave_speed` 等 IDZ 参数，可模拟不同时滞/回水特性；必要时使用 `--dt-hours` 改变 MPC 采样时间。

## IDZ ״̬���¹���

- ÿһ�����ӽڵ㶼����� upstream/downstream ˮ��͵�֡�
  - channel_upstream_discharge ��¼�������뾶��ͬʱ��ͨ�����һ�������Լ束�����ڳ�ʼ��ƽ��
  - channel_downstream_discharge �ǰ������շ���, ͨ�� IDZ ��̬��ʱ����ϵ����ʵ�뾶��Ӧ
  - channel_upstream_level �� channel_downstream_level �����������뾶-ˮ��ӳ���ṩ������ʽ, �ʺ�����ʾ闸ǰ/��ˮƽ
- OptiChat/scripts/animate_gate_chain_profile.py --show-anomaly ���Դӿ� GIF, ��ʽ�������뾶ˮλ����ʾ�������뾶�۵���Ӱ


## ʹ��ע��

- �Ƽ�ʹ�� HiGHS (appsi_highs/highs) ���� GLPK ������������װ���ߵ��ү��Ӳ���Ӧʹ� ÷���·���ѡ���ʻ��������� --solver highs
- ƽ����ת - ���ڻ�ˮ��֤�����׶Σ��ɸ���渠����ؽṹ���Ż� reach ������ߡŵ��Լ storage_time/wave_speed ��ִ����Ӱ��IDZ �ӳٶȡ�


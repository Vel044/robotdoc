# 实验 08 — 用 action 替代 observation.state 作为关节状态输入

> 灵感来源：实验 01（chunk_size 参数扫描）分析过程中的推论

## 背景与动机

ACT 模型的关节状态输入来自舵机编码器的**实测反馈值**（`observation.state`）。舵机反馈存在固有的物理滞后：电机惯性、通信延迟、PID 响应时间叠加，导致模型看到的"当前状态"落后于已经命令的目标位置。

**核心假设**：如果把上一帧推理输出的 `action_{t-1}` 直接替换掉 `observation.state_t`，模型就在"自己的命令空间"内自洽运行，可能缓解 chunk 边界处的轨迹跳变问题。

```
当前方案：obs_t = [camera_image, joint_pos_observed_t]
新方案：  obs_t = [camera_image, action_{t-1}]
```

## 理论分析

### 预期收益

| 方面 | 分析 |
|------|------|
| 自洽性 | 模型上一步输出直接作为下一步输入，输入输出形成闭合链，消除"命令-反馈"之间的偏差 |
| chunk 边界连续性 | 上一 chunk 最后一帧 action 作为下一 chunk step 0 的"当前状态"，两段轨迹衔接理论上更平滑 |
| OOD 程度降低 | 训练时数据中 `obs_{k+1}.state ≈ action_k`（理想执行），推理时若也用 `action_{k}` 作为 `obs_{k+1}.state`，分布更接近训练数据 |

### 已知风险

| 风险 | 说明 | 缓解方向 |
|------|------|---------|
| 累积误差 | 若舵机因碰撞/堵转等原因未执行到命令位置，"想象状态"与现实持续偏离，无法自我纠正 | 可加入周期性的真实状态重置（每 N 步同步一次实测值） |
| 训练/推理分布不一致 | 现有模型训练数据 `observation.state` 均为实测值，不能直接用；需重新采集数据 + 重新训练 | 无法绕过，必须重采数据 |
| 无法感知外界扰动 | 外力、夹取阻力等无法通过关节反馈感知 | 与视觉信号配合，靠图像判断是否执行成功 |

## 实验设计

### 阶段一：验证思路可行性

1. **修改数据采集**：在 `record.py` 的 `build_dataset_frame` 中，将写入 `observation.state` 的值改为上一帧的 `action`（首帧用初始关节角）
2. **重新采集数据**：用同一任务（抓取瓶子），同一硬件，采集新的训练数据集
3. **重新训练模型**：使用修改后的数据，保持 `chunk_size=100`（已验证可成功的基线）
4. **对比评估**：与原始 `observation.state` 方案对比任务完成率和轨迹平滑度

### 阶段二：小 chunk_size 下的效果

若阶段一成功，在 `chunk_size=10` 下重复测试，验证 action 替代 state 是否能缓解 chunk 边界跳变导致的失败。

### 对照组

| 方案 | chunk_size | observation.state 输入 |
|------|-----------|----------------------|
| 基线（实验01结果）| 100 | 实测值 |
| 新方案 A | 100 | 上一帧 action |
| 新方案 B | 10 | 上一帧 action |

## 代码改动点

### 数据采集侧（`record.py`）

```python
# build_dataset_frame 中，用 last_action 替换 observation.state
# last_action: 上一帧推理输出的 action（首帧用 robot.get_observation()["observation.state"] 初始化）
observation_frame["observation.state"] = last_action  # ndarray, shape=(action_dim,)
```

### 推理侧（`control_utils.py`）

推理时的 `observation["observation.state"]` 同样改为上一帧 action，保持与训练一致。

## 待定问题

1. 首帧 `action_{-1}` 如何初始化？用真实初始关节角还是全零？
2. 任务重置后（第二个 episode 开始），上一帧 action 是否需要重置为当前真实状态？
3. 如果舵机大幅未跟上命令（如 10° 误差积累），是否需要 clamp 或周期同步？

## 预期结论方向

- **乐观估计**：chunk 边界跳变减少，cs=10 成功率上升，临界 chunk_size 阈值从 ~20 降至 ~5~10
- **悲观估计**：累积误差导致"想象状态"快速偏离现实，模型在错误的状态空间中规划，反而更差
- **最可能结果**：在无外界扰动的简单抓取任务上有改善；在有摩擦/碰触的场景下因无法感知阻力而失败

## 状态

- [ ] 待执行
- 提出时间：2026-05-06
- 来源实验：[01_chunk_size参数扫描](../01_chunk_size参数扫描/README.md)

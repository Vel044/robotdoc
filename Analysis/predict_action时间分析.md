# ACT 推理瓶颈分析

## 问题描述

用户反馈：ACT 模型配置 `chunk_size=100`，意味着第一帧推理后，接下来100帧不调用模型（从队列取动作），但这100帧的 `inference_time` 仍然消耗大量时间。

## 日志数据分析

### 第一次调用（模型推理，填充动作队列）
```
[predict_action] total=3231.5ms | copy=0.0ms | tensor_loop=30.9ms | preprocessor=11.6ms | select_action=3183.0ms | postprocessor=0.2ms | final=0.6ms
```
- `select_action=3183ms`：模型真正推理一次，生成100个动作填充队列

### 后续调用（从队列取动作，不推理）
```
[predict_action] total=15~190ms | tensor_loop=8~108ms | preprocessor=5~83ms | select_action=1.3~1.8ms | ...
```

| 阶段 | 第一次推理 | 后续帧（不推理）| 占比 |
|------|-----------|----------------|------|
| tensor_loop | 30.9ms | 8~108ms | **主要瓶颈** |
| preprocessor | 11.6ms | 5~83ms | **主要瓶颈** |
| select_action | 3183ms | 1.3~1.8ms | ~1% |
| postprocessor | 0.2ms | 0.1~3ms | 可忽略 |
| copy | 0.0ms | 0.0ms | 可忽略 |

## 瓶颈根因

即使不调用模型推理，每帧仍然执行：

### 1. tensor_loop（每帧必执行）
```python
for name in observation:
    observation[name] = torch.from_numpy(observation[name])  # numpy→tensor
    if "image" in name:
        observation[name] = observation[name].type(torch.float32) / 255  # 归一化
        observation[name] = observation[name].permute(2, 0, 1).contiguous()  # HWC→CHW
    observation[name] = observation[name].unsqueeze(0)  # 加batch维
    observation[name] = observation[name].to(device)  # 移动到设备
```
- **图像 HWC→CHW 转换**：`permute(2, 0, 1)` 在 ARM CPU 上较慢
- **内存拷贝**：`torch.from_numpy()` 和 `to(device)` 都有拷贝开销
- **数据类型转换**：uint8 → float32

### 2. preprocessor（每帧必执行）
```python
NormalizerProcessorStep(...)  # 归一化处理器
```
- 遍历所有观测特征，进行统计归一化
- 在 ARM CPU 上计算密集

## 问题本质

`predict_action` 设计上**每一帧都会做完整的观测处理**，无论是否需要模型推理。

```python
# record.py:368
action_values = predict_action(
    observation=observation_frame,  # 每帧构建新字典
    policy=policy,
    ...
)
```

而在 `select_action` 内部：
```python
# modeling_act.py:115
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)  # 模型推理
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()  # 队列有动作则直接返回
```

**只有队列空时才真正推理，但 tensor_loop 和 preprocessor 每帧都执行。**

## 优化方向

### 方向 1：跳过不必要的观测处理（当队列有动作时）
```python
# 在 predict_action 或调用处，当队列有动作时，跳过观测转换
if len(policy._action_queue) > 0:
    # 跳过观测处理，直接取动作
    return policy._action_queue.popleft().to("cpu")
```

### 方向 2：缓存 tensor 转换结果
- 观测字典的图像部分可以缓存 tensor，避免每帧重复转换

### 方向 3：延迟预处理
- 将预处理推迟到真正需要模型推理时再做

## 结论

| 指标 | 第一帧（推理）| 后续帧（取动作）|
|------|--------------|-----------------|
| 实际耗时 | ~3232ms | 15~190ms |
| 模型推理 | 3183ms | 0ms |
| tensor+preprocess | ~42ms | 13~191ms |

**后续帧 90%+ 的时间浪费在 tensor_loop 和 preprocessor 上，而非模型推理。**

由于用户使用 `n_action_steps=100`，第一帧推理生成100个动作，但每帧仍需 15~190ms 处理观测数据，这在 ARM CPU 上是主要瓶颈。

# ACT 推理计算全流程分析（`predict_action_chunk`）

本文档对应源码：[modeling_act.py](../../../lerobot/src/lerobot/policies/act/modeling_act.py)、[configuration_act.py](../../../lerobot/src/lerobot/policies/act/configuration_act.py)

目标：**只考虑推理**（`@torch.no_grad()`），搞清楚一次 `predict_action_chunk` 内部，从 `observation.state` + 多路摄像头图像，到输出 `(B, chunk_size, action_dim)` 动作序列，究竟做了哪些矩阵乘法、注意力、卷积和加法。

---

## 0. 本次实验的具体配置

用户的录制命令：

```bash
python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.cameras="{'handeye': {'index_or_path': 0, 'width': 640, 'height': 360},
                    'fixed':   {'index_or_path': 2, 'width': 640, 'height': 360}}" \
  --policy.path=${HF_USER}/so101_act_bottle_cs100
```

结合 ACT 默认配置（chunk_size=100，故命名 `cs100`），得到以下形状表。推理时 `B=1`。

| 符号          | 含义                              | 值                                                                                                                           |
| ------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `B`           | batch_size 推理时单帧推理         | `1`                                                                                                                          |
| `S`           | state_dim（关节数）               | `6`（SO101 单臂 6 DOF）                                                                                                      |
| `A`           | action_dim                        | `6`                                                                                                                          |
| `T`           | chunk_size（一次预测的动作步数）  | `100`                                                                                                                        |
| `D`           | dim_model（Transformer 隐藏维度） | `512`                                                                                                                        |
| `H`           | n_heads                           | `8`（每头 `D/H = 64` 维）                                                                                                    |
| `F`           | dim_feedforward                   | `3200`                                                                                                                       |
| `L_enc`       | Transformer encoder 层数          | `4`                                                                                                                          |
| `L_dec`       | Transformer decoder 层数          | `1`（与原 ACT 代码对齐，见 [configuration_act.py:168](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L168)） |
| `Z`           | latent_dim                        | `32`                                                                                                                         |
| `N_cam`       | 摄像头数                          | `2`（handeye + fixed）                                                                                                       |
| `H_in × W_in` | 输入图像尺寸                      | `360 × 640`                                                                                                                  |
| `h × w`       | ResNet18 `layer4` 特征图尺寸      | `360/32 × 640/32 = 12 × 20 = 240` 个空间位置                                                                                 |
| `C_res`       | ResNet18 `layer4` 通道数          | `512`                                                                                                                        |
| `N_img`       | 所有摄像头的图像 token 总数       | `N_cam · h · w = 2 · 240 = 480`                                                                                              |
| `L_enc_seq`   | Transformer encoder 输入序列长度  | `1(latent) + 1(state) + 480(image) = 482`                                                                                    |

推理时 `use_vae=True` 但模型处于 `eval()` 模式，走 **非训练分支**：`latent_sample = 0 ∈ ℝ^{B×Z}`，VAE encoder 不参与前向。

---

## 1. 外层入口：`predict_action`（utils/control_utils.py）

源码：[control_utils.py:126](../../lerobot/src/lerobot/utils/control_utils.py#L126)

`predict_action` 是整个推理流水线的**最外层包装**，在 `record.py` 的控制循环里每帧都被调用。它的任务是把原始传感器数据（numpy）转成模型能吃的 Tensor，调用策略，再把结果转回机器人能执行的物理量。

### 1.0 完整源码

```python
def predict_action(
    observation: dict[str, np.ndarray],   # 原始传感器数据：{key: np.ndarray}
    policy: PreTrainedPolicy,             # 已加载的 ACT 策略模型
    device: torch.device,                 # 推理设备（树莓派上为 cpu）
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    # preprocessor: 归一化流水线（MEAN_STD 归一化 + ImageNet 归一化）
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    # postprocessor: 反归一化流水线（归一化空间 → 真实物理单位）
    use_amp: bool,                        # 是否开启混合精度（树莓派 CPU 上无效）
    task: str | None = None,              # 任务描述字符串（语言条件策略用）
    robot_type: str | None = None,        # 机器人类型字符串（ACT 不使用）
):
    from lerobot.constants import ACTION

    # ── 完整推理路径：队列为空时执行（每 chunk_size 帧触发一次）────────────
    observation = copy(observation)    # 浅拷贝，不改原始 obs dict

    with (
        torch.inference_mode(),        # 关闭梯度，省内存和时间
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp
        else nullcontext(),            # 树莓派 CPU 不开混合精度
    ):
        # ── 步骤 A：numpy → Tensor，图像格式转换 ─────────────────────────
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                # 图像原始格式：(H=360, W=640, C=3) uint8 [0,255]（OpenCV 输出）
                observation[name] = observation[name].type(torch.float32) / 255
                # (H, W, C) → (C, H, W)，PyTorch 要求 channel-first
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            # unsqueeze(0)：加 batch 维 → (1, ...)，满足模型对 B≥1 的要求
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)   # 移到推理设备（cpu）

        # ── 步骤 B：附加元数据 ────────────────────────────────────────────
        observation["task"] = task if task else ""
        # robot_type 会在 preprocessor 内部被 batch_to_transition 静默丢弃
        observation["robot_type"] = robot_type if robot_type else ""

        # ── 步骤 C：预处理流水线（归一化）────────────────────────────────
        # observation.state: (x - μ_state) / σ_state → 均值0方差1
        # observation.images.*: (pixel - μ_ImageNet) / σ_ImageNet → ResNet 友好
        observation = preprocessor(observation)

        # ── 步骤 D：模型推理 ──────────────────────────────────────────────
        # select_action 内部：predict_action_chunk → ACT.forward
        # 输出第 0 帧动作并将后续 99 帧缓存到 _action_queue
        action = policy.select_action(observation)   # Tensor: (1, action_dim) = (1, 6)

        # ── 步骤 E：后处理（反归一化）────────────────────────────────────
        action = postprocessor(action)   # action_real = action_norm * σ_action + μ_action
        action = action.squeeze(0)       # (1, 6) → (6,)
        action = action.to("cpu")        # 树莓派：通常 no-op

    return action   # Tensor: (6,) float32，单位：舵机目标角度（度）
```

---

### 1.1 快速路径：动作分块缓存（Action Chunking），这里是我后来加的优化，本来所有都会走预处理很浪费时间

```python
# ACT 每次推理产出 chunk_size=100 帧动作，缓存在 _action_queue
# 后续 99 帧直接 popleft，不重新推理
if hasattr(policy, "_action_queue") and len(policy._action_queue) > 0:
    action = policy._action_queue.popleft()   # Tensor (1, action_dim)
    action = postprocessor(action)             # 反归一化 → 真实关节角度
    return action.squeeze(0).to("cpu")         # → (action_dim,) = (6,)
```

**重要**：ACT 每 100 步才真正跑一次 `ACT.forward`，其余 99 步只是从队列里取缓存值，这也是在树莓派 5 上能跑通的关键原因。

### 1.2 完整推理路径（队列为空时）

```python
observation = copy(observation)    # 浅拷贝，不改原始 obs dict
```

#### 步骤 A：numpy → PyTorch Tensor，格式转换

```python
for name in observation:
    # observation[name]: np.ndarray，来自摄像头 / 舵机驱动
    # numpy → PyTorch Tensor
    observation[name] = torch.from_numpy(observation[name])

    if "image" in name: # 图像处理
        # 图像原始格式: (H=360, W=640, C=3) uint8 [0,255]（OpenCV 输出）
        # 1.类型转化 归一化到 [0, 1]
        observation[name] = observation[name].type(torch.float32) / 255
        # permute: (H,W,C) → (C,H,W)，PyTorch 要求 channel-first
        # OpenCV / 摄像头的图像格式是 (H, W, C)，PyTorch 的卷积层要求 (C, H, W)：
        observation[name] = observation[name].permute(2, 0, 1).contiguous()

    # unsqueeze(0): 加 batch 维 → (1, ...)，满足模型对 B≥1 的要求
    # 模型训练时输入是一批数据，batch 纬度为1
    # state:  (6,)          →  unsqueeze(0)  →  (1, 6)
    # image:  (3, 360, 640) →  unsqueeze(0)  →  (1, 3, 360, 640)
    observation[name] = observation[name].unsqueeze(0)
    observation[name] = observation[name].to(device)   # 移到推理设备（cpu）
```

本次配置（SO101 + 2 摄像头）执行后各 key 的形状：

| key                          | 转换前（numpy）       | 转换后（Tensor）                   |
| ---------------------------- | --------------------- | ---------------------------------- |
| `observation.state`          | `(6,)` float32        | `(1, 6)` float32，on device        |
| `observation.images.handeye` | `(360, 640, 3)` uint8 | `(1, 3, 360, 640)` float32 ∈ [0,1] |
| `observation.images.fixed`   | `(360, 640, 3)` uint8 | `(1, 3, 360, 640)` float32 ∈ [0,1] |

{
    "observation.state":           Tensor(1, 6) float32,        # 6 个关节角度，已加 batch 维
    "observation.images.handeye":  Tensor(1, 3, 360, 640) float32,  # ∈ [0,1]，channel-first
    "observation.images.fixed":    Tensor(1, 3, 360, 640) float32,  # ∈ [0,1]，channel-first
    "task":                        "",                          # 语言条件策略用，ACT 不用
    "robot_type":                  "",                          # 同上，会被后续静默丢弃
}


#### 步骤 C：预处理流水线（归一化）

```python
# preprocessor: PolicyProcessorPipeline（processor_act.py:197-200）
# 内部依次执行 4 个 step（processor_act.py:141-169）：
#   1. RenameObservationsProcessorStep — 重命名键（ACT 默认空 map，跳过）
#   2. AddBatchDimensionProcessorStep  — 已在步骤 B 做过，这里再确保有 batch 维
#   3. DeviceProcessorStep             — tensor 搬到 config.device（树莓派 = cpu）
#   4. NormalizerProcessorStep         — ★ 归一化核心，下面展开
observation = preprocessor(observation)
```

**`NormalizerProcessorStep` 调用链**（`normalize_processor.py`）：

```python
# ── 入口：normalize_processor.py NormalizerProcessorStep.__call__ ──
def __call__(self, transition):
    # transition = {OBSERVATION: {state:(1,6), images.handeye:(1,3,360,640), images.fixed:(1,3,360,640)},
    #               ACTION: None}   ← 推理时 ACTION 还没产出，跳过
    observation = transition[OBSERVATION]       # dict，key 是字段名
    transition[OBSERVATION] = self._normalize_observation(observation, inverse=False)
    return transition

# ── 遍历观测字段：_normalize_observation ──
def _normalize_observation(self, observation, inverse=False):
    for key, feature in self.features.items():
        # feature.type ∈ {STATE, VISUAL, ACTION, ...}
        # ACT 的 normalization_mapping 配置（configuration_act.py:113-119）：
        #   VISUAL → MEAN_STD,  STATE → MEAN_STD,  ACTION → MEAN_STD
        # 所以 state 和图像全部走同一个 MEAN_STD 分支（不是 ImageNet 常量！）
        if feature.type != ACTION and key in observation:
            tensor = torch.as_tensor(observation[key])
            # key = "observation.state", feature.type = STATE
            # key = "observation.images.handeye", feature.type = VISUAL
            # key = "observation.images.fixed", feature.type = VISUAL
            observation[key] = self._apply_transform(tensor, key, feature.type, inverse=False)
    return observation

# ── 核心数学：_apply_transform（MEAN_STD 分支）──
def _apply_transform(self, tensor, key, feature_type, *, inverse=False):
    norm_mode = self.norm_map[feature_type]     # ACT 全部是 MEAN_STD
    stats = self._tensor_stats[key]
    # stats 结构示例（来自 checkpoint 保存的训练集统计量）：
    #   observation.state:  mean=Tensor(6,),  std=Tensor(6,)     ← 6 个关节各自的 μ 和 σ
    #   observation.images.handeye:  mean=Tensor(3,1,1), std=Tensor(3,1,1)  ← 3 通道，广播到 (1,3,H,W)
    #   observation.images.fixed:    mean=Tensor(3,1,1), std=Tensor(3,1,1)
    mean, std = stats["mean"], stats["std"]
    denom = std + 1e-8                          # ε 防止 std≈0 时除零

    # 归一化公式：z = (x - μ) / (σ + ε是1e-8约等于0)   （把真实舵机角度变成无量纲数）
    return (tensor - mean) / denom
```

在pytorch/torch/_tensor.py 进行运算符重载

**经过这步的数据变化**：

| 字段                         | 输入                    | 输出                    |
| ---------------------------- | ----------------------- | ----------------------- |
| `observation.state`          | `(1,6)` 真实角度（度）  | `(1,6)` 无量纲，≈N(0,1) |
| `observation.images.handeye` | `(1,3,360,640)` ∈ [0,1] | `(1,3,360,640)` ≈N(0,1) |
| `observation.images.fixed`   | `(1,3,360,640)` ∈ [0,1] | `(1,3,360,640)` ≈N(0,1) |

归一化参数（μ, σ）全部来自**训练集自身的统计量**，保存在 checkpoint 里，不是 ImageNet 预设值。
统计量的计算和加载链路见下方 **§1.4 归一化统计量的来源**。

#### 步骤 C'：batch 摄像头键重组（`modeling_act.py:319`）

进入 `select_action` 后，在调用 `ACT.forward` 之前，把独立的摄像头键合并为一个列表：

**之前**（散装键，归一化输出）：
```python
{
    "observation.state":           Tensor(1, 6),           # 无量纲 ≈N(0,1)
    "observation.images.handeye":  Tensor(1, 3, 360, 640), # ≈N(0,1)
    "observation.images.fixed":    Tensor(1, 3, 360, 640), # ≈N(0,1)
}
```

**之后**（合并为列表，键名变为 `"observation.images"`）：
```python
{
    "observation.state":    Tensor(1, 6),
    "observation.images":  [Tensor(1, 3, 360, 640),   # handeye（index 0）
                            Tensor(1, 3, 360, 640)],   # fixed（index 1）
}
```

顺序由 `config.image_features` 决定，后续 `ACT.forward` 用 `for img in batch["observation.images"]` 逐路送入 ResNet backbone。

#### 步骤 D：模型推理

```python
# 调用 ACTPolicy.select_action，内部走 predict_action_chunk → ACT.forward
# 详见下方 §2 起的各节
action = policy.select_action(observation)   # → Tensor (1, action_dim) = (1, 6)
```

#### 步骤 E：后处理（反归一化）

```python
# postprocessor: PolicyProcessorPipeline（processor_act.py:207-212）
# 内部依次执行 2 个 step（processor_act.py:176-187）：
#   1. UnnormalizerProcessorStep — ★ 反归一化核心，下面展开
#   2. DeviceProcessorStep      — tensor.to("cpu")（树莓派上已经是 cpu，no-op）
action = postprocessor(action)   # (1, 6)，单位：舵机目标角度（度）
action = action.squeeze(0)       # → (6,)
action = action.to("cpu")        # 树莓派：模型在 cpu，这步通常是 no-op
```

**`UnnormalizerProcessorStep` 调用链**（`normalize_processor.py`）：

```python
# ── 入口：UnnormalizerProcessorStep.__call__ ──
def __call__(self, transition):
    # transition = {OBSERVATION: None, ACTION: Tensor(1,6)}   ← 只有 action
    # ACT 后处理只带 action，observation=None，跳过观测侧反归一化
    action = transition[ACTION]                 # Tensor (1,6)，归一化空间
    transition[ACTION] = self._normalize_action(action, inverse=True)
    return transition

# ── 动作反归一化：_normalize_action ──
def _normalize_action(self, action, inverse=True):
    # key 固定 "action"，norm_map[ACTION] = MEAN_STD
    return self._apply_transform(action, "action", ACTION, inverse=True)

# ── 核心数学：_apply_transform（inverse=True）──
def _apply_transform(self, tensor, key, feature_type, *, inverse=True):
    stats = self._tensor_stats["action"]
    # stats 结构：mean=Tensor(6,),  std=Tensor(6,)   ← 6 个关节各自的 μ_action, σ_action
    mean, std = stats["mean"], stats["std"]

    # 反归一化公式：x̂ = z · σ + μ   （把无量纲数还原成真实舵机角度）
    return tensor * std + mean
```

**经过这步的数据变化**：

| 字段     | 输入                       | 输出                                            |
| -------- | -------------------------- | ----------------------------------------------- |
| `action` | `(1,6)` ≈N(0,1) 归一化空间 | `(1,6)` 真实角度（度），可直接写入 Feetech 舵机 |


### 1.3 归一化统计量（μ, σ）的来源

归一化用的 mean/std 不是 ImageNet 常量，而是**训练集自身的统计量**，通过 `compute_stats.py` 计算，随模型保存到 checkpoint，推理时从 checkpoint 加载。

**计算阶段**（训练时，`compute_stats.py`）：

```python
# ── 第 1 步：单个 episode 内统计量 ──
# get_feature_stats() — compute_stats.py:75-82
# 对一个 episode 的数据沿指定 axis 做 numpy 统计
def get_feature_stats(array, axis, keepdims):
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),   # ← std 诞生地
        "count": np.array([len(array)]),
    }

# compute_episode_stats() 为不同类型选择归约轴：
#   state/action: axis=0（沿时间轴）→ mean/std shape=(6,)
#   image: axis=(0,2,3)（沿 batch+H+W）→ mean/std shape=(3,1,1)，只保留通道维
#   图像额外做 /255.0 把 [0,255] 转到 [0,1] 后再算统计量

# ── 第 2 步：多个 episode 聚合为全局统计量 ──
# aggregate_feature_stats() — compute_stats.py:126-152
# 使用并行方差算法（parallel variance algorithm）：
#   total_mean = Σ(μ_i × n_i) / Σ(n_i)              ← 加权均值
#   total_var  = Σ((σ_i² + (μ_i - total_mean)²) × n_i) / Σ(n_i)
#   total_std  = √(total_var)                         ← 全局 std
# 注意：不能简单平均各 episode 的 std，必须用方差公式严格合并
```

**保存与加载链路**：

```
训练阶段：
  compute_stats.py → dataset.meta.stats = {
      "observation.state":          {"mean": array(6,), "std": array(6,)},
      "observation.images.handeye": {"mean": array(3,1,1), "std": array(3,1,1)},
      "observation.images.fixed":   {"mean": array(3,1,1), "std": array(3,1,1)},
      "action":                     {"mean": array(6,), "std": array(6,)},
  }
      ↓
  processor_act.py → NormalizerProcessorStep(stats=dataset_stats, ...)
      ↓
  训练结束 → policy.save_pretrained() 把 stats 写入 preprocessor.safetensors
      key 格式："normalizer_processor.observation.state.mean" → Tensor(6,)
                "normalizer_processor.observation.state.std"  → Tensor(6,)
                "normalizer_processor.action.mean"            → Tensor(6,)
                ...

推理阶段：
  record.py:1083 → make_pre_post_processors(pretrained_path="xxx/checkpoint")
      ↓
  factory.py:203 → PolicyProcessorPipeline.from_pretrained(pretrained_path)
      ↓ 读 preprocessor.json（流水线结构） + preprocessor.safetensors（统计量）
  normalize_processor.py:171 → load_state_dict()
      ↓ 展平的 key 反序列化回 self._tensor_stats 嵌套 dict
  self._tensor_stats = {
      "observation.state":          {"mean": Tensor(6,), "std": Tensor(6,)},       ← 这就是推理时用的
      "observation.images.handeye": {"mean": Tensor(3,1,1), "std": Tensor(3,1,1)},
      "observation.images.fixed":   {"mean": Tensor(3,1,1), "std": Tensor(3,1,1)},
      "action":                     {"mean": Tensor(6,), "std": Tensor(6,)},
  }
      ↓
  每帧推理：_apply_transform 里 stats["std"] ← 就是上面这个值
```

---

## 2. 顶层调用链 

```bash
ACTPolicy.select_action(batch)                     # modeling_act.py:224
  │
  ├─ [路径 B] 队列为空时：
  │    actions = self.predict_action_chunk(batch)  # modeling_act.py:289
  │        │
  │        ├─ 把 config.image_features 里的 key 聚合成
  │        │  batch["observation.images"] = [img_handeye, img_fixed]
  │        │
  │        └─ actions = self.model(batch)[0]       # ACT.forward, modeling_act.py:782
  │               │
  │               ├─ (1) 准备 latent
  │               ├─ (2) 构造 encoder 输入 token
  │               ├─ (3) Transformer Encoder × 4 层
  │               ├─ (4) Transformer Decoder × 1 层
  │               └─ (5) action_head 线性投影
  │
  └─ 从队列 popleft 返回 (B, A)
```

`@torch.no_grad()` 保证整个路径不建计算图（[modeling_act.py:288](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L288)）。`self.eval()` 关闭 dropout，`FrozenBatchNorm2d` 使得 ResNet 的 BN 退化成固定仿射。

### 2.1 推理时 `ACT.forward` 完整执行的代码

下面这段是把 [modeling_act.py:782-979](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782-L979) 的 `ACT.forward` 按 **SO101 + 2 摄像头 + `eval()`** 这次命令的实际配置"展平"的结果：所有永远走不到的训练分支、`env_state` 分支、`else batch_size` 分支、训练期 `assert` 全部删除，剩下的每一行在推理时都会被真实执行。后面的 §3–§7 只对这段代码做数学和 FLOPs 展开，不再重复贴代码。

```python
# 关闭梯度计算，因为推理时不需要反向传播
@torch.no_grad()
def forward(self, batch):
    # 确定 batch_size：推理时每次只传入一帧观测，所以 batch_size=1
    # batch["observation.images"] 是摄像头图像列表，[0] 取第一路，.shape[0] 即 B 维度
    batch_size = batch["observation.images"][0].shape[0]  # 推理时 = 1

    # ── 步骤 1：latent（推理时直接全零，不走 VAE encoder） ─────────
    mu = log_sigma_x2 = None # 不走VAE,推理时无需返回分布参数
    # 隐变量全部置0
    latent_sample = torch.zeros(
        [batch_size, self.config.latent_dim],  # Tensor(1, 32) 全零
        dtype=torch.float32,
    ).to(batch["observation.state"].device) # 潜变量全部全零

    # ── 步骤 2：构造 Transformer Encoder 输入序列 ─────────────────
    # token 顺序: [latent, robot_state, cam0_pixels..., cam1_pixels...]
    encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(
        self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
    )

    # Robot state token so101有关节状态，走着里
    encoder_in_tokens.append(
        self.encoder_robot_state_input_proj(batch["observation.state"])
    )

    # 图像 token（每路摄像头 240 个像素 token）
    for img in batch["observation.images"]:
        # ① ResNet18.layer4: (B, 3, 360, 640) → (B, 512, 12, 20)
        cam_features = self.backbone(img)["feature_map"]
        # ② 2D 正弦位置编码: (B, dim_model, 12, 20)
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
            dtype=cam_features.dtype
        )
        # ③ 1×1 卷积投影通道数: 512 → dim_model
        cam_features = self.encoder_img_feat_input_proj(cam_features)
        # ④ 展平空间维度，每个像素变成一个 token
        cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
        cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
        encoder_in_tokens.extend(list(cam_features))
        encoder_in_pos_embed.extend(list(cam_pos_embed))

    # 堆叠: (seq_len=482, B, dim_model)
    encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

    # ── 步骤 3：Transformer Encoder × 4 层 ────────────────────────
    encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

    # ── 步骤 4：Transformer Decoder × 1 层（DETR 风格 object queries）──
    decoder_in = torch.zeros(
        (self.config.chunk_size, batch_size, self.config.dim_model),  # (100, 1, 512) 全零
        dtype=encoder_in_pos_embed.dtype,
        device=encoder_in_pos_embed.device,
    )
    decoder_out = self.decoder(
        decoder_in,
        encoder_out,
        encoder_pos_embed=encoder_in_pos_embed,
        decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
    )
    decoder_out = decoder_out.transpose(0, 1)   # (chunk_size, B, D) → (B, 100, D)

    # ── 步骤 5：动作输出头 ────────────────────────────────────────
    actions = self.action_head(decoder_out)     # (B, 100, 6)

    return actions, (mu, log_sigma_x2)
```

几个关键的"被删掉"的点（解释一下为什么不在上面）：

- `if self.config.use_vae and self.training: assert "action" in batch` —— `self.training=False`，整句不执行
- `if "observation.images" in batch: ... else: batch_size = batch["observation.environment_state"].shape[0]` —— 走真分支，else 整段跳过
- 整个 VAE 编码器训练分支（`cls_embed` / `vae_encoder_action_input_proj` / `self.vae_encoder(...)` / `latent_pdf_params` / 重参数化采样）—— 条件 `self.config.use_vae and "action" in batch and self.training` 为 False，整段跳过。**但这些子模块的参数仍然被 `ACT.__init__` 无条件构造并加载到内存**，属于"加载了但永远不跑"，详见 §2.2
- `if self.config.env_state_feature:` 追加环境状态 token —— SO101 没有这个特征，整段跳过

---

## 3. 步骤 1：latent（推理时为全零）

### 3.1 数学表达


$$
z = \mathbf{0} \in \mathbb{R}^{B \times Z} = \mathbb{R}^{1 \times 32}
$$

```python
# 推理时没有 action 可编码，latent 直接置全零 Tensor(1, 32)
latent_sample = torch.zeros(
    [batch_size, self.config.latent_dim],  # (1, 32)
    dtype=torch.float32,
).to(batch["observation.state"].device)
```

## 4. 步骤 2：构造 Encoder 输入 token

```python
# latent token: (B,32) → Linear(32→512) → (B,512)，推理时全零
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
# robot_state token: (B,6) → Linear(6→512) → (B,512)
encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
# 图像 token 见 §4.2
```

### 4.0 这一步到底做了什么

把 3 类**维度不同**的原始数据（32 维 latent、6 维 state、每个像素 512 维的特征图）**统一投影到 D=512 维**，然后拼成一个长度 482 的 token 序列，喂给 Transformer Encoder。

为什么必须统一到 512 维：Transformer 的 self-attention 要求**所有 token 同维度**才能做矩阵乘法（Q·K^T）。6 维的关节状态不投影，根本没法和 512 维的图像特征"坐在同一张桌子上"做注意力。

### 4.1 输入序列的长什么样

$$
X_{enc}^{(0)} = [\underbrace{x_{latent}}_{1},\; \underbrace{x_{state}}_{1},\; \underbrace{x_{cam0,1}, \dots, x_{cam0,240}}_{240},\; \underbrace{x_{cam1,1}, \dots, x_{cam1,240}}_{240}] \in \mathbb{R}^{482 \times 1 \times 512}
$$

| 位置        | token 数 | 来源                           | 投影方式              |
| ----------- | -------- | ------------------------------ | --------------------- |
| `[0]`       | 1        | latent（推理时全零）           | `nn.Linear(32 → 512)` |
| `[1]`       | 1        | 6 个关节角度                   | `nn.Linear(6 → 512)`  |
| `[2:242]`   | 240      | handeye 摄像头（12×20 特征图） | ResNet18 + 1×1 Conv   |
| `[242:482]` | 240      | fixed 摄像头（12×20 特征图）   | ResNet18 + 1×1 Conv   |

下面 §4.2 / §4.3 / §4.4 / §4.5 分别展开这 4 类 token 的构造过程。

---

### 4.2 Latent token（1 个）

```python
# __init__ 里定义（modeling_act.py:714）：W(512,32) 和 b(512,) 在这里创建，训练时学到
self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
#                                                         32 →        512

# forward 里调用：把 latent_sample 送进去做矩阵乘法
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
# latent_sample: (1, 32) 全零
```

**做了什么**：`nn.Linear(32→512)` 就是一个矩阵乘法，把 32 维向量变成 512 维：

```
输入 z   = [0, 0, 0, ..., 0]     ← (1,32) 全零
权重 W   = 32行×512列的矩阵       ← 训练学到的
偏置 b   = 512个数                ← 训练学到的

计算：y = z @ W.T + b
        = [0,0,...,0] @ W.T + b
        = b                       ← 全零输入，结果就等于偏置b

输出 x_latent = (1, 512)
```

推理时 latent 全零，所以这个 token 就是偏置 `b`（1，512维），是模型训练时学到的"默认意图"。

---

### 4.3 Robot state token（1 个）

```python
# __init__ 里定义（modeling_act.py:720）：W(512,6) 和 b(512,) 在这里创建，训练时学到
self.encoder_robot_state_input_proj = nn.Linear(
    self.config.robot_state_feature.shape[0], config.dim_model  # 6 → 512
)

# forward 里调用（modeling_act.py:910）：把归一化后的关节角度送进去做矩阵乘法
# batch["observation.state"] 是已经归一化的 (1,6)——归一化发生在 preprocessor 流水线里
encoder_in_tokens.append(
    self.encoder_robot_state_input_proj(batch["observation.state"])
)
```

**两阶段调用关系**：归一化和 Linear 投影不在同一个函数里，而是分属推理循环的前后两步：

```
record.py 推理循环每一帧：
  ① preprocessor(obs)                ← 流水线在 make_act_pre_post_processors() 里组装
      └→ NormalizerProcessorStep        归一化 (x-μ)/σ
  ② model.forward(batch)             ← 归一化后的 batch 传进模型
      └→ encoder_robot_state_input_proj   矩阵乘法 6→512
```

#### 归一化的完整调用链

```python
# ── ① 配置 ─ configuration_act.py ─ ACTConfig 类字段 ─────────────────
# 三种模态全部用 MEAN_STD（减均值除标准差）
normalization_mapping: dict[str, NormalizationMode] = field(
    default_factory=lambda: {
        "VISUAL": NormalizationMode.MEAN_STD,   # 图像
        "STATE":  NormalizationMode.MEAN_STD,   # 关节角度 ← robot_state 走这条
        "ACTION": NormalizationMode.MEAN_STD,   # 动作
    }
)

# ── ② 组装 ─ processor_act.py:80 ─ make_act_pre_post_processors() ───
# 推理初始化时调用一次，构建 preprocessor / postprocessor 两条流水线
# preprocessor 后来被 record.py 每帧调用：preprocessor(obs) → 归一化后的 obs
def make_act_pre_post_processors(config, dataset_stats=None):
    ...
    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,   # {"STATE": MEAN_STD, ...}
            stats=dataset_stats,   # 训练集统计量 {key: {"mean": Tensor(6,), "std": Tensor(6,)}}
            device=config.device,
        )
    )

# ── ③ 入口 ─ normalize_processor.py:361 ─ NormalizerProcessorStep.__call__() ─
# 每帧推理时由 PolicyProcessorPipeline 驱动调用
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()          # 浅拷贝，不污染调用方的 dict
        observation = new_transition.get(TransitionKey.OBSERVATION)  # 取观测 dict，含 state 和 images
        if observation is not None:
            # inverse=False 表示正向归一化 (x-μ)/σ，把真实角度/像素值压到 ≈N(0,1)
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False          # → 跳到 ④
            )
        return new_transition                       # 归一化后的 transition，交给下一步（或传给模型）

# ── ④ 遍历特征 ─ normalize_processor.py:242 ─ _normalize_observation() ─
# 遍历 config 里声明的所有 feature（observation.state、observation.images.* 等）
def _normalize_observation(self, observation: dict[str, Any], inverse: bool) -> dict[str, Tensor]:
    new_observation = dict(observation)             # 浅拷贝，value 仍指向原 Tensor
    for key, feature in self.features.items():      # key="observation.state", feature.type=STATE
        ...  # 白名单过滤、跳过 ACTION 类型
        if feature.type != FeatureType.ACTION and key in new_observation:
            tensor = torch.as_tensor(new_observation[key])  # state: float32 (1,6)；image: uint8 (1,3,H,W)
            # 按 feature.type 查 norm_map 选归一化模式，ACT 全走 MEAN_STD
            new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)  # → 跳到 ⑤
    return new_observation

# ── ⑤ 核心计算 ─ normalize_processor.py:287 ─ _apply_transform() ─
# 实际执行 (x-μ)/σ 或 x*σ+μ，按 feature_type 查 norm_map 选模式
def _apply_transform(self, tensor: Tensor, key: str, feature_type: FeatureType,
                     *, inverse: bool = False) -> Tensor:
    norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
    # 从 ① 的字典里查：STATE → MEAN_STD（减均值除标准差）
    # .get() 第二个参数是兜底值 IDENTITY（不做归一化），没在字典里的 feature 直接跳过
    if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
        return tensor                               # 不需要归一化的字段直接返回
    ...  # device/dtype 对齐（树莓派上 tensor 和 stats 都在 cpu/float32，通常不触发）
    stats = self._tensor_stats[key]                 # {"mean": Tensor(6,), "std": Tensor(6,)}

    if norm_mode == NormalizationMode.MEAN_STD and "mean" in stats and "std" in stats:
        mean, std = stats["mean"], stats["std"]     # mean=[μ₁,μ₂,...,μ₆], std=[σ₁,σ₂,...,σ₆]
        denom = std + self.eps                      # eps=1e-8 防止某关节 std≈0 除零
        if inverse:
            return tensor * std + mean              # 反归一化：归一化值 → 真实角度（⑥走这里）
        return (tensor - mean) / denom              # 正向归一化：真实角度 → ≈N(0,1) ← robot_state 走这里
    ...  # MIN_MAX 分支，Diffusion/pi0 等策略用，ACT 不走

# ── ⑥ 反归一化 ─ normalize_processor.py:441 ─ UnnormalizerProcessorStep.__call__() ─
# 后处理流水线：模型输出动作后调用，把归一化值还原成真实舵机角度发给 Feetech
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)  # 模型输出的 action: Tensor(1,6)，归一化空间
        ...
        # inverse=True → _apply_transform 里走 tensor * std + mean
        # 把 ≈N(0,1) 的值还原成真实舵机目标角度（度）
        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)
        return new_transition
```

---

**做了什么**：`nn.Linear(6→512)` 同样是矩阵乘法，把 6 个关节角度变成 512 维：

```
输入 s   = [θ1, θ2, θ3, θ4, θ5, θ6]   ← (1,6) 归一化后的关节角度（≈N(0,1)）
权重 W   = 6行×512列的矩阵              ← 训练学到的
偏置 b   = 512个数                      ← 训练学到的

计算：y = s @ W.T + b
        = (1,6) @ (6,512) + (512,)
        = (1,512)

输出 x_state = (1, 512)
```

6个关节角度里的每一个都会被"打散"贡献到512个新维度，具体怎么打散由训练学到的 W 决定。这样 state token 才能和图像 token（也是512维）一起做注意力。

**计算量**：`2·D·S ≈ 6k FLOPs`，可忽略。

---

### 4.4 图像 token（每路 240 个，共 480 个）

这是 Encoder 输入构造里**最复杂、最费时**的部分（占整次推理计算量的 >50%）。对每路摄像头图像，要依次做 4 步：

```python
# batch["observation.images"] = [Tensor(1,3,360,640), Tensor(1,3,360,640)]  ← handeye, fixed
for img in batch["observation.images"]:
    # img: (1, 3, 360, 640)，ImageNet 归一化后的 RGB 图像

    cam_features  = self.backbone(img)["feature_map"]
    # ① ResNet18.layer4：(1,3,360,640) → (1,512,12,20)   ← 空间压缩32倍，每位置512维

    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
    # ② 2D正弦位置编码：(1,512,12,20) → (1,512,12,20)   ← 告诉Transformer每个位置的(行,列)坐标

    cam_features  = self.encoder_img_feat_input_proj(cam_features)
    # ③ 1×1 Conv：(1,512,12,20) → (1,512,12,20)          ← 对每个像素独立做线性变换，适配Transformer

    cam_features  = einops.rearrange(cam_features, "b c h w -> (h w) b c")
    # ④ 展平：(1,512,12,20) → (240,1,512)                ← 12×20个位置变成240个token

    encoder_in_tokens.extend(list(cam_features))      # 追加240个token到序列
    encoder_in_pos_embed.extend(list(cam_pos_embed))  # 追加对应的240个位置编码
```

整体流程的形状变化：

| 步骤                 | 形状               | 说明                       |
| -------------------- | ------------------ | -------------------------- |
| 输入图像             | `(1, 3, 360, 640)` | ImageNet 归一化后          |
| ① ResNet18.layer4    | `(1, 512, 12, 20)` | 空间缩小 32×，通道升到 512 |
| ② 位置编码 pos_embed | `(1, 512, 12, 20)` | 每个空间位置的坐标向量     |
| ③ 1×1 Conv           | `(1, 512, 12, 20)` | 通道数不变，换表示空间     |
| ④ 展平               | `(240, 1, 512)`    | 每个像素变成一个 token     |

两路摄像头走完，共 2×240 = **480 个 image tokens**。

---

#### 4.4.1 ① ResNet18 backbone：图像 → 特征图

##### 定义与调用

```python
# __init__ 里定义（modeling_act.py:689-705）：
#   先从 torchvision 加载 ImageNet 预训练的 ResNet18
# ResNet18 源码位于 torchvision：vision/torchvision/models/resnet.py
#   conv1 + bn1 + relu + maxpool：L197-200
#   layer1/2/3/4 = self._make_layer(...)：L201-204
#   BasicBlock（每个 layer 内的基本单元）：L65-107
backbone_model = torchvision.models.resnet18(
    weights="IMAGENET1K_V1",       # 加载 ImageNet 预训练权重
    norm_layer=FrozenBatchNorm2d,  # 把所有 BatchNorm 换成冻结版
)
#   IntermediateLayerGetter：包装一下，让 forward 只跑到 layer4 就停，不走 avgpool 和 fc 分类头
self.backbone = IntermediateLayerGetter(
    backbone_model, return_layers={"layer4": "feature_map"}
)
# 返回的是 dict：{"feature_map": Tensor(B,512,h,w)}，所以下面要用 ["feature_map"] 取出来

# forward 里调用（modeling_act.py:926）：
cam_features = self.backbone(img)["feature_map"]
# img:          (1, 3, 360, 640)  ImageNet 归一化后的 RGB 图像
# cam_features: (1, 512, 12, 20)  每个空间位置一个 512 维特征向量
```

##### ResNet18 源码调用链

```python
# ── 入口 ─ vision/torchvision/models/resnet.py:684 ─ resnet18() ────────
def resnet18(*, weights: Optional[ResNet18_Weights] = None,
             progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet18_Weights.verify(weights)
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
    #                    ↑          ↑
    #              残差块类型   layer1~4 各 2 个 block

# ── 构建 ─ resnet.py:288 ─ _resnet() ─────────────────────────────────
def _resnet(block: type[Union[BasicBlock, Bottleneck]],
            layers: list[int], weights, progress, **kwargs) -> ResNet:
    ...  # 权重参数预处理
    model = ResNet(block, layers, **kwargs)  # norm_layer=FrozenBatchNorm2d 从 kwargs 传入
    if weights is not None:                                              # ACT 传了 "IMAGENET1K_V1"
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)   # 从 URL 下载 .pth → {参数名: Tensor}
        )                                                                # 灌进 model 的 conv/BN，否则全是随机值，提不出有效特征
    return model

# ── 搭建各层 ─ resnet.py:166 ─ ResNet.__init__() ─────────────────────
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, ..., norm_layer=None):
        ...  # 参数校验，默认值处理
        # ↓ 以下只是搭建结构、分配权重空间，不执行计算
        # 实际计算在推理时由 IntermediateLayerGetter.forward() 驱动（见下方）
        self.inplanes = 64  # 当前通道数，每经过一个 _make_layer 会翻倍
        self.conv1   = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = norm_layer(self.inplanes)   # FrozenBatchNorm2d（ACT 传入）
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block, 64,  layers[0])                          # 2×BasicBlock(64)
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)                # 2×BasicBlock(128)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)                # 2×BasicBlock(256)
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2)                # 2×BasicBlock(512)
        ...  # 权重初始化（kaiming 初始化 conv，zero-init 最后一个 BN）

# ── 堆叠 BasicBlock ─ resnet.py:225 ─ ResNet._make_layer() ───────────
def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
    norm_layer = self._norm_layer
    downsample = None
    ...  # dilation 处理
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),  # 1×1 卷积对齐通道和空间
            norm_layer(planes * block.expansion),
        )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, ...))  # 第 1 个 block（可能带 downsample）
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, ...))  # 后续 block 不降采样
    return nn.Sequential(*layers)

# ── 残差块 ─ resnet.py:59 ─ BasicBlock ────────────────────────────────
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ..., norm_layer=None):
        ...  # 参数校验
        self.conv1 = conv3x3(inplanes, planes, stride)  # nn.Conv2d(in, out, 3, stride, 1, bias=False)
        self.bn1   = norm_layer(planes)                  # FrozenBatchNorm2d
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)             # stride=1
        self.bn2   = norm_layer(planes)
        self.downsample = downsample  # 可选：1×1 conv + BN，对齐捷径尺寸

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)       # → FrozenBatchNorm2d.forward()
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)       # → FrozenBatchNorm2d.forward()
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity           # 残差连接
        out = self.relu(out)
        return out

# ── 冻结批归一化 ─ vision/torchvision/ops/misc.py:14 ─ FrozenBatchNorm2d ─
class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        ...  # 注册 4 个 buffer（不参与梯度）：weight(γ), bias(β), running_mean(μ), running_var(σ²)

    def forward(self, x: Tensor) -> Tensor:
        w  = self.weight.reshape(1, -1, 1, 1)       # γ: (1,C,1,1)
        b  = self.bias.reshape(1, -1, 1, 1)         # β: (1,C,1,1)
        rv = self.running_var.reshape(1, -1, 1, 1)   # σ²: (1,C,1,1)
        rm = self.running_mean.reshape(1, -1, 1, 1)  # μ: (1,C,1,1)
        scale = w * (rv + self.eps).rsqrt()          # γ / √(σ² + ε)
        bias  = b - rm * scale                      # β - μ·scale
        return x * scale + bias
    # 等价于 y[c] = γ[c]·(x[c]-μ[c])/√(σ²[c]+ε) + β[c]，与普通 BN 唯一区别是 μ/σ² 冻结不更新

# ── 截断包装 ─ vision/torchvision/models/_utils.py:13 ─ IntermediateLayerGetter ─
# __init__ 里只把 layer4 及之前的子模块装进来，avgpool/fc 直接丢弃
# 所以 ACT 推理时 ResNet 的 avgpool 和 fc 完全不存在，不会执行
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        layers = OrderedDict()
        for name, module in model.named_children():  # conv1, bn1, relu, maxpool, layer1, ..., avgpool, fc
            layers[name] = module
            if name in return_layers:                # "layer4" 找到了
                del return_layers[name]
            if not return_layers:                    # return_layers 清空
                break                                # ← 停！avgpool、fc 没装进来

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():            # 只遍历装进来的 8 个子模块
            x = module(x)                            # 逐个执行
            if name in self.return_layers:           # layer4 时捕获输出
                out_name = self.return_layers[name]
                out[out_name] = x
        return out                                   # {"feature_map": Tensor(1,512,12,20)}

# ── 前向流程 ─ resnet.py:266 ─ ResNet._forward_impl() ─────────────────
# ACT 实际走的是 IntermediateLayerGetter.forward()，等效于下面的 _forward_impl 跑到 layer4 就停
# 下面是 ResNet 原始完整逻辑，用来看数据流和形状变化
def _forward_impl(self, x: Tensor) -> Tensor:
    # ── 入口卷积 ─────────────────────────────────────────────────────────
    x = self.conv1(x)       # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                            # 64 个 7×7 滤波器扫 RGB 三通道，stride=2 空间÷2
                            # (1,3,360,640) → (1,64,180,320)
    x = self.bn1(x)         # FrozenBatchNorm2d(64)：逐通道 y=γ·(x-μ)/√(σ²+ε)+β，推理时 μ/σ/γ/β 全固定
                            # 形状不变 (1,64,180,320)，但数值分布拉回 ≈N(0,1)
    x = self.relu(x)        # max(0,x)，负数清零引入非线性，形状不变
    x = self.maxpool(x)     # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            # 3×3 窗口取最大值，stride=2 空间再÷2
                            # (1,64,180,320) → (1,64,90,160)

    # ── 4 个残差 stage ───────────────────────────────────────────────────
    x = self.layer1(x)      # 2×BasicBlock(64), stride=1, 不降采样
                            # (1,64,90,160) → (1,64,90,160)
                            # 每个 BasicBlock: conv3×3 → BN → relu → conv3×3 → BN + shortcut → relu
    x = self.layer2(x)      # 2×BasicBlock(128), 首个 block stride=2
                            # (1,64,90,160) → (1,128,45,80)
                            # 通道 64→128（滤波器翻倍），空间 90×160→45×80（÷2）
    x = self.layer3(x)      # 2×BasicBlock(256), 首个 block stride=2
                            # (1,128,45,80) → (1,256,23,40)
    x = self.layer4(x)      # 2×BasicBlock(512), 首个 block stride=2
                            # (1,256,23,40) → (1,512,12,20)
                            # ← IntermediateLayerGetter 在这里截断并返回 feature_map
                            # 总下采样 = 2^5 = 32（conv1×2 + maxpool×2 + layer2/3/4 各×2）
                            # 360×640 / 32 = 12×20 = 240 个空间位置
    return x
```

**调用链**：

```
self.backbone(img)                                          # modeling_act.py:926
  │
  │  self.backbone 是 IntermediateLayerGetter 实例           # _utils.py:13
  │  调用 IntermediateLayerGetter.forward(img)：
  │    for name, module in self.items():
  │        x = module(x)          ← 逐个执行下面的子模块
  │
  ├── x = Conv2d.forward(x)                                 # nn.Conv2d(3→64, 7×7, stride=2)
  ├── x = FrozenBatchNorm2d.forward(x)                       # misc.py:54，x * scale + bias
  ├── x = ReLU.forward(x)                                    # max(0, x)
  ├── x = MaxPool2d.forward(x)                               # 3×3 窗口取 max，stride=2
  ├── x = layer1.forward(x)                                  # nn.Sequential.forward()
  │     ├── x = BasicBlock.forward(x)                        # resnet.py:89
  │     │     ├── x' = Conv2d.forward(x)                     # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')
  │     │     ├── x' = ReLU.forward(x')
  │     │     ├── x' = Conv2d.forward(x')                    # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')
  │     │     ├── shortcut = x (直传，无 downsample)
  │     │     └── x = ReLU.forward(x' + shortcut)
  │     └── x = BasicBlock.forward(x)                        # 同上
  ├── x = layer2.forward(x)                                  # nn.Sequential.forward()
  │     ├── x = BasicBlock.forward(x)                        # 有 downsample
  │     │     ├── x' = Conv2d.forward(x)                     # 3×3 conv, stride=2
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')
  │     │     ├── x' = ReLU.forward(x')
  │     │     ├── x' = Conv2d.forward(x')                    # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')
  │     │     ├── shortcut = downsample(x)                    # conv1x1(stride=2) + BN
  │     │     └── x = ReLU.forward(x' + shortcut)
  │     └── x = BasicBlock.forward(x)                        # 无 downsample
  ├── x = layer3.forward(x)                                  # 同 layer2 结构
  └── x = layer4.forward(x)                                  # 同 layer2 结构
        ↑ 这里捕获输出，返回 {"feature_map": Tensor(1, 512, 12, 20)}
```


##### `self.backbone` 是什么

`IntermediateLayerGetter` 是 torchvision 提供的一个工具（`vision/torchvision/models/_utils.py:13`），它接收一个完整模型（这里是 ResNet18），把你指定的某些层的**中间输出**抓出来返回。这里指定 `{"layer4": "feature_map"}` 意思是"跑到 layer4 为止，把 layer4 的输出用键名 `feature_map` 返回"。所以调用时返回的是一个 `dict`，你要用 `["feature_map"]` 拿到真正的 Tensor。

原版 ResNet18 是给 ImageNet **分类**用的，最后还有 `avgpool`（全局池化到 (1,1)）+ `fc`（512→1000类）两步。ACT 只要**空间特征图**，不要分类结果，所以在 layer4 就截断。

##### 每一步详解

**第 1 步：conv1 — 入口卷积**

`conv1` 是 ResNet 给"第一个卷积层"起的变量名，实际类型是 `nn.Conv2d`（全程只用 Conv2d，没有 Conv1d/Conv3d）。

```
权重: W ∈ ℝ^{64 × 3 × 7 × 7}    ← 64 个 7×7×3 滤波器，无偏置（bias=False，后面 BN 吸收）
输入: x ∈ ℝ^{1 × 3 × 360 × 640}  ← 归一化后的 RGB 图像

stride=2, padding=3，对输出每个位置 (m,n)、每个输出通道 k：
    y[k, m, n] = ∑_{c=0}^{2} ∑_{i=0}^{6} ∑_{j=0}^{6} W[k,c,i,j] · x[c, 2m+i-3, 2n+j-3]

每个输出像素 = 3×7×7 = 147 次乘加
输出: y ∈ ℝ^{1 × 64 × 180 × 320}    ← 空间 ÷2（stride=2），通道 3→64（滤波器数量）

空间公式：H_out = ⌊(H_in + 2·padding - kernel) / stride⌋ + 1 = ⌊(360 + 6 - 7) / 2⌋ + 1 = 180
```

**第 2 步：bn1 — 冻结批归一化**

ACT 把 ResNet 所有 BN 替换成 `FrozenBatchNorm2d`（推理时 running_mean/var 不更新，等价固定仿射变换）。

$$
y_c = \gamma_c \cdot \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}} + \beta_c, \quad \varepsilon = 10^{-5}
$$

实际计算（合并成一次乘加，源码 `misc.py:54-63`）：

$$
\text{scale}_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \varepsilon}}, \quad \text{bias}_c = \beta_c - \mu_c \cdot \text{scale}_c, \quad y_c = x_c \cdot \text{scale}_c + \text{bias}_c
$$

形状不变 `(1, 64, 180, 320)`，逐通道拉回 ≈N(0,1)，数值稳定。推理时可 fuse 进前面 conv 加速。

**第 3 步：relu**

$$
y = \max(0, x)
$$

逐元素，负数清零引入非线性（否则多层卷积叠加等价一层线性变换）。形状不变。

**第 4 步：maxpool**

`nn.MaxPool2d(kernel_size=3, stride=2, padding=1)`：3×3 窗口取最大值，stride=2 空间÷2。

$$
y[m, n] = \max_{i,j \in \{0,1,2\}} x[2m+i, 2n+j]
$$

```
(1, 64, 180, 320) → (1, 64, 90, 160)
```

每个输出像素代表"这片 3×3 区域里特征响应最强的值"。

**第 5 步：layer1~layer4 — 4 个残差 stage**

每个 stage = `nn.Sequential(BasicBlock × 2)`，四层结构相同，区别只在通道数和 stride：

| stage   | 输入形状            | 输出形状            | stride | 通道变化  |
| ------- | ------------------- | ------------------- | ------ | --------- |
| layer1  | (1, 64, 90, 160)    | (1, 64, 90, 160)    | 1      | 64→64     |
| layer2  | (1, 64, 90, 160)    | (1, 128, 45, 80)    | 2      | 64→128    |
| layer3  | (1, 128, 45, 80)    | (1, 256, 23, 40)    | 2      | 128→256   |
| layer4  | (1, 256, 23, 40)    | (1, 512, 12, 20)    | 2      | 256→512   |

每个 **BasicBlock** 的计算（`resnet.py:59-105`）：

$$
\text{out} = \text{Conv2d}_{3\times3}(x) \to \text{BN} \to \text{ReLU} \to \text{Conv2d}_{3\times3} \to \text{BN}
$$

$$
\text{output} = \text{ReLU}(\text{out} + \text{shortcut}(x))
$$

其中 $\text{shortcut}(x)$：
- 如果输入输出形状相同（layer1 的两个 block、每个 stage 的第 2 个 block）：$\text{shortcut}(x) = x$（直传）
- 如果形状变了（每个 stage 的第 1 个 block，stride=2 或通道翻倍）：

$$
\text{shortcut}(x) = \text{BN}(\text{Conv2d}_{1\times1}(x))
$$

用 1×1 卷积对齐通道数，stride 对齐空间尺寸。

残差连接 $\text{out} + \text{shortcut}(x)$ 的意义：让网络学习的是"残差"（当前层比恒等映射好多少），而不是从头学习完整映射。缓解深层网络梯度消失，使训练更稳定。

**输出**：layer4 输出 `(1, 512, 12, 20)` 就是 `feature_map`。

总下采样因子 $= 2^5 = 32$（conv1 stride=2 + maxpool stride=2 + layer2/3/4 各 stride=2），所以 $360/32 \approx 12$，$640/32 = 20$。每个空间位置对应原图 $32 \times 32$ 区域，512 维 = 512 个 ImageNet 预训练检测器的响应强度。

**计算量**：ResNet18 对 360×640 约 12 GFLOPs，两路摄像头共约 **24 GFLOPs**，占整个推理 >50%。


#### 4.4.2 ② 生成 2D 正弦位置编码：给每个空间位置打"坐标"

```python
# __init__ 里定义（modeling_act.py:751-753）：
self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
    config.dim_model // 2  # 256，y 和 x 方向各编 256 维，拼起来正好 512
)
# ACTSinusoidalPositionEmbedding2d 定义在 modeling_act.py:1346，不是 torchvision，是 ACT 自己写的
# ACTSinusoidalPositionEmbedding2d 不带可学习参数，纯公式计算

# forward 里调用：
cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
# 输入 cam_features: (1, 512, 12, 20) 只用它的形状，不用它的值
# 输出 cam_pos_embed: (1, 512, 12, 20) 每个位置 (y,x) 的 512 维"坐标向量"
```

**做了什么**：给特征图的每个空间位置 (y, x) 生成一个 512 维的"坐标向量"，告诉 Transformer"这个 token 来自图像的第几行第几列"。

**为什么需要**：卷积输出的特征图展平之后，240 个像素 token 对 Transformer 来说是"无序"的——attention 只做加权求和，不关心顺序。位置编码把 (i, j) 的**坐标"写"进每个 token 里**，让 attention 能利用空间结构（比如"相邻像素应该更相关"）。


ACTSinusoidalPositionEmbedding2d.forward 的实现（[modeling_act.py:1374](../../lerobot/src/lerobot/policies/act/modeling_act.py#L1374)）：

```python
def forward(self, x: Tensor) -> Tensor:
    # x: (B, C, H, W) 输入特征图，这里只用它的形状 (H, W)

    not_mask = torch.ones_like(x[0, :1])                          # (1, H, W) 全1

    y_range = not_mask.cumsum(1, dtype=torch.float32)              # (1, H, W) 行号 1..H
    x_range = not_mask.cumsum(2, dtype=torch.float32)              # (1, H, W) 列号 1..W

    y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi   # 归一化到 (0, 2π]
    x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

    inverse_frequency = self._temperature ** (                     # (dimension,) 即 (256,)
        2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2)
        / self.dimension
    )

    x_range = x_range.unsqueeze(-1) / inverse_frequency           # (1, H, W, 256) 广播除法
    y_range = y_range.unsqueeze(-1) / inverse_frequency

    pos_embed_x = torch.stack(                                     # (1, H, W, 256) sin/cos 交错
        (x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1
    ).flatten(3)
    pos_embed_y = torch.stack(
        (y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1
    ).flatten(3)

    pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
    # cat: (1, H, W, 512)  permute: (1, 512, H, W)
    return pos_embed
```

全部是 PyTorch tensor 操作，没有可学习参数。以 `cam_features: (1, 512, 12, 20)` 为例，H=12，W=20：

##### 第1步：生成坐标网格（cumsum 累加）

```python
not_mask = torch.ones_like(x[0, :1])   # 取第0个batch的第0个通道 → (1, 12, 20) 全1
y_range = not_mask.cumsum(1)            # 沿 H 维做前缀和
x_range = not_mask.cumsum(2)            # 沿 W 维做前缀和
```

`cumsum(dim)` 对某维度做前缀和：`[1,1,1,1] → [1,2,3,4]`，全1数组累加就得到每个位置的行/列编号：

```
not_mask (1, 12, 20)：每个值都是 1

y_range (1, 12, 20)：  沿 H(dim=1) 累加，每行填入行号
  第0行: [[1, 1, 1, ..., 1]]   ← 20 个 1
  第1行: [[2, 2, 2, ..., 2]]
  ...
  第11行:[[12,12,12,...,12]]

x_range (1, 12, 20)：  沿 W(dim=2) 累加，每列填入列号
  每行都是: [[1, 2, 3, 4, ..., 20]]
```

##### 第2步：归一化到 [0, 2π]

```python
y_range = y_range / (y_range[:,-1:,:] + 1e-6) * 2π
x_range = x_range / (x_range[:,:,-1:] + 1e-6) * 2π
```

```
y_range[:,-1:,:] 取最后一行 → 值全是 12，形状 (1, 1, 20)，广播到 (1, 12, 20)
除法结果（再乘 2π）：
  第0行:  1/12 × 2π ≈ 0.524
  第5行:  6/12 × 2π ≈ 3.14
  第11行: 12/12 × 2π = 6.283

x_range[:,:,-1:] 取最后一列 → 值全是 20，形状 (1, 12, 1)，广播到 (1, 12, 20)
除法结果（再乘 2π）：
  第0列:  1/20 × 2π ≈ 0.314
  第9列:  10/20 × 2π ≈ 3.14
  第19列: 20/20 × 2π = 6.283
```

每个位置的坐标值现在落在 `(0, 2π]`，范围对称，方便 sin/cos 采样。

##### 第3步：生成 256 个频率值

```python
inverse_frequency = 10000 ** (2*(torch.arange(256) // 2) / 256)
# 形状：(256,)
```

```
arange(256) = [0,   1,   2,   3,   4,   5,   ..., 254, 255]
//2         = [0,   0,   1,   1,   2,   2,   ..., 127, 127]
×2/256      = [0,   0,  2/256, 2/256, 4/256, ..., 254/256, 254/256]

为什么是 10000^(2i/256) 这个公式？
这是原始 Transformer 论文（Attention is All You Need）的公式，用等比数列让128个频率从1到10000均匀分布（对数尺度）：
inverse_frequency:
  idx=0,1   → 10000^0      =    1.00  （最低频）
  idx=2,3   → 10000^0.0078 ≈   1.19
  idx=4,5   → 10000^0.0156 ≈   1.41
  ...
  idx=254,255→ 10000^0.992  ≈ 9441    （最高频）
```

相邻两个 idx 共享同一个频率值，一个后面取 sin，一个取 cos。

##### 第4步：每个坐标除以 256 个频率，得到 256 个"相位角"，再取 sin/cos

**核心思路**：一个坐标值（如 x=3.14）除以 256 个不同频率，得到 256 个相位角；每个相位角再用 sin 或 cos 映射到 [-1,1]，组成这个位置的"指纹"。

```python
x_range = x_range.unsqueeze(-1) / inverse_frequency
# (1,12,20,1) 广播除以 (256,) → (1,12,20,256)
# 每个坐标值 × 1个数，变成 256 个相位角
```

以位置 `(row=5, col=9)` 为例（x 归一化坐标 = 10/20×2π = **3.14**）：

```
用 256 把尺子（频率）来量这个坐标 3.14：

  相位角[0] = 3.14 ÷  1.00 = 3.14   （低频尺，整图量一圈）
  相位角[1] = 3.14 ÷  1.00 = 3.14
  相位角[2] = 3.14 ÷  1.19 = 2.64   （稍高频）
  相位角[3] = 3.14 ÷  1.19 = 2.64
  ...
  相位角[254]= 3.14 ÷ 9441 = 0.00033 （高频尺，几个像素量一圈）
  相位角[255]= 3.14 ÷ 9441 = 0.00033

得到 256 个相位角，形状 (256,)
```

然后把 256 个相位角交替取 sin/cos，变成 256 个 [-1,1] 之间的值：

```
相位角[0]=3.14 → sin(3.14) ≈  0.0016   ← 偶数位取 sin
相位角[1]=3.14 → cos(3.14) ≈ -1.00     ← 奇数位取 cos
相位角[2]=2.64 → sin(2.64) ≈  0.872
相位角[3]=2.64 → cos(2.64) ≈ -0.489
...
```

```python
pos_embed_x = torch.stack((x_range[...,0::2].sin(), x_range[...,1::2].cos()), dim=-1).flatten(3)
# x_range[...,0::2]: (1,12,20,128)   取偶数位相位角，全部 sin
# x_range[...,1::2]: (1,12,20,128)   取奇数位相位角，全部 cos
# stack(dim=-1):      (1,12,20,128,2) 把每对 (sinᵢ, cosᵢ) 并排
# flatten(3):         (1,12,20,256)   展开成 [sin₀,cos₀,sin₁,cos₁,...,sin₁₂₇,cos₁₂₇]
```

不同列的坐标不同，256个 sin/cos 值就不同——这就是每个位置唯一的"指纹"：

```
col=0  (x=0.314): [sin(0.314)=0.309, cos(0.314)=0.951, ...]
col=9  (x=3.14):  [sin(3.14) =0.002, cos(3.14) =-1.00, ...]
col=19 (x=6.28):  [sin(6.28) =0.000, cos(6.28) =1.00,  ...]
```

y 方向同理，pos_embed_y 也是 `(1,12,20,256)`。

##### 第5步：拼接 y 和 x 编码

```python
pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3)  # (1, 12, 20, 256+256) = (1, 12, 20, 512)
pos_embed = pos_embed.permute(0, 3, 1, 2)                  # → (1, 512, 12, 20)
```

```
每个位置 (i, j) 的 512 维向量排列：
  [0]   sin(y_coord/freq₀)
  [1]   cos(y_coord/freq₀)
  [2]   sin(y_coord/freq₁)
  [3]   cos(y_coord/freq₁)
  ...
  [254] sin(y_coord/freq₁₂₇)
  [255] cos(y_coord/freq₁₂₇)
  [256] sin(x_coord/freq₀)
  [257] cos(x_coord/freq₀)
  ...
  [511] cos(x_coord/freq₁₂₇)

permute(0,3,1,2) 把通道维从最后移到第二位，对齐 cam_features 的格式 (B,C,H,W)。
```

##### 输出：cam_pos_embed (1, 512, 12, 20)

```
cam_features:  (1, 512, 12, 20)  ← backbone 输出的视觉特征
cam_pos_embed: (1, 512, 12, 20)  ← 这一步生成的位置编码，形状与 cam_features 完全相同
                                    不依赖 cam_features 的值，只用它的形状 (h=12, w=20) 确定网格大小
```

每个位置 `(i, j)` 对应的 512 维向量内容：

```
前 256 维（y 方向，编码行号 i）：
  [sin(i/12×2π/freq₀), cos(i/12×2π/freq₀),
   sin(i/12×2π/freq₁), cos(i/12×2π/freq₁), ...]  共 256 个值

后 256 维（x 方向，编码列号 j）：
  [sin(j/20×2π/freq₀), cos(j/20×2π/freq₀), ...]  共 256 个值
```

不同 `(i, j)` 的 512 维向量各不相同，Transformer 通过这个向量就能区分每个 token 来自图像哪个位置。

**直观理解**：不同频率的 sin/cos 就像"不同精度的时钟指针"——低频指针扫过整张图很慢，用来区分图像的大区域；高频指针变化快，用来区分邻近像素。Transformer 看到每个 token 的 512 维位置编码，就能"解码"出它来自哪个 (y, x) 位置。

**计算量**：只有 sin/cos 运算，可忽略。

---

#### 4.4.3 ③ 1×1 卷积投影：换"表示空间"

```python
# __init__ 里定义（modeling_act.py:732-736）：
self.encoder_img_feat_input_proj = nn.Conv2d(
    backbone_model.fc.in_features,  # 512（ResNet18 layer4 的输出通道）
    config.dim_model,                # 512（Transformer 隐藏维度）
    kernel_size=1,                   # 1×1 卷积，不改变空间尺寸，只换通道
)
# 这里 W 的形状是 (512, 512, 1, 1)，b 是 (512,)，训练时学到

# forward 里调用：
cam_features = self.encoder_img_feat_input_proj(cam_features)
# (1, 512, 12, 20) → (1, 512, 12, 20)  形状不变，只是通道维度被重新线性组合
```


**1×1 Conv 是什么**：kernel_size=1 的卷积等价于"**对每个像素独立做一次 nn.Linear(512→512)**"。具体计算：

其实就是；*通道之间的矩阵乘*
```
对 12×20=240 个空间位置中的每一个 (i, j)：
  输入 x_ij = [c1, c2, ..., c512]     ← 这个位置的 512 维 ResNet 特征
  权重 W   = 512行×512列的矩阵         ← 训练学到的
  偏置 b   = 512个数                   ← 训练学到的

  y_ij = W @ x_ij + b                   ← (512,) 新的 512 维向量

  输出位置 (i,j) 的新特征 = y_ij

所有 240 个位置共享同一组 W 和 b（这是卷积的"权重共享"特性）。
```

等价于把 `(1, 512, 12, 20)` 先 reshape 成 `(240, 512)`，对每行做 `Linear(512→512)`，再 reshape 回来。

**以 cam_features (1, 512, 12, 20) 为例**：

```
输入：cam_features (1, 512, 12, 20)
       ↓ Conv2d(512→512, kernel=1)，权重 W:(512,512,1,1)，偏置 b:(512,)

对每个空间位置 (i, j)（共 12×20=240 个）：
  input_ij  = cam_features[0, :, i, j]  → (512,) 的向量
  output_ij = W_reshaped @ input_ij + b → (512,) 的向量（W_reshaped 是 (512,512)）

输出：cam_features (1, 512, 12, 20)  ← 形状不变，但每个位置的 512 维被重新线性组合过
```

**为什么 512 → 512 还需要投影**：虽然维度没变，但 ResNet18.layer4 的输出是为 **ImageNet 分类**学到的表示空间，和 Transformer 需要的"attention 友好"空间不一样。这个 1×1 Conv 就是一个"**适配器**"——它的权重是 ACT 训练时和 Transformer 一起学出来的，负责把 ResNet 的表示重新编排成 Transformer 能用的形式。

**计算量**：每个位置一次 512×512 矩阵乘 = $2·512^2$ FLOPs，240 个位置 × 2 个摄像头 ≈ **126 MFLOPs**。

---

#### 4.4.4 ④ 展平为 token 序列

```python
cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
# (1, 512, 12, 20) → (240, 1, 512)
```

**做了什么**：把二维的空间网格"摊平"成一个长度 240 的序列，每个元素是一个 512 维的 token。

**`einops` 记号解读**：

- `b c h w → (h w) b c`
- `(h w)`：把 12×20=240 个空间位置**排成一列**，放到最前面（即"序列维"）
- `b`：batch 维移到第二个位置
- `c`：通道维移到最后，变成"每个 token 的 512 维特征"


**以 cam_features (1, 512, 12, 20) 为例**：

```
输入：(1, 512, 12, 20)

rearrange "b c h w -> (h w) b c"：
  h=12, w=20 合并 → (h w)=240，作为序列长度放最前面
  b=1 移到第二维
  c=512 移到最后

输出：(240, 1, 512)

直观：原来是一张"12行×20列的图"，现在变成"240个token排成一列"
  第0个token = 原图位置 (row=0, col=0) 的 512 维特征
  第1个token = 原图位置 (row=0, col=1) 的 512 维特征
  ...
  第19个token = 原图位置 (row=0, col=19) 的 512 维特征
  第20个token = 原图位置 (row=1, col=0) 的 512 维特征
  ...
  第239个token = 原图位置 (row=11, col=19) 的 512 维特征

cam_pos_embed 做同样的 rearrange：(1, 512, 12, 20) → (240, 1, 512)
  第 k 个位置编码 = 第 k 个 token 对应的 (i, j) 坐标的 512 维 sin/cos 向量
```

**为什么要展平**：Transformer 处理的是序列，attention 不关心"空间位置是第几行第几列"。所以把 12×20 的 2D 网格变成长度 240 的 1D 序列。失去的空间信息由前面 §4.4.2 的 2D 位置编码补回来（位置编码里已经写好了 (i, j) 坐标）。

---

### 4.5 最终堆叠


```python
encoder_in_tokens   = torch.stack(encoder_in_tokens, axis=0)     
encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)  
```

**关键理解**：`encoder_in_tokens` 和 `encoder_in_pos_embed` 是**两个并行的 list**——一个装"特征内容"，一个装"位置编码"。两个 list 的元素**一一对应、长度相等**，最终 stack 出来的两个张量形状完全一致，都是 `(482, 1, 512)`。

**以双摄像头（cam0 + cam1）为例，完整堆叠过程**：

```
                    encoder_in_tokens          encoder_in_pos_embed
                    ─────────────────          ────────────────────
  [0]   latent token   (1,512)                  latent_pos        (1,512)  ← 1D 可学习
  [1]   state  token   (1,512)                  state_pos         (1,512)  ← 1D 可学习

  cam0 (handeye 摄像头)：
    backbone → cam_features (1,512,12,20)       pos_embed 模块 → cam_pos_embed (1,512,12,20)
    1×1 Conv 投影                                ↓ 不需要投影
    rearrange → (240, 1, 512)                   rearrange → (240, 1, 512)
    list(...) = 240 个 (1,512)                   list(...) = 240 个 (1,512)
    extend → tokens[2:242]                       extend → pos_embed[2:242]

  cam1 (fixed 摄像头)：
    同上 → tokens[242:482]                       同上 → pos_embed[242:482]

最终 list 长度都是 1+1+240+240 = 482

torch.stack(encoder_in_tokens,    axis=0) → encoder_in_tokens:    (482, 1, 512)
torch.stack(encoder_in_pos_embed, axis=0) → encoder_in_pos_embed: (482, 1, 512)
                                            ↑↑↑ 两个张量形状完全一致
```

**做了什么**：分别构造**两路并行的 list**——一路装 token 特征，一路装对应的位置编码。每路摄像头各贡献 240 对 (feature, pos)，加上 latent 和 state 的 (1, 1) 对，最终 stack 成两个形状为 `(482, 1, 512)` 的张量，元素一一对应。

**token / 位置编码对齐表**：

| 索引        | token 内容                      | 位置编码来源                                                                                                                                    |
| ----------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `[0]`       | latent token                    | 可学习 1D embedding（`encoder_1d_feature_pos_embed[0]`，[modeling_act.py:744](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L744)） |
| `[1]`       | robot state token               | 可学习 1D embedding（`encoder_1d_feature_pos_embed[1]`）                                                                                        |
| `[2:242]`   | cam0（handeye）240 个像素 token | 2D 正弦编码（§4.4.2）                                                                                                                           |
| `[242:482]` | cam1（fixed）240 个像素 token   | 2D 正弦编码（§4.4.2）                                                                                                                           |

**位置编码只加到 Q 和 K**：Transformer 里位置编码**不是加到 value 上的**，而是在每层 attention 计算前单独加到 Q 和 K 上（下一节 §5.1 详解）。这是 DETR 的标准做法：Q·K 是"算相关性"的地方，加位置编码让"相邻位置更相关"可学习；V 是"内容本身"，不该被位置信息污染。

---

## 5. 步骤 3：Transformer Encoder（4 层）

```py
encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
```

**输入输出形状**：

| 参数                               | 形状                | 内容                                                     |
| ---------------------------------- | ------------------- | -------------------------------------------------------- |
| `encoder_in_tokens` (x)            | `(482, 1, 512)`     | 堆叠后的 token：1 latent + 1 state + 240 cam0 + 240 cam1 |
| `encoder_in_pos_embed` (pos_embed) | `(482, 1, 512)`     | 对应的位置编码（两种来源：1D 可学习 + 2D 正弦）          |
| **`encoder_out`**                  | **`(482, 1, 512)`** | **4 层 Encoder 后的输出，形状不变，内容融合全局上下文**  |

### 5.0 这 4 层到底在做什么

Encoder 的输入是上一步构造的 482 个 token，每个 512 维。Encoder 把这 482 个 token 通过 4 层相同结构的 Transformer block "翻来覆去"做两件事：

1. **Self-Attention**：让每个 token 去"看"所有其他 token，按相关性加权平均，融合信息
2. **FFN（前馈网络）**：对每个 token 独立做一次两层 MLP，增加非线性表达

一层结束后每个 token 都"知道"了别人在说什么，4 层叠加之后每个 token 都是融合了全局上下文的特征。

**ACT 用 post-norm**：残差加完后再做 LayerNorm（与原始 Transformer 2017 论文一致，非后来流行的 pre-norm），`pre_norm=False`。

### 5.1 单层完整代码

```python
class ACTEncoderLayer(nn.Module):
    def forward(
        self,
        x, # 是 encoder_in_tokens
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None, # 没输入
    ) -> Tensor:
        """单层编码器前向传播。

        Args:
            x: (seq_len, B, dim_model) 当前层输入，
            pos_embed: (seq_len, 1, dim_model) 位置编码（加到 Q 和 K）
            key_padding_mask: (B, seq_len) padding 掩码

        Returns:
            x: (seq_len, B, dim_model) 编码后的特征
        """
        skip = x  # 残差连接

        if self.pre_norm:
            x = self.norm1(x)

        # Q = K = x + pos_embed（位置编码加到 Query 和 Key）
        # V = x（不加位置编码，这是标准做法）
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # FeedForward: Linear → Activation → Dropout → Linear
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)

        return x
```

**以输入 (482, 1, 512) 为例，一层完整数据流**：

```py
输入 x:        (482, 1, 512)  ← 482个token，每个512维
pos_embed:     (482, 1, 512)  ← 对应位置编码（每层都用原始的那份）

── 子层①：Self-Attention ──
skip = x                              (482, 1, 512)  原始输入的备份
q = k = x + pos_embed                (482, 1, 512)  逐元素相加
v = x                                 (482, 1, 512)  V 不加位置编码
attn_out = self_attn(q, k, v)[0]     (482, 1, 512)  Self-Attention 输出
dropout_out = dropout(attn_out)      (482, 1, 512)  Dropout（随机丢弃）
residual = skip + dropout_out        (482, 1, 512)  残差连接：逐元素相加
x = norm1(residual)                  (482, 1, 512)  LayerNorm

── 子层②：FFN ──
skip = x                              (482, 1, 512)  输入备份
linear1_out = linear1(x)              (482, 1, 3200) 512→3200 扩维
relu_out = relu(linear1_out)          (482, 1, 3200) 逐元素激活函数
dropout_out = dropout(relu_out)       (482, 1, 3200) Dropout
ffn_out = linear2(dropout_out)        (482, 1, 512)  3200→512 压回
residual = skip + ffn_out             (482, 1, 512)  残差连接：逐元素相加
x = norm2(residual)                   (482, 1, 512)  LayerNorm

输出 x:        (482, 1, 512)  ← 形状和输入完全一样，但每个token都"看过"其他所有token
```

第 $\ell$ 层的输入是 $X^{(\ell-1)} \in \mathbb{R}^{482 \times 1 \times 512}$，输出 $X^{(\ell)}$ 同形状。

---

### 5.2 子层①：多头自注意力（Self-Attention）

```python
skip = x                                    # (482, 1, 512) 保存原始输入
q = k = x + pos_embed                       # (482, 1, 512) Q、K 加位置编码
v = x                                       # (482, 1, 512) V 不加位置编码
attn_out = self.self_attn(q, k, value=v)[0]  # (482, 1, 512) Self-Attention 计算
x = skip + self.dropout1(attn_out)          # (482, 1, 512) 残差连接 + Dropout
x = self.norm1(x)                           # (482, 1, 512) LayerNorm 归一化
```

**三步具体计算**：

**步骤①：Self-Attention 计算**（实现：[pytorch/torch/nn/modules/activation.py:L1091](../../pytorch/torch/nn/modules/activation.py#L1091) `class MultiheadAttention`，底层调用 [pytorch/torch/nn/functional.py:L6228](../../pytorch/torch/nn/functional.py#L6228) `multi_head_attention_forward`）
```py
输入：
  q = (482, 1, 512)  ← x + pos_embed（位置感知的查询）
  k = (482, 1, 512)  ← x + pos_embed（位置感知的键）
  v = (482, 1, 512)  ← x（原始内容）

计算流程：
  1) Q·K^T / √d
     (482, 1, 512) @ (512, 1, 482) = (482, 1, 482)
     ÷ √d，其中 d = 512 / 8 heads = 64 → √64 = 8
     ↓ 含义：482 个 token 两两之间的相似度 → 482×482 相似度矩阵
     ↓ √d 的作用：缩放点积，防止值太大导致梯度消失
     
  2) softmax(...)
     (482, 1, 482) → (482, 1, 482)
     ↓ 含义：每个 token，其他 token 的"重要度"（权重和=1）
     
  3) 权重 · V
     (482, 1, 482) @ (482, 1, 512) = (482, 1, 512)
     ↓ 含义：每个 token = 所有 token 的加权平均（权重来自相似度）

输出：attn_out = (482, 1, 512) ← 融合全局信息的新特征
```

**步骤②：残差连接 + Dropout**
```
skip = (482, 1, 512)              ← 原始输入（之前保存）
attn_out = (482, 1, 512)          ← Self-Attention 输出

dropout(attn_out)：
  形状仍是 (482, 1, 512)
  但随机将其中 ~10% 的元素置为 0（其余元素 ÷ 0.9，保证均值不变）
  → 仍是 (482, 1, 512)
  
x = skip + dropout(attn_out)
  = (482, 1, 512) + (482, 1, 512) = (482, 1, 512)
  逐元素相加，形状完全一样
  
含义：保留原始特征，加上新的 attention 特征（防止梯度消失，Dropout 防止过拟合）
```

**步骤③：LayerNorm 归一化**
```py
x = self.norm1(x)       # nn.LayerNorm(512)
```

输入：x = (482, 1, 512)

**计算（对每个 token 的 512 维独立做）**：
```py
第 i 个 token 的 512 维向量：x[i] = [v_1, v_2, ..., v_512]

step 1: 计算这个 token 的统计量
  mean_i = (v_1 + v_2 + ... + v_512) / 512  → 一个标量
  std_i = √[ Σ(v_j - mean_i)² / 512 ]      → 一个标量
  
step 2: 归一化（均值变 0、方差变 1）
  x_norm[i] = (x[i] - mean_i) / (std_i + ε)
  现在 x_norm[i] 的 512 个值：均值=0，方差=1
  
step 3: 缩放和平移（模型可学习）
  γ = (512,) 形状的参数，对每个维度的缩放
  β = (512,) 形状的参数，对每个维度的偏移
  x[i] = γ * x_norm[i] + β  ← 模型自己决定最终形式
```

**直观例子**：
```
某个 token 的 512 维值：[3.5, -2.1, 0.8, ..., 5.2]  (分布散乱)
                     均值=1.2，标准差=3.4
                     ↓ LayerNorm 归一化
归一化后：[-0.76, -0.97, -0.12, ..., 1.21]  (均值=0，标准差=1，分布规整)
                     ↓ 乘以 γ、加 β
最终输出：[0.2, -0.5, 0.3, ..., 0.8]  (被调整到模型需要的形式)
```

输出：x = (482, 1, 512)，形状不变

**LayerNorm 的作用**：
1. **稳定训练**：特征分布一致，梯度不会太大也不会太小
2. **加速收敛**：避免梯度爆炸/消失
3. **可学习性**：γ、β 让模型自己决定最终的缩放/偏移（比固定的标准化更灵活）


**以 (482, 1, 512) 输入为例**：

q = k = x + pos_embed  →  (482, 1, 512)
v = x                  →  (482, 1, 512)

MultiheadAttention 内部：
  W_Q, W_K, W_V 各是 (512, 512) 的投影矩阵
  对 q/k/v 各做线性投影 → 仍是 (482, 1, 512)
  然后按 8 个头切分 → 每头 (482, 1, 64)，8头并行计算

attention 结果拼合 + 输出投影 W_O：
  concat 8头 → (482, 1, 512)
  × W_O(512,512)             → (482, 1, 512)

#### 本子层 FLOPs（单层，B=1）

| 子步骤       | 计算                 | FLOPs             |
| ------------ | -------------------- | ----------------- |
| Q/K/V 投影   | `3 × 2 × 482 × 512²` | ~758 M            |
| Q·K^T        | `2 × 8 × 482² × 64`  | ~238 M            |
| softmax · V  | `2 × 8 × 482² × 64`  | ~238 M            |
| 输出投影 W_O | `2 × 482 × 512²`     | ~253 M            |
| **小计**     |                      | **≈ 1.49 GFLOPs** |

---

### 5.3 子层②：前馈网络（FFN）

```python
skip = x                                    # 保存原始输入
x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # FFN 计算
x = skip + self.dropout2(x)           # 残差连接 + Dropout
x = self.norm2(x)                           # LayerNorm
```

**步骤①：FFN（两层 MLP）**

| 步骤      | 计算                             | 形状变化                       |
| --------- | -------------------------------- | ------------------------------ |
| `linear1` | 512 → 3200 扩维                  | (482, 1, 512) → (482, 1, 3200) |
| `ReLU`    | 激活函数：x > 0 保留，x ≤ 0 变 0 | 形状不变 (482, 1, 3200)        |
| `dropout` | 随机置零 ~10%                    | 形状不变 (482, 1, 3200)        |
| `linear2` | 3200 → 512 压回                  | (482, 1, 3200) → (482, 1, 512) |

**具体矩阵乘法**：

**linear1 的扩维计算**（512 → 3200）

权重矩阵：`W_1.shape = (3200, 512)`（可学习参数）

单个 token 的计算：
- 输入一个 token：`x[i] = [v_1, v_2, ..., v_512]`（512 个数）
- 矩阵乘法：`y[i] = W_1 @ x[i]` = `(3200, 512) @ (512,)` = `(3200,)`
- 输出：一个 3200 维的向量（每一维都是输入 512 个值的加权和）

全部 482 个 token：
```
(482, 1, 512) @ W_1^T = (482, 1, 3200)
```

**linear2 的压回计算**（3200 → 512）

权重矩阵：`W_2.shape = (512, 3200)`（可学习参数）

单个 token 的计算：
- 输入（经过 ReLU 和 dropout）：`y[i] = [u_1, u_2, ..., u_3200]`（3200 个数）
- 矩阵乘法：`out[i] = W_2 @ y[i]` = `(512, 3200) @ (3200,)` = `(512,)`
- 输出：回到 512 维向量（每一维都是输入 3200 个值的加权和）

全部 482 个 token：
```
(482, 1, 3200) @ W_2^T = (482, 1, 512) ✓ 回到原维度
```

**为什么要扩维再压回**：

就计算来说，是的——**就是两次矩阵乘法**。但：

- **W_1、W_2 都是可学习参数**（通过反向传播优化）
- 如果没有 ReLU，就只是 `(x @ W_1^T) @ W_2^T = x @ (W_2 @ W_1)^T`，仍然是**线性变换**，可以一次矩阵乘法搞定
- **ReLU 引入非线性**：`W_2 @ ReLU(W_1 @ x)` 才能学到**任意非线性函数**
- 扩到 3200 维的原因：更多的中间维度 = 更强的表达能力（3200 > 512，参数更多）

**ReLU 详解（Rectified Linear Unit）**

定义：`ReLU(x) = max(0, x)`

计算示例：
```
输入：y = [0.5, -1.2, 3.4, -0.1, 2.1, -0.8, ...]  (3200 个数)

ReLU(y) = [0.5,   0,  3.4,   0,  2.1,   0,  ...]
           ↑     ↑    ↑     ↑    ↑     ↑
        正→保留 负→0 正→保留 负→0 正→保留 负→0
```

- **正数**：原样保留
- **负数**：全部置为 0
- **形状**：(482, 1, 3200) 保持不变

**为什么用 ReLU**：
- **引入非线性**：神经网络没有激活函数就只能做线性变换，有了非线性才能学复杂函数
- **计算简单**：就是 max(0, x)，速度快
- **梯度好**：导数是 0 或 1，不会梯度消失/爆炸（相比 sigmoid、tanh）
- **生物学启发**：类似神经元"激发"的机制——强的信号通过，弱的信号被抑制

**简单类比**：
- 没有 ReLU：`y = 2x`（线性，图像是直线）
- 有 ReLU：`y = W_2 @ ReLU(W_1 @ x)`（非线性，W_1、W_2 可学习，图像可以是任意折线）

**步骤②：残差连接 + Dropout**

```python
x = skip + self.dropout2(x)
```

**计算过程**：

skip 是之前保存的原始输入，ffn_out 是 FFN 的输出，两者形状都是 (482, 1, 512)

```
skip:        (482, 1, 512)  ← Self-Attention 子层之后的输入
ffn_out:     (482, 1, 512)  ← FFN 的输出（经过 linear2）

dropout(ffn_out)：随机置零 ~10%，形状仍是 (482, 1, 512)

x = skip + dropout(ffn_out)
  = (482, 1, 512) + (482, 1, 512)  ← 逐元素相加
  = (482, 1, 512)
```

**含义**：保留原始特征（skip），加上新计算的 FFN 特征，融合两者信息。防止梯度消失。

**步骤③：LayerNorm 归一化**

```python
x = self.norm2(x)
```

**此时 x 是什么**：

x 是上一步残差连接的结果：`x = skip + dropout2(ffn_out)`，形状 `(482, 1, 512)`。


所以 x 是一个 `(482, 1, 512)` 的矩阵，有 482 行（token），每行 512 个数。经过 FFN 的非线性变换后，每个 token 的 512 维值分布变得不规则——有的维度偏大，有的偏小，有的偏正，有的偏负。LayerNorm 的目的就是把每个 token 的 512 维值重新拉回到一个标准分布。

**具体计算**：

对 482 个 token **逐个独立**做同样的三步运算（token 之间互不影响，每个 token 只看自己的 512 个值）：

```
对第 i 个 token 的 512 维向量 x[i] = [v_1, v_2, ..., v_512]：

1) 算统计量：mean = mean(v_1..v_512)，std = std(v_1..v_512)     ← 标量
2) 归一化：  x_norm = (x[i] - mean) / (std + ε)                 ← 512 维，均值=0，方差=1
3) 缩放偏移：output = γ * x_norm + β                            ← γ、β 是 (512,) 可学习参数
```

就是逐元素运算：每个值先**减均值除标准差，再乘 γ 加 β**，482×512 个值各算各的。

**为什么需要**：FFN 的两次矩阵乘法 + ReLU 让值范围不可预测，直接传给下一层会导致梯度爆炸/消失。LayerNorm 把每个 token 的特征拉回一致的分布。

输出：(482, 1, 512)，形状不变


#### FLOPs（单层，B=1）

$$
2 \cdot 482 \cdot (512 \cdot 3200 + 3200 \cdot 512) \approx 3.16\,\text{GFLOPs}
$$

FFN 是单层里计算量最大的部分（比 attention 还贵 2 倍）。

---

### 5.4 4 层叠加

```python
# ACTEncoder.forward (modeling_act.py:1050)
for layer in self.layers:        # 4 层
    x = layer(x, pos_embed=pos_embed)  # 注意 pos_embed 每层都传进去重新加
encoder_out = x                  # (482, 1, 512)
```

**以 4 层叠加为例，形状始终不变**：

```
层0输入:  encoder_in_tokens  (482, 1, 512)
          pos_embed          (482, 1, 512)  ← 固定不变，每层都用这份
  ↓ layer[0]
层1输入:  x (482, 1, 512)  ← 每个token看过了其他所有token
  ↓ layer[1]
层2输入:  x (482, 1, 512)  ← 信息进一步融合
  ↓ layer[2]
层3输入:  x (482, 1, 512)
  ↓ layer[3]
encoder_out: (482, 1, 512)  ← 最终输出，4轮全局信息融合后的结果
```

**关键细节**：位置编码 `pos_embed` 每一层都用**原始的那份**，不随 `x` 一起更新。这是因为每一层都要在 Q/K 处重新加一次位置编码，如果用更新后的 x，位置信息会被稀释或丢失。

**总计算量**：

| 每层                 | FLOPs       |
| -------------------- | ----------- |
| 子层① Self-Attention | ~1.49 G     |
| 子层② FFN            | ~3.16 G     |
| **单层合计**         | **~4.65 G** |
| **4 层合计**         | **~18.6 G** |

### 5.5 Encoder 最终输出

$$
H_{enc} = \text{Enc}(X_{enc}^{(0)}; P_{enc}) \in \mathbb{R}^{482 \times 1 \times 512}
$$

**含义**：482 个"上下文特征"，每个都已经"看过"其他所有 token——
- `H_enc[0]`（latent token 位）里融合了视觉和关节状态
- `H_enc[1]`（state token 位）里融合了视觉和 latent 风格
- `H_enc[2:482]`（image token 位）里融合了其他像素、机器人状态和 latent

---

## 6. 步骤 4：Transformer Decoder（DETR 风格，1 层）

### 6.0 进入 Decoder 前：构造全零输入

```python
# decoder_in: (chunk_size, B, dim_model) = (100, 1, 512)，全零占位
# dtype/device 跟 encoder_in_pos_embed 保持一致
decoder_in = torch.zeros(
    (self.config.chunk_size, batch_size, self.config.dim_model),
    dtype=encoder_in_pos_embed.dtype, # torch.float32
    device=encoder_in_pos_embed.device,# cpu
)

# decoder_out: (chunk_size, B, dim_model) = (100, 1, 512)
decoder_out = self.decoder(
    decoder_in,                                      # Q 的"内容"部分：全零
    encoder_out,                                     # K/V：Encoder 的 482 个上下文特征
    encoder_pos_embed=encoder_in_pos_embed,          # Encoder 位置编码（正弦），加到 K
    decoder_pos_embed=self.decoder_pos_embed.weight  # Decoder 位置编码（可学习），加到 Q
        .unsqueeze(1),                               # (100, dim_model) → (100, 1, dim_model)
)

# 转置: (chunk_size, B, dim_model) → (B, chunk_size, dim_model) = (1, 100, 512)
decoder_out = decoder_out.transpose(0, 1)
```

**为什么 `decoder_in` 全零？**

| 对比项            | 标准 seq2seq Decoder | DETR / ACT Decoder         |
| ----------------- | -------------------- | -------------------------- |
| 输入内容          | 上一时刻输出 token   | **全零占位**               |
| 位置/语义信息来源 | token 本身           | 可学习 `decoder_pos_embed` |
| 生成方式          | 串行（t 依赖 t-1）   | **并行**（100 步同时生成） |

`decoder_pos_embed` 是 `nn.Embedding(chunk_size, dim_model)`（完全可学习），训练后每个位置编码携带了"第 t 步动作应该关注哪类特征"的先验。全零向量只是让这个位置编码"干净"地注入，不被初始内容干扰。



### 6.1 单层完整代码（3 个子层）

```python
class ACTDecoderLayer(nn.Module):

    def __init__(self, config: ACTConfig):
        super().__init__()
        # 自注意力和交叉注意力
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )

        # FeedForward 网络
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(
        self, tensor: Tensor, pos_embed: Tensor | None
    ) -> Tensor:
        """可选地添加位置编码。"""
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,              # (chunk_size, B, dim_model) 解码器输入
        encoder_out: Tensor,    # (enc_seq_len, B, dim_model) 编码器输出
        decoder_pos_embed: Tensor | None = None,  # 解码器位置编码
        encoder_pos_embed: Tensor | None = None,  # 编码器位置编码
    ) -> Tensor:
        """单层解码器前向传播。

        Args:
            x: 解码器当前输入
            encoder_out: 编码器上下文
            decoder_pos_embed: 解码器位置编码
            encoder_pos_embed: 编码器位置编码

        Returns:
            (chunk_size, B, dim_model) 解码后的特征
        """
        # ── 子层 1：自注意力 ───────────────────────────────────────────
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # ── 子层 2：交叉注意力 ─────────────────────────────────────────
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        # ── 子层 3：前馈网络 ───────────────────────────────────────────
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)

        return x
```

---

### 6.2 Decoder 输入初始化

```python
decoder_in = torch.zeros((T=100, B=1, D=512))                       # 全零占位
decoder_pos_embed = self.decoder_pos_embed.weight.unsqueeze(1)      # (100, 1, 512) 可学习
```

| 张量                | 形状            | 内容                            |
| ------------------- | --------------- | ------------------------------- |
| `decoder_in` (x)    | `(100, 1, 512)` | 全零                            |
| `decoder_pos_embed` | `(100, 1, 512)` | 100 个可学习的 "object queries" |

**为什么输入是全零**：所有"问什么"的信息由 `decoder_pos_embed` 携带。每个 query $q_t$ 是训练出来的"未来第 t 步的提问模板"——比如 $q_0$ 可能代表"抓取前一瞬间要看哪儿"，$q_{99}$ 可能代表"抓取完成后要看哪儿"。这 100 个 query 向量在训练时学出来，推理时固定。

---

### 6.3 子层①：Query 自注意力（推理时近似无效）

```python
# decoder_pos_embed：nn.Embedding(chunk_size=100, dim_model=512).weight.unsqueeze(1)
#   → (100, 512) 可学习矩阵，unsqueeze(1) → (100, 1, 512)
#   含义：100 个完全可学习的"提问模板"向量，训练时梯度更新，推理时固定

# x 初始为全零占位
skip = x                # (100, 1, 512)，全 0
q = k = x + decoder_pos_embed
                        # (100, 1, 512) + (100, 1, 512) = (100, 1, 512)
                        # 全零 + pos_embed，实际上 q = k = decoder_pos_embed
                        # decoder_pos_embed是纯可学习矩阵
x = self.self_attn(q, k, value=x)[0]
                        # q=(100,1,512), k=(100,1,512), v=(100,1,512) 全零
                        # 输出 (100, 1, 512)；因 V=0，输出也 ≈ 0
x = skip + self.dropout1(x)
                        # (100,1,512) + dropout(≈0) = (100,1,512) 仍 ≈ 0
x = self.norm1(x)       # LayerNorm((100,1,512)) → 输出 β（可学习偏置）
skip = x                # (100, 1, 512)，保存给交叉注意力的残差用
```

**Self-Attention 内部计算**：

```
Q = K = decoder_pos_embed        # (100, 1, 512)  ← 100 个可学习的"提问向量"
V = x = 0                        # (100, 1, 512)  ← 全零

① 线性投影（每个头 64 维，共 8 头）：
   Q → Q' = Q · W_Q  (100, 1, 512) → (100, 1, 512)
   K → K' = K · W_K  (100, 1, 512) → (100, 1, 512)
   V → V' = V · W_V  (100, 1, 512) → (100, 1, 512)  ← 全零 × W_V = 全零

② 注意力分数：
   score = Q' · K'^T / √64       # (100, 1, 100)  ← query t 对其他 99 个 query 的相似度
   weight = softmax(score)        # (100, 1, 100)  ← 每行加和=1（但内容无意义，因为 V=0）

③ 加权求和：
   output = weight · V'           # (100, 1, 100) × (100, 1, 512) → (100, 1, 512)
                                   # V'=0 → output = 0，无论 weight 怎么分布都白算

④ 残差：
   x = skip + dropout(output)     # 0 + 0 = 0
```

**做了什么**：理论上让 100 个 query 互相协调，但**这一子层在 ACT 里是空跑**。

**为什么空跑**：V = x = 全零 → `softmax(QK^T) · 0 = 0` → 残差后仍 0 → `norm1(0) = β`（LayerNorm 的可学习偏置，一个常数）。输出跟输入无关。

**为什么 ACT 没人修**：原论文代码默认 `n_decoder_layers=7`，但存在 bug 导致只有第 1 层真正被使用，后面 6 层被丢弃。HuggingFace lerobot 为了**如实复现原论文行为**，直接把默认改成 `n_decoder_layers=1`（见 [configuration_act.py:166-168](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L166-L168) 注释）。所以在只有 1 层的情况下，self-attention 永远吃到全零的 V，这个子层实际上是**装饰**——信息流全靠下面的 cross-attention。

---

### 6.4 子层②：交叉注意力（Cross-Attention）—— 真正做事的一步

```python
x = self.multihead_attn(
    query = x + decoder_pos_embed,          # (100, 1, 512) ← 来自 decoder（100 个 queries）
    key   = encoder_out + encoder_pos_embed,# (482, 1, 512) ← 来自 encoder
    value = encoder_out,                    # (482, 1, 512) ← 来自 encoder，不加 pos
)[0]
x = skip + dropout(x)
x = self.norm2(x)
skip = x
```

**Cross-Attention 内部计算**（实现：[pytorch/torch/nn/modules/activation.py:L1091](../../pytorch/torch/nn/modules/activation.py#L1091) `class MultiheadAttention`，底层调用 [pytorch/torch/nn/functional.py:L6228](../../pytorch/torch/nn/functional.py#L6228) `multi_head_attention_forward`）：

```
输入三个张量：
decoder_pos_embed是完全可学习的

  Q_in = x + decoder_pos_embed       # (100, 1, 512)  来自 decoder，x 此时=β，加位置编码
  K_in = encoder_out + encoder_pos_embed
                                     # (482, 1, 512)  来自 encoder，加位置编码
  V_in = encoder_out                 # (482, 1, 512)  来自 encoder，不加位置编码 ★关键：非零

① 线性投影（三组独立权重 W_Q/W_K/W_V，形状都是 (512, 512)）：
   Q = Q_in · W_Q                    # (100, 1, 512)
   K = K_in · W_K                    # (482, 1, 512)
   V = V_in · W_V                    # (482, 1, 512)

② 分 8 个头（每头 64 维，8 头并行计算）：
   Q → (8, 100, 64)  per head
   K → (8, 482, 64)  per head
   V → (8, 482, 64)  per head

③ 注意力分数（每个头独立算）：
   score = Q · K^T / √64
         = (8, 100, 64) × (8, 64, 482) = (8, 100, 482)
   含义：第 t 个 query 对 encoder 482 个位置的原始相似度

④ Softmax（本质就是"归一化"，只是对 Tensor 的每个切片逐一套用同一个公式）：

   输入：score (8, 100, 482)   —— 每个元素是一个实数，代表"相似度原始分"
   操作：对最后一维 dim=-1 做 softmax

   ─── softmax 本身是什么 ───
   softmax 只做一件事：把一个向量归一化成"和为 1 的概率分布"。
   给定任意 n 个实数 x_0, x_1, ..., x_{n-1}：
       softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
   结果每个值落在 (0, 1)，总和 = 1。就是一个纯数学归一化公式。

   ─── 在 Tensor 上怎么套用 ───
   score 是 3D Tensor (8, 100, 482)，PyTorch 把它看成 "8×100 = 800 个长度为 482 的向量"，
   对每一个这样的向量独立调一次 softmax，互不干扰，结果再拼回成同样形状的 Tensor。
   也就是：固定 (h, t)，取出 score[h, t, :] 这 482 个数，算 softmax，写回 weight[h, t, :]。

   写成下标形式（h, t, i 只是 Tensor 里某个元素的坐标，i 遍历 482）：
       weight[h, t, i] = exp(score[h, t, i]) / Σ_{j=0..481} exp(score[h, t, j])
                         ↑                      ↑
                         第 i 个位置的指数        同一行 482 个位置指数求和（归一化分母）

   ─── 直观理解 ───
   · 把 (8, 100, 482) 想象成一个"表格"：8 张表、每张表 100 行、每行 482 个数。
   · softmax 就是对"每一行"独立做归一化，让那 482 个数变成概率（加和=1）。
   · h, t 只是"我在哪张表、哪一行"的坐标，不参与计算；真正被归一化的是 i 这一维。

   性质：
     · weight[h, t, i] ∈ (0, 1)
     · Σ_i weight[h, t, i] = 1   （每行 482 个数加和恰好为 1，构成概率分布）

   输出：weight (8, 100, 482) —— 形状完全不变，只是数值变成了"注意力权重/概率"

   几何含义：第 t 个 query 在第 h 头上"关注"encoder 482 个 token 的概率分布；
             权重越大的位置，后面加权求和时贡献越多。

⑤ 加权求和，就是矩阵乘法：

   head_out = weight · V
            = (8, 100, 482) × (8, 482, 64) = (8, 100, 64)
   8 个头各自独立做二维矩阵乘法（batch matmul），head_out[h] = weight[h] · V[h]

⑥ 拼接多头 + 输出投影 W_O，就是 reshape + 矩阵乘法：
    reshape就是取消多头
   (8, 100, 64) → transpose+reshape → (100, 512)   # 8×64 拼回 512，零拷贝
   output = (100, 512) · W_O^T + b_O = (100, 512) → unsqueeze → (100, 1, 512)

⑦ 残差 + 归一化：
   x = skip + dropout(output)        # skip 是 self-attn 后的 β（常数）
                                     # (100, 1, 512) + (100, 1, 512) = (100, 1, 512)
   x = norm2(x)                      # nn.LayerNorm(512)，对每个 token 的 512 维独立计算：
                                     #   mean = 512个值的均值（标量）
                                     #   std  = 512个值的标准差（标量）
                                     #   逐元素归一化：x = (x - mean) / (std + ε)
                                     #   逐元素缩放平移：x = γ * x + β   （γ/β 是可学习参数，形状 (512,)）
                                     #   输出 (100, 1, 512)
   skip = x                          # (100, 1, 512)
```



#### FLOPs（单层，B=1）

| 子步骤             | 计算                     | FLOPs             |
| ------------------ | ------------------------ | ----------------- |
| Q 投影（长度 100） | `2 × 100 × 512²`         | ~52 M             |
| K 投影（长度 482） | `2 × 482 × 512²`         | ~253 M            |
| V 投影（长度 482） | `2 × 482 × 512²`         | ~253 M            |
| Q·K^T              | `2 × 8 × 100 × 482 × 64` | ~49 M             |
| softmax · V        | 同上                     | ~49 M             |
| 输出投影 W_O       | `2 × 100 × 512²`         | ~52 M             |
| **小计**           |                          | **≈ 0.71 GFLOPs** |

---

### 6.5 子层③：FFN 前馈网络

```python
skip = x
x = self.linear2(dropout(activation(self.linear1(x))))
x = skip + dropout(x)
x = self.norm3(x)
```

和 encoder 的 FFN 结构**完全一致**（详见 [§5.3 子层②：前馈网络（FFN）](#53-子层前馈网络ffn)），只是序列长度变成 T=100：

```
输入 x：(100, 1, 512)

linear1：nn.Linear(512, 3200)，即 W1 (512, 3200) + b1 (3200,)
  x = x · W1^T + b1       # (100, 1, 512) × (512, 3200) = (100, 1, 3200)  扩维

ReLU：逐元素 max(0, x)    # (100, 1, 3200)，负值清零

linear2：nn.Linear(3200, 512)，即 W2 (3200, 512) + b2 (512,)
  x = x · W2^T + b2       # (100, 1, 3200) × (3200, 512) = (100, 1, 512)  压回

残差 + norm3：x = norm3(skip + dropout(x))  # (100, 1, 512)
```


**FLOPs**：$4 \cdot 100 \cdot 512 \cdot 3200 \approx 0.66\,\text{GFLOPs}$

---

### 6.6 Decoder 最终输出

```python
decoder_out = self.decoder.norm(x)          # nn.LayerNorm(512)，对每个 token 的 512 维独立计算：
                                            #   mean = 512个值的均值（标量）
                                            #   std  = 512个值的标准差（标量）
                                            #   逐元素归一化：x = (x - mean) / (std + ε)
                                            #   逐元素缩放平移：x = γ * x + β  （γ/β 形状 (512,)）
                                            #   输出 (100, 1, 512)，这是 ACTDecoder 整体最后的收尾 norm
decoder_out = decoder_out.transpose(0, 1)   # (1, 100, 512)，把 batch 移到前
```

$$
H_{dec} \in \mathbb{R}^{1 \times 100 \times 512}
$$

**含义**：100 个"动作特征"，每个 512 维。$H_{dec}[0, t]$ 是 ACT 对"未来第 t 步应该做什么"的 512 维内部表示，还需要最后一步 §7 的 `nn.Linear(512 → 6)` 才能变成真正的 6 维关节目标。

**Decoder 单层合计 ≈ 1.37 GFLOPs**：
- 子层① Query Self-Attention：≈ 0（近似无效）
- 子层② Cross-Attention：~0.71 G
- 子层③ FFN：~0.66 G

---

## 7. 步骤 5：Action Head（输出头）

```python
self.action_head = nn.Linear(D=512, A=6)
actions = self.action_head(decoder_out)     # nn.Linear(512, 6)，对每个 token 独立做矩阵乘法：
                                            #   W (512, 6) + b (6,)
                                            #   actions[0, t, :] = decoder_out[0, t, :] · W^T + b
                                            #                     = (512,) × (512, 6) = (6,)
                                            #   100 个 token 各算一次，输出 (1, 100, 6)
```

**做了什么**：用一个 `nn.Linear(512 → 6)` 把每个 512 维的动作特征投影到 6 维（= SO101 的 6 个关节角度）。

**操作分解**：

|                    | 值                                       |
| ------------------ | ---------------------------------------- |
| `self.action_head` | `nn.Linear(dim_model=512, action_dim=6)` |
| 输入 H_dec         | `(1, 100, 512)`                          |
| 输出 actions       | `(1, 100, 6)`                            |

**数学**：

$$
\hat{a}_t = W_a\, H_{dec,t} + b_a,\quad W_a\in\mathbb{R}^{6\times 512},\quad t=0,\dots,99
$$

100 个时刻**共享同一组权重** $W_a$（batched matmul，一次算完 100 帧）。

**输出形状含义**：

| 维度 | 大小 | 含义                              |
| ---- | ---- | --------------------------------- |
| 0    | 1    | batch（推理时 B=1）               |
| 1    | 100  | 未来 100 帧（chunk_size）         |
| 2    | 6    | 每帧 6 个关节的**归一化**目标角度 |

**计算量**：$2 \cdot 100 \cdot 512 \cdot 6 \approx 614\,\text{k FLOPs}$，可忽略。

**返回值**：`actions.shape == (1, 100, 6)`，值**还在归一化空间**。下游要做两件事：

1. `ACTPolicy.select_action` 把这 100 帧塞进 `_action_queue`，后续 99 帧直接 popleft（§1.1 快速路径）
2. 每帧出队后，`postprocessor` 做反归一化 $a_{real} = \hat{a}\cdot\sigma_{action} + \mu_{action}$，得到真正的舵机目标角度（度）

---

## 8. 后处理：action chunk 队列

回到 `ACTPolicy.select_action`（[modeling_act.py:224](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L224)）：

```python
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)[:, :n_action_steps]   # 截取前 n_action_steps
    self._action_queue.extend(actions.transpose(0, 1))
return self._action_queue.popleft()                                   # (B, A)
```

关键事实：

1. **队列为空才推理**。默认 `n_action_steps = chunk_size = 100`，所以每 100 步环境调用一次 `ACT.forward`。
2. 本次命令没设 `--policy.temporal_ensemble_coeff`，所以不会走时序集成路径。
3. 返回给上层 robot 的 tensor 形状是 `(B, A) = (1, 6)`。

这一层的 CPU 时间绝大部分是 **deque 的 popleft**，几乎为 0；真正贵的部分集中在 "队列空" 的那一帧——这一帧要完整跑一次 `ACT.forward`。

---

## 9. 整体计算总量估算（单次 `predict_action_chunk`，B=1）

| 组件                         | FLOPs          | 说明                                   |
| ---------------------------- | -------------- | -------------------------------------- |
| ResNet18 × 2 摄像头          | **~24 G**      | 360×640，主导                          |
| 1×1 Conv 投影 (512→D) × 2    | ~0.13 G        |                                        |
| 2D 正弦 pos embed            | ~0             | 无乘法密集计算                         |
| Encoder 4 层 (L=482)         | **~18.6 G**    | 其中 FFN 占 ~12.6 G，self-attn 占 ~6 G |
| Decoder 1 层 (Q=100, KV=482) | **~1.37 G**    | cross-attn 0.71 + FFN 0.66             |
| Action head                  | ~0.6 M         | 忽略                                   |
| **合计**                     | **~44 GFLOPs** | ResNet 和 encoder FFN 是两个大头       |

在树莓派 5（ARM Cortex-A76，无 GPU）上，假设单核 PyTorch 跑浮点可达 ~5 GFLOPS，则一次推理粗估 **~8–10 秒**——这与 `infer分析` 姊妹目录里 futex/pselect6 测得的观察基本吻合，也解释了为何默认配置里 `n_action_steps = chunk_size = 100`（尽量减少推理频率）。

> **单位说明**：上面用 2·m·n·k 近似 GEMM 的 FLOPs（一次乘一次加），与 `thop` / `fvcore` 的 MACs 数值差一个系数 2。

---

## 10. 一次推理从"数学总览"回看

把上面所有步骤压缩成一个公式流（符号见 §0）：

1. **Latent（全零）**：
$$z = \mathbf{0}_{B\times Z}$$

2. **1D token**：
$$x_{latent} = W_{lat}z + b_{lat},\quad x_{state} = W_{state}s + b_{state}$$

3. **图像 token**（对每路相机 $c\in\{0,1\}$）：
$$F_c = \text{ResNet18.layer4}(I_c)\in\mathbb{R}^{B\times C_{res}\times h\times w}$$
$$G_c = W_{img}*F_c + b_{img}\quad(\text{1×1 conv})$$
$$X_{cam,c} = \text{flatten}_{h,w}(G_c)\in\mathbb{R}^{(h w)\times B\times D}$$

4. **Encoder 序列**（拼起来 $L_{enc\_seq}=482$）：
$$X_{enc}^{(0)} = [x_{latent};\; x_{state};\; X_{cam,0};\; X_{cam,1}]$$
$$P_{enc} = [p_{lat};\; p_{state};\; \text{SinPos2d}(F_0);\; \text{SinPos2d}(F_1)]$$

5. **Encoder（4 层 post-norm）**：每层
$$X^{(\ell)} = \text{LN}\!\Big(X^{(\ell-1)} + \text{MHA}(X^{(\ell-1)}{+}P_{enc},\,X^{(\ell-1)}{+}P_{enc},\,X^{(\ell-1)})\Big)$$
$$X^{(\ell)} \leftarrow \text{LN}\!\Big(X^{(\ell)} + W_2\text{ReLU}(W_1 X^{(\ell)})\Big)$$
最终 $H_{enc} = X^{(4)}$。

6. **Decoder（1 层）**，输入 $Y^{(0)} = \mathbf{0}_{T\times B\times D}$，$P_{dec}\in\mathbb{R}^{T\times 1\times D}$（可学习 object queries）：
$$Y' = \text{LN}\!\Big(Y^{(0)} + \text{MHA}(Y^{(0)}{+}P_{dec},\,Y^{(0)}{+}P_{dec},\,Y^{(0)})\Big) \;\text{(首层 value=0 近似无效)}$$
$$Y'' = \text{LN}\!\Big(Y' + \text{CrossAttn}(Y'{+}P_{dec},\,H_{enc}{+}P_{enc},\,H_{enc})\Big)$$
$$Y^{(1)} = \text{LN}\!\Big(Y'' + W_2\text{ReLU}(W_1 Y'')\Big)$$
$$H_{dec} = \text{LN}(Y^{(1)})^\top \in \mathbb{R}^{B\times T\times D}$$

7. **动作头**：
$$\hat{A} = H_{dec}\, W_a^\top + b_a \in \mathbb{R}^{B\times T\times A}$$

8. **反归一化**（由 `normalize.py` 外层处理）：
$$A_{out} = \hat{A}\cdot \sigma_{action} + \mu_{action}$$

---

## 11. 推理时"省掉"的部分

与训练相比，推理里被跳过的计算：

| 被跳过的东西                                        | 原因                                              |
| --------------------------------------------------- | ------------------------------------------------- |
| VAE encoder（4 层 Transformer，输入 1+1+100 token） | `self.training=False`，走 else 分支，`latent = 0` |
| VAE 的 CLS token 投影、重参数化采样                 | 同上                                              |
| L1 loss、KL loss 计算                               | 推理不算损失                                      |
| Dropout                                             | `eval()` 模式下等价于恒等                         |
| BatchNorm 统计量更新                                | `FrozenBatchNorm2d` 本就不更新                    |

另外，**整个前向在 `@torch.no_grad()` 下执行**，所有中间 tensor 都不保留 grad_fn，也不会写 autograd buffer，这对峰值内存和 cache 友好度都有明显帮助。

---

## 12. 交叉引用

- 源码主体：[`ACT.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782)
- 配置默认值：[`ACTConfig`](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L37)
- Encoder 层：[`ACTEncoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1080)
- Decoder 层：[`ACTDecoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1238)
- 2D 正弦位置编码：[`ACTSinusoidalPositionEmbedding2d`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)
- 调用点：`ACTPolicy.select_action → predict_action_chunk`，[modeling_act.py:224](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L224)

---

## 13. 模块调用关系图

```
record.py（控制循环）
└─ control_utils.predict_action()                     utils/control_utils.py:126
   │
   ├─ [快速路径] policy._action_queue.popleft()        每帧直接取缓存，不调模型
   │    └─ postprocessor(action)                       反归一化后返回
   │
   └─ [完整路径] 队列为空时每 100 帧触发一次
        │
        ├─ A. numpy → Tensor 格式转换                  utils/control_utils.py:79-88
        │
        ├─ B. preprocessor(observation)                policies/act/processor_act.py
        │    └─ NormalizerProcessorStep.__call__()     policies/act/normalize_processor.py
        │         └─ _apply_transform()                (x - μ) / σ，统计量来自 checkpoint
        │
        ├─ C. ACTPolicy.select_action(batch)           policies/act/modeling_act.py:224
        │    └─ predict_action_chunk(batch)            policies/act/modeling_act.py:289
        │         └─ ACT.forward(batch)                policies/act/modeling_act.py:782
        │              │
        │              ├─ 1. latent = zeros(1, 32)     推理时跳过 VAE encoder
        │              │
        │              ├─ 2. 构造 encoder 输入序列
        │              │    ├─ encoder_latent_input_proj    nn.Linear(32→512)
        │              │    ├─ encoder_robot_state_input_proj  nn.Linear(6→512)
        │              │    └─ for img in observation.images（×2路摄像头）
        │              │         ├─ backbone(img)           torchvision.IntermediateLayerGetter
        │              │         │    └─ ResNet18.layer4    Conv2d×8 + BN + ReLU + maxpool
        │              │         │         (1,3,360,640)→(1,512,12,20)，共 ~12 GFLOPs/路
        │              │         ├─ encoder_cam_feat_pos_embed   policies/act/modeling_act.py:1346
        │              │         │    ACTSinusoidalPositionEmbedding2d，纯 sin/cos，无参数
        │              │         ├─ encoder_img_feat_input_proj  nn.Conv2d(512→512, 1×1)
        │              │         └─ einops.rearrange → (240,1,512) token 序列
        │              │
        │              ├─ 3. Transformer Encoder × 4 层   policies/act/modeling_act.py:1080
        │              │    每层：MultiheadAttention(512,8头) + FFN(512→3200→512)
        │              │    输入/输出：(482, 1, 512)
        │              │
        │              ├─ 4. Transformer Decoder × 1 层   policies/act/modeling_act.py:1238
        │              │    输入 query: zeros(100,1,512)
        │              │    memory: encoder_out (482,1,512)
        │              │    Self-Attn + Cross-Attn + FFN → (100,1,512)
        │              │
        │              └─ 5. action_head  nn.Linear(512→6)
        │                   (1,100,512) → (1,100,6)  即 (B, chunk_size, action_dim)
        │
        └─ D. postprocessor(action)                    policies/act/processor_act.py
             └─ UnnormalizerProcessorStep.__call__()  policies/act/normalize_processor.py
                  └─ _apply_transform(inverse=True)   x̂ = z·σ + μ → 真实舵机角度
```

### 涉及的源文件索引

| 文件 | 职责 |
|---|---|
| [record.py](../../../lerobot/src/lerobot/record.py) | 控制循环入口 |
| [control_utils.py](../../../lerobot/src/lerobot/utils/control_utils.py) | `predict_action` 包装层 |
| [processor_act.py](../../../lerobot/src/lerobot/policies/act/processor_act.py) | 前/后处理流水线定义 |
| [normalize_processor.py](../../../lerobot/src/lerobot/policies/act/normalize_processor.py) | 归一化 / 反归一化核心 |
| [modeling_act.py](../../../lerobot/src/lerobot/policies/act/modeling_act.py) | ACTPolicy、ACT.forward、Encoder/Decoder 层、2D 位置编码 |
| [configuration_act.py](../../../lerobot/src/lerobot/policies/act/configuration_act.py) | 超参数配置 |
| torchvision ResNet18 | backbone（IntermediateLayerGetter + BasicBlock） |


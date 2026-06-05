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

## 4. 步骤 2A：构造 X_enc（内容 token）

```python
# latent token: (B,32) → Linear(32→512) → (B,512)，推理时全零
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
# robot_state token: (B,6) → Linear(6→512) → (B,512)
encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
# 图像 token 见 §4.4；P_enc 在下一章单独构造
```

### 4.0 这一步到底做了什么

这一章只讲 `X_enc`，也就是源码里的 `encoder_in_tokens`：把 3 类**维度不同**的原始数据（32 维 latent、6 维 state、每个像素 512 维的视觉特征）统一投影到 `D=512` 维，然后拼成长度 482 的内容 token 序列。

`P_enc`（源码里的 `encoder_in_pos_embed`）是另一条并行的位置编码序列，下一章单独讲。

为什么必须统一到 512 维：Transformer 的 self-attention 要求**所有 token 同维度**才能做矩阵乘法（Q·K^T）。6 维的关节状态不投影，根本没法和 512 维的图像特征"坐在同一张桌子上"做注意力。

### 4.1 X_enc 的长什么样

$$
X_{enc}^{(0)} = [\underbrace{x_{latent}}_{1},\; \underbrace{x_{state}}_{1},\; \underbrace{x_{cam0,1}, \dots, x_{cam0,240}}_{240},\; \underbrace{x_{cam1,1}, \dots, x_{cam1,240}}_{240}] \in \mathbb{R}^{482 \times 1 \times 512}
$$

| 位置        | token 数 | 来源                           | 投影方式              |
| ----------- | -------- | ------------------------------ | --------------------- |
| `[0]`       | 1        | latent（推理时全零）           | `nn.Linear(32 → 512)` |
| `[1]`       | 1        | 6 个关节角度                   | `nn.Linear(6 → 512)`  |
| `[2:242]`   | 240      | handeye 摄像头（12×20 特征图） | ResNet18 + 1×1 Conv   |
| `[242:482]` | 240      | fixed 摄像头（12×20 特征图）   | ResNet18 + 1×1 Conv   |

下面 §4.2 / §4.3 / §4.4 分别展开 latent、state 和 image 三类内容 token 的构造过程。

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

这是 Encoder 输入构造里**最复杂、最费时**的部分（占整次推理计算量的 >50%）。对每路摄像头图像，要依次做 3 步：

```python
# batch["observation.images"] = [Tensor(1,3,360,640), Tensor(1,3,360,640)]  ← handeye, fixed
for img in batch["observation.images"]:
    # img: (1, 3, 360, 640)，归一化后的 RGB 图像

    cam_features  = self.backbone(img)["feature_map"]
    # ① ResNet18.layer4：(1,3,360,640) → (1,512,12,20)   ← 空间压缩32倍，每位置512维

    cam_features  = self.encoder_img_feat_input_proj(cam_features)
    # ② 1×1 Conv：(1,512,12,20) → (1,512,12,20)          ← 对每个像素独立做线性变换，适配Transformer

    cam_features  = einops.rearrange(cam_features, "b c h w -> (h w) b c")
    # ③ 展平：(1,512,12,20) → (240,1,512)                ← 12×20个位置变成240个token

    encoder_in_tokens.extend(list(cam_features))      # 追加240个内容 token 到 X_enc 序列
```

整体流程的形状变化：

| 步骤              | 形状               | 说明                       |
| ----------------- | ------------------ | -------------------------- |
| 输入图像          | `(1, 3, 360, 640)` | 归一化后的图像             |
| ① ResNet18.layer4 | `(1, 512, 12, 20)` | 空间缩小 32×，通道升到 512 |
| ② 1×1 Conv        | `(1, 512, 12, 20)` | 通道数不变，换表示空间     |
| ③ 展平            | `(240, 1, 512)`    | 每个空间位置变成一个 token |

两路摄像头走完，共 2×240 = **480 个 image tokens**。

---

#### 4.4.1 ① ResNet18 backbone：图像 → 特征图

> 详细逐层展开见：[ResNet18视觉特征提取.md](./ResNet18视觉特征提取.md)。主链路这里只保留输入输出。

```python
cam_features = self.backbone(img)["feature_map"]
```

| 项目 | 形状 | 含义 |
| --- | --- | --- |
| 输入 `img` | `(1, 3, 360, 640)` | 单路归一化后的摄像头图像 |
| 输出 `cam_features` | `(1, 512, 12, 20)` | ResNet18 `layer4` 特征图 |

单路图像经过 ResNet18 后，空间尺寸从 `360×640` 压到 `12×20`，也就是 240 个空间位置；每个位置 512 维。


#### 4.4.2 ② 1×1 卷积投影：换"表示空间"

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

#### 4.4.3 ③ 展平为 token 序列

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
```

**为什么要展平**：Transformer 处理的是序列，attention 不关心"空间位置是第几行第几列"。所以把 12×20 的 2D 网格变成长度 240 的 1D 序列。失去的空间信息由下一章的 `P_enc` 补回来。

---

### 4.5 X_enc 最终堆叠

```python
encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
```

`encoder_in_tokens` 是一路只装**内容特征**的 list，最后 stack 成 `X_enc`：

```text
[0]       latent token   (1,512)
[1]       state token    (1,512)
[2:242]   handeye image  240 个 (1,512)
[242:482] fixed image    240 个 (1,512)

X_enc = encoder_in_tokens = (482, 1, 512)
```

这里的 482 个 token 只回答一个问题：**每个 token 的内容是什么**。

---

## 5. 步骤 2B：构造 P_enc（位置编码）

`P_enc` 对应源码里的 `encoder_in_pos_embed`。它和 `X_enc` 长度完全一致，也是 `(482, 1, 512)`，但它回答的是另一个问题：**每个 token 从哪里来**。

```python
encoder_in_pos_embed = list(
    self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
)
```

### 5.0 P_enc 的长什么样

$$
P_{enc} = [p_{latent};\; p_{state};\; P_{cam0};\; P_{cam1}] \in \mathbb{R}^{482 \times 1 \times 512}
$$

| 位置 | token 数 | 位置编码来源 |
| --- | --- | --- |
| `[0]` | 1 | latent token 的 1D 可学习位置编码 |
| `[1]` | 1 | state token 的 1D 可学习位置编码 |
| `[2:242]` | 240 | handeye 图像的 2D 正弦位置编码 |
| `[242:482]` | 240 | fixed 图像的 2D 正弦位置编码 |

### 5.1 前两个 token：1D 可学习位置编码

```python
encoder_in_pos_embed = list(
    self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
)
```

`self.encoder_1d_feature_pos_embed` 是一个可学习 embedding，至少提供两个位置向量：

```text
p_latent: (1,512)  ← 对应 X_enc[0]
p_state:  (1,512)  ← 对应 X_enc[1]
```

它们不是传感器数据，而是训练得到的“这个 token 类型/位置是谁”的标记。

### 5.2 图像位置编码：2D 正弦位置编码

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

#### 第1步：生成坐标网格（cumsum 累加）

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

#### 第2步：归一化到 [0, 2π]

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

#### 第3步：生成 256 个频率值

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

#### 第4步：每个坐标除以 256 个频率，得到 256 个"相位角"，再取 sin/cos

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

#### 第5步：拼接 y 和 x 编码

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

#### 输出：cam_pos_embed (1, 512, 12, 20)

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



### 5.3 P_enc 最终堆叠

图像位置编码和图像内容 token 做同样的展平：

```python
cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
encoder_in_pos_embed.extend(list(cam_pos_embed))
encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
```

最终：

```text
P_enc = encoder_in_pos_embed = (482, 1, 512)
```

`X_enc` 和 `P_enc` 一一对齐：

| 索引 | `X_enc` 内容 | `P_enc` 位置 |
| --- | --- | --- |
| `[0]` | latent 内容 token | latent 可学习位置编码 |
| `[1]` | state 内容 token | state 可学习位置编码 |
| `[2:242]` | handeye 图像内容 token | handeye 2D 正弦位置编码 |
| `[242:482]` | fixed 图像内容 token | fixed 2D 正弦位置编码 |

进入 Encoder 时：

```python
encoder_out = self.encoder(
    encoder_in_tokens,                 # X_enc，内容 token，(482,1,512)
    pos_embed=encoder_in_pos_embed,     # P_enc，位置编码，(482,1,512)
)
```

位置编码只加到 Q 和 K：Transformer 里位置编码**不是加到 value 上的**，而是在每层 attention 计算前单独加到 Q 和 K 上。Q·K 是“算相关性”的地方，加位置编码让相关性计算知道 token 的来源；V 是“内容本身”，不额外加入位置编码。

---

## 6. 步骤 3：Transformer Encoder（4 层）

> 详细逐层展开见：[TransformerEncoder推理.md](./TransformerEncoder推理.md)，结构图见：[TransformerEncoder结构.svg](./TransformerEncoder结构.svg)。主链路这里只保留输入输出。

```py
encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
```

| 参数 | 形状 | 内容 |
| --- | --- | --- |
| `encoder_in_tokens` / `X_enc` | `(482, 1, 512)` | 1 个 latent token + 1 个 state token + 480 个 image token |
| `encoder_in_pos_embed` / `P_enc` | `(482, 1, 512)` | 与 token 一一对应的位置编码 |
| `encoder_out` / `H_enc` | `(482, 1, 512)` | 4 层 Encoder 后的 memory，形状不变，内容融合全局上下文 |

$$
H_{enc} = \text{Enc}(X_{enc}^{(0)}; P_{enc}) \in \mathbb{R}^{482 \times 1 \times 512}
$$

`H_enc` 后续进入 Decoder：100 个未来动作 query 会去查询这 482 个 memory token。

---

## 7. 步骤 4：Transformer Decoder（DETR 风格，1 层）

> 详细逐层展开见：[TransformerDecoder推理.md](./TransformerDecoder推理.md)，结构图见：[TransformerDecoder结构.svg](./TransformerDecoder结构.svg)。主链路这里只保留输入输出。

```python
decoder_in = torch.zeros((self.config.chunk_size, batch_size, self.config.dim_model))

decoder_out = self.decoder(
    decoder_in,
    encoder_out,
    encoder_pos_embed=encoder_in_pos_embed,
    decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
)

decoder_out = decoder_out.transpose(0, 1)
```

| 参数 | 形状 | 内容 |
| --- | --- | --- |
| `decoder_in` | `(100, 1, 512)` | 100 个动作 query 的内容占位，推理时全零 |
| `decoder_pos_embed` / `P_dec` | `(100, 1, 512)` | 来自 `self.decoder_pos_embed.weight` 的可学习参数，对应未来 100 个动作步 |
| `encoder_out` / `H_enc` | `(482, 1, 512)` | Encoder 输出的观测 memory |
| `encoder_in_pos_embed` / `P_enc` | `(482, 1, 512)` | Encoder 侧位置编码，Cross-Attention 中加到 Key |
| `decoder_out` | `(100, 1, 512)` | Decoder 输出，尚未转置 |
| `H_dec` | `(1, 100, 512)` | 转置后的动作特征，送入 Action Head |

核心关系：

$$
H_{dec} = \text{Dec}(\mathbf{0}_{100\times 1\times 512},\; H_{enc},\; P_{dec},\; P_{enc}) \in \mathbb{R}^{1 \times 100 \times 512}
$$

Decoder 用 100 个未来动作 query 查询 Encoder 的 482 个 memory token，生成 100 个动作特征。

---

## 8. 步骤 5：Action Head（输出头）

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

## 9. 后处理：action chunk 队列

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

## 10. 整体计算总量估算（单次 `predict_action_chunk`，B=1）

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

## 11. 一次推理从"数学总览"回看

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

5. **Encoder（4 层）**：
$$H_{enc} = \text{Enc}(X_{enc}^{(0)}, P_{enc})\in\mathbb{R}^{482\times B\times D}$$

6. **Decoder（1 层）**：
$$H_{dec} = \text{Dec}(\mathbf{0}_{T\times B\times D},\; H_{enc},\; P_{dec},\; P_{enc})^\top \in \mathbb{R}^{B\times T\times D}$$

7. **动作头**：
$$\hat{A} = H_{dec}\, W_a^\top + b_a \in \mathbb{R}^{B\times T\times A}$$

8. **反归一化**（由 `normalize.py` 外层处理）：
$$A_{out} = \hat{A}\cdot \sigma_{action} + \mu_{action}$$

---

## 12. 推理时"省掉"的部分

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

## 13. 交叉引用

- 源码主体：[`ACT.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782)
- 配置默认值：[`ACTConfig`](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L37)
- Encoder 层：[`ACTEncoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1080)
- Transformer Encoder 详解：[TransformerEncoder推理.md](./TransformerEncoder推理.md)
- Transformer Encoder 结构图：[TransformerEncoder结构.svg](./TransformerEncoder结构.svg)
- Decoder 层：[`ACTDecoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1295)
- Transformer Decoder 详解：[TransformerDecoder推理.md](./TransformerDecoder推理.md)
- Transformer Decoder 结构图：[TransformerDecoder结构.svg](./TransformerDecoder结构.svg)
- 2D 正弦位置编码：[`ACTSinusoidalPositionEmbedding2d`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)
- ResNet18 视觉链路详解：[ResNet18视觉特征提取.md](./ResNet18视觉特征提取.md)
- 调用点：`ACTPolicy.select_action → predict_action_chunk`，[modeling_act.py:224](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L224)

---

## 14. 模块调用关系图

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
        │              │         │    └─ ResNet18.layer4    (1,3,360,640) → (1,512,12,20)
        │              │         ├─ encoder_cam_feat_pos_embed   policies/act/modeling_act.py:1346
        │              │         │    ACTSinusoidalPositionEmbedding2d，纯 sin/cos，无参数
        │              │         ├─ encoder_img_feat_input_proj  nn.Conv2d(512→512, 1×1)
        │              │         └─ einops.rearrange → (240,1,512) token 序列
        │              │
        │              ├─ 3. Transformer Encoder × 4 层   policies/act/modeling_act.py:1080
        │              │    输入 X_enc/P_enc：(482,1,512)，输出 H_enc：(482,1,512)
        │              │
        │              ├─ 4. Transformer Decoder × 1 层   policies/act/modeling_act.py:1238
        │              │    输入 query：(100,1,512)，memory：H_enc (482,1,512)
        │              │    输出 H_dec：(100,1,512) → transpose 后 (1,100,512)
        │              │
        │              └─ 5. action_head  nn.Linear(512→6)
        │                   (1,100,512) → (1,100,6)  即 (B, chunk_size, action_dim)
        │
        └─ D. postprocessor(action)                    policies/act/processor_act.py
             └─ UnnormalizerProcessorStep.__call__()  policies/act/normalize_processor.py
                  └─ _apply_transform(inverse=True)   x̂ = z·σ + μ → 真实舵机角度
```

### 涉及的源文件索引

| 文件                                                                                       | 职责                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| [record.py](../../../lerobot/src/lerobot/record.py)                                        | 控制循环入口                                            |
| [control_utils.py](../../../lerobot/src/lerobot/utils/control_utils.py)                    | `predict_action` 包装层                                 |
| [processor_act.py](../../../lerobot/src/lerobot/policies/act/processor_act.py)             | 前/后处理流水线定义                                     |
| [normalize_processor.py](../../../lerobot/src/lerobot/policies/act/normalize_processor.py) | 归一化 / 反归一化核心                                   |
| [modeling_act.py](../../../lerobot/src/lerobot/policies/act/modeling_act.py)               | ACTPolicy、ACT.forward、Encoder/Decoder 层、2D 位置编码 |
| [configuration_act.py](../../../lerobot/src/lerobot/policies/act/configuration_act.py)     | 超参数配置                                              |
| [ResNet18视觉特征提取.md](./ResNet18视觉特征提取.md)                                      | ResNet18 backbone 逐层视觉特征提取链路                  |
| [TransformerEncoder推理.md](./TransformerEncoder推理.md)                                    | ACT Transformer Encoder 单层结构、自注意力、FFN 和计算量 |
| [TransformerDecoder推理.md](./TransformerDecoder推理.md)                                    | ACT Transformer Decoder 单层结构、Cross-Attention、FFN 和计算量 |
| torchvision ResNet18                                                                       | backbone（IntermediateLayerGetter + BasicBlock）        |

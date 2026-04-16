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

| 符号 | 含义 | 值 |
|---|---|---|
| `B` | batch_size 推理时单帧推理 | `1` |
| `S` | state_dim（关节数） | `6`（SO101 单臂 6 DOF）|
| `A` | action_dim | `6` |
| `T` | chunk_size（一次预测的动作步数） | `100` |
| `D` | dim_model（Transformer 隐藏维度） | `512` |
| `H` | n_heads | `8`（每头 `D/H = 64` 维）|
| `F` | dim_feedforward | `3200` |
| `L_enc` | Transformer encoder 层数 | `4` |
| `L_dec` | Transformer decoder 层数 | `1`（与原 ACT 代码对齐，见 [configuration_act.py:168](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L168)）|
| `Z` | latent_dim | `32` |
| `N_cam` | 摄像头数 | `2`（handeye + fixed）|
| `H_in × W_in` | 输入图像尺寸 | `360 × 640` |
| `h × w` | ResNet18 `layer4` 特征图尺寸 | `360/32 × 640/32 = 12 × 20 = 240` 个空间位置 |
| `C_res` | ResNet18 `layer4` 通道数 | `512` |
| `N_img` | 所有摄像头的图像 token 总数 | `N_cam · h · w = 2 · 240 = 480` |
| `L_enc_seq` | Transformer encoder 输入序列长度 | `1(latent) + 1(state) + 480(image) = 482` |

推理时 `use_vae=True` 但模型处于 `eval()` 模式，走 **非训练分支**：`latent_sample = 0 ∈ ℝ^{B×Z}`，VAE encoder 不参与前向。

---

## 1. 外层入口：`predict_action`（control_utils.py）

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

| key | 转换前（numpy） | 转换后（Tensor） |
|---|---|---|
| `observation.state` | `(6,)` float32 | `(1, 6)` float32，on device |
| `observation.images.handeye` | `(360, 640, 3)` uint8 | `(1, 3, 360, 640)` float32 ∈ [0,1] |
| `observation.images.fixed` | `(360, 640, 3)` uint8 | `(1, 3, 360, 640)` float32 ∈ [0,1] |

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

**经过这步的数据变化**：

| 字段 | 输入 | 输出 |
|------|------|------|
| `observation.state` | `(1,6)` 真实角度（度） | `(1,6)` 无量纲，≈N(0,1) |
| `observation.images.handeye` | `(1,3,360,640)` ∈ [0,1] | `(1,3,360,640)` ≈N(0,1) |
| `observation.images.fixed` | `(1,3,360,640)` ∈ [0,1] | `(1,3,360,640)` ≈N(0,1) |

归一化参数（μ, σ）全部来自**训练集自身的统计量**，保存在 checkpoint 里，不是 ImageNet 预设值。
统计量的计算和加载链路见下方 **§1.4 归一化统计量的来源**。

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

| 字段 | 输入 | 输出 |
|------|------|------|
| `action` | `(1,6)` ≈N(0,1) 归一化空间 | `(1,6)` 真实角度（度），可直接写入 Feetech 舵机 |

### 1.3 数据流小结

```
摄像头 / 舵机（numpy）
  │
  ├─ [快速路径] 队列有缓存 → popleft → postprocess → (6,) 返回
  │
  └─ [慢路径，每 100 帧触发一次]
       numpy → Tensor + channel-first + batch维
       ↓
       preprocessor（4步流水线：rename → add_batch → device → MEAN_STD 归一化）
       ↓ NormalizerProcessorStep: 对 state 和图像统一做 (x-μ)/σ，μ/σ 来自训练集统计量
       ↓ observation dict：{state:(1,6)≈N(0,1), handeye:(1,3,360,640)≈N(0,1), fixed:(1,3,360,640)≈N(0,1)}
       ↓
       policy.select_action()         ← §2 起开始详细展开
       ↓ action: (1, 6) 归一化空间
       ↓
       postprocessor（2步流水线：UnnormalizerStep(x·σ+μ) → DeviceStep(to cpu)）
       ↓
       (6,) float32，CPU             ← 传给机器人驱动执行（Feetech 舵机目标角度，单位度）
```

### 1.4 归一化统计量（μ, σ）的来源

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

```
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
    mu = log_sigma_x2 = None # 不走VAE
    latent_sample = torch.zeros(
        [batch_size, self.config.latent_dim],  # (1, 32) 全零向量
        dtype=torch.float32,
    ).to(batch["observation.state"].device) # 潜变量全部全零

    # ── 步骤 2：构造 Transformer Encoder 输入序列 ─────────────────
    # token 顺序: [latent, robot_state, cam0_pixels..., cam1_pixels...]
    encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(
        self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
    )

    # Robot state token
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

### 3.2 VAE 在推理时到底用不用得到？—— **完全用不到**

虽然 `use_vae=True`，但 [ACT.forward 的判断条件](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L826)是：

```python
if self.config.use_vae and "action" in batch and self.training:
    # 走 VAE encoder 分支
    ...
else:
    # 推理分支：latent = 0
    ...
```

推理时同时有两个条件不满足：
1. `self.training = False`（`ACTPolicy.select_action` 里先调了 `self.eval()`，[modeling_act.py:262](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L262)）
2. `batch` 里根本没有 `"action"` 这个 key（推理时只有观测，没有未来动作）

所以整个 `vae_encoder` 子模块及其投影层（`vae_encoder_cls_embed`、`vae_encoder_robot_state_input_proj`、`vae_encoder_action_input_proj`、`vae_encoder_latent_output_proj`）**前向传播一次都不会被调用**。

> **但参数依然在内存里**：`ACT.__init__` 里只要 `config.use_vae=True` 就会无条件构造这些子模块（[modeling_act.py:647-683](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L647-L683)），它们的权重在 checkpoint 里也会被保存和加载。推理时只是不走它们的前向 —— 属于"加载了但不跑"。
>
> 在资源紧张的树莓派 5 上，如果想节省这部分内存，可以在加载完 checkpoint 后手动 `del policy.model.vae_encoder` 等，然后改成直接构造 `latent_sample = zeros(...)`。但对比整个模型参数量这部分占比很小（VAE encoder ≈ 4 层 Transformer，约 12M 参数），多数情况下不值得折腾。

**结论**：`use_vae` 只影响**训练**阶段是否走变分目标；对**推理**而言，不论 `use_vae` 是 True 还是 False，都走同一条 `latent = 0` 路径。

> 本步骤计算量：0。

---

## 4. 步骤 2：构造 Encoder 输入 token

源码：[modeling_act.py:898-948](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L898-L948)

### 4.0 这一步到底做了什么

把 3 类**维度不同**的原始数据（32 维 latent、6 维 state、每个像素 512 维的特征图）**统一投影到 D=512 维**，然后拼成一个长度 482 的 token 序列，喂给 Transformer Encoder。

为什么必须统一到 512 维：Transformer 的 self-attention 要求**所有 token 同维度**才能做矩阵乘法（Q·K^T）。6 维的关节状态不投影，根本没法和 512 维的图像特征"坐在同一张桌子上"做注意力。

### 4.1 输入序列的长什么样

$$
X_{enc}^{(0)} = [\underbrace{x_{latent}}_{1},\; \underbrace{x_{state}}_{1},\; \underbrace{x_{cam0,1}, \dots, x_{cam0,240}}_{240},\; \underbrace{x_{cam1,1}, \dots, x_{cam1,240}}_{240}] \in \mathbb{R}^{482 \times 1 \times 512}
$$

| 位置 | token 数 | 来源 | 投影方式 |
|---|---|---|---|
| `[0]` | 1 | latent（推理时全零） | `nn.Linear(32 → 512)` |
| `[1]` | 1 | 6 个关节角度 | `nn.Linear(6 → 512)` |
| `[2:242]` | 240 | handeye 摄像头（12×20 特征图） | ResNet18 + 1×1 Conv |
| `[242:482]` | 240 | fixed 摄像头（12×20 特征图） | ResNet18 + 1×1 Conv |

下面 §4.2 / §4.3 / §4.4 / §4.5 分别展开这 4 类 token 的构造过程。

---

### 4.2 Latent token（1 个）

```python
encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
# latent_sample: (1, 32) 全零
```

**做了什么**：用一个 `nn.Linear(32 → 512)` 把全零的 latent 向量"拉伸"到 512 维。

**操作分解**：

| | 值 |
|---|---|
| `self.encoder_latent_input_proj` | `nn.Linear(latent_dim=32, dim_model=512)` |
| 输入 z | `(1, 32)` 全零 |
| 计算 | `x_latent = z @ W.T + b = 0 + b = b` |
| 输出 x_latent | `(1, 512)` |

由于 z 是全零，结果恰好是 `nn.Linear` 的偏置 $b_{lat}$（一个可学习的 512 维向量）。

**数学**：

$$
x_{latent} = W_{lat}\, z + b_{lat},\quad W_{lat} \in \mathbb{R}^{512 \times 32}
$$

**latent token 的意义**：这是 VAE 留下的"风格/意图"通道。训练时它从 VAE encoder 采样得到，表达"这次动作序列的风格"；推理时没有 VAE 输入，所以用全零占位。模型训练时见过大量"z=0"的情况，已经学会了一个默认行为。

**计算量**：`2·D·Z ≈ 33k FLOPs`，可忽略。

---

### 4.3 Robot state token（1 个）

```python
encoder_in_tokens.append(
    self.encoder_robot_state_input_proj(batch["observation.state"])
)
# batch["observation.state"]: (1, 6)，归一化后的关节角度
```

**做了什么**：用一个 `nn.Linear(6 → 512)` 把 6 维的关节角度投影到 512 维。

**操作分解**：

| | 值 |
|---|---|
| `self.encoder_robot_state_input_proj` | `nn.Linear(state_dim=6, dim_model=512)` |
| 输入 s | `(1, 6)`，MEAN_STD 归一化后的关节角度（均值 0 方差 1） |
| 输出 x_state | `(1, 512)` |

**数学**：

$$
x_{state} = W_{state}\, s + b_{state},\quad W_{state} \in \mathbb{R}^{512 \times 6}
$$

**本质就是矩阵乘法**：把一个 6 维向量"搬运"到 512 维空间，让它能和图像特征一起参与注意力计算。6 维里的每一维（每个关节）都会被"打散"到 512 个新维度里，具体怎么打散由训练学到的权重 $W_{state}$ 决定。

**计算量**：`2·D·S ≈ 6k FLOPs`，可忽略。

---

### 4.4 图像 token（每路 240 个，共 480 个）

这是 Encoder 输入构造里**最复杂、最费时**的部分（占整次推理计算量的 >50%）。对每路摄像头图像，要依次做 4 步：

```python
for img in batch["observation.images"]:
    # img: (1, 3, 360, 640)，已做 ImageNet 归一化
    cam_features  = self.backbone(img)["feature_map"]                  # ① ResNet18 提取特征
    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)      # ② 2D 正弦位置编码
    cam_features  = self.encoder_img_feat_input_proj(cam_features)     # ③ 1×1 Conv 投影
    cam_features  = einops.rearrange(cam_features, "b c h w -> (h w) b c")  # ④ 展平为序列
    encoder_in_tokens.extend(list(cam_features))
    encoder_in_pos_embed.extend(list(cam_pos_embed))
```

整体流程的形状变化：

| 步骤 | 形状 | 说明 |
|---|---|---|
| 输入图像 | `(1, 3, 360, 640)` | ImageNet 归一化后 |
| ① ResNet18.layer4 | `(1, 512, 12, 20)` | 空间缩小 32×，通道升到 512 |
| ② 位置编码 pos_embed | `(1, 512, 12, 20)` | 每个空间位置的坐标向量 |
| ③ 1×1 Conv | `(1, 512, 12, 20)` | 通道数不变，换表示空间 |
| ④ 展平 | `(240, 1, 512)` | 每个像素变成一个 token |

两路摄像头走完，共 2×240 = **480 个 image tokens**。

---

#### 4.4.1 ① ResNet18 backbone：图像 → 特征图

```python
cam_features = self.backbone(img)["feature_map"]
```

**做了什么**：用 ImageNet 预训练的 ResNet18 卷积网络，把 (360, 640) 的 RGB 图像压缩成 (12, 20) 的特征图，每个位置 512 维。

`self.backbone` 是通过 `IntermediateLayerGetter`（torchvision，[modeling_act.py:702](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L702)）包装的 ResNet18，**只取 layer4 的输出**（不走最后的 avgpool 和 fc 分类头）。

**ResNet18 的管道**：

```
输入 (1, 3, 360, 640)
  ↓ conv1 (7×7, stride=2) + maxpool (3×3, stride=2)  → (1, 64, 90, 160)
  ↓ layer1 (2 个 BasicBlock, stride=1)                → (1, 64, 90, 160)
  ↓ layer2 (2 个 BasicBlock, stride=2)                → (1, 128, 45, 80)
  ↓ layer3 (2 个 BasicBlock, stride=2)                → (1, 256, 23, 40)
  ↓ layer4 (2 个 BasicBlock, stride=2)                → (1, 512, 12, 20)  ← 取这个
```

空间下采样总因子 = $2^5$ = 32，所以输出尺寸 ≈ (360/32, 640/32) = (12, 20)，共 **240 个空间位置**。

**FrozenBatchNorm2d**：ACT 把 ResNet18 里的 BatchNorm 全部替换成 `FrozenBatchNorm2d`（[modeling_act.py:708](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L708)）——不更新 running statistics，等价于固定的仿射变换：

$$
y = \gamma\cdot \frac{x-\mu_{BN}}{\sqrt{\sigma^2_{BN}+\epsilon}} + \beta
$$

推理时这能被编译器 fuse 进前面的 conv，更快。

**每个位置 512 维特征的直观含义**：特征图 (12, 20) 里 (i, j) 位置的 512 维向量，相当于"原图对应那个 32×32 区域对 512 个不同检测器的响应"——某个维度可能代表"有没有竖直边缘"，另一个维度代表"有没有红色物体"，等等。这些"检测器"是 ImageNet 预训练权重自带的。

**计算量**：ResNet18 对 360×640 输入约 12 GFLOPs，两路摄像头共约 **24 GFLOPs**，是整个推理最贵的部分。

---

#### 4.4.2 ② 2D 正弦位置编码：给每个空间位置打"坐标"

```python
cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
```

**做了什么**：给特征图的每个空间位置 (y, x) 生成一个 512 维的"坐标向量"，告诉 Transformer"这个 token 来自图像的第几行第几列"。

**为什么需要**：卷积输出的特征图展平之后，240 个像素 token 对 Transformer 来说是"无序"的——attention 只做加权求和，不关心顺序。位置编码把 (i, j) 的坐标"写"进每个 token 里，让 attention 能利用空间结构（比如"相邻像素应该更相关"）。

**操作分解**（无可学参数，纯计算）：

源码：[`ACTSinusoidalPositionEmbedding2d`, modeling_act.py:1345](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)

对特征图的每个空间位置 $(y, x)$，$y \in [1, 12]$，$x \in [1, 20]$：

1. **归一化坐标**到 $[0, 2\pi]$：
$$\tilde{y} = \frac{y}{12} \cdot 2\pi, \qquad \tilde{x} = \frac{x}{20} \cdot 2\pi$$

2. **生成 256 维频率向量**：$\omega_i = 10000^{2\lfloor i/2\rfloor/256}$

3. **对 y 方向算 256 维 sin/cos**：
$$\text{PE}_y[i] = \begin{cases}\sin(\tilde{y}/\omega_i) & i\text{ 偶}\\ \cos(\tilde{y}/\omega_i) & i\text{ 奇}\end{cases}$$

4. **同理算 x 方向 256 维 PE_x**

5. **拼接**：`pos_embed(y, x) = concat(PE_y, PE_x) ∈ ℝ^512`

**输出形状**：`(1, 512, 12, 20)`，和 cam_features 一样，后面会一起 flatten。

**计算量**：只有 sin/cos 运算，可忽略。

---

#### 4.4.3 ③ 1×1 卷积投影：换"表示空间"

```python
self.encoder_img_feat_input_proj = nn.Conv2d(512, 512, kernel_size=1)
cam_features = self.encoder_img_feat_input_proj(cam_features)
```

**做了什么**：对每个空间位置 (i, j) 的 512 维 ResNet 特征向量，做一次 $512 \times 512$ 的线性变换。

**1×1 Conv 是什么**：kernel_size=1 的卷积等价于"**对每个像素独立做一次 nn.Linear**"。数学上就是：

$$
\text{cam}_{ij}^{\text{out}} = W_{img}\, \text{cam}_{ij}^{\text{in}} + b_{img}, \quad W_{img}\in\mathbb{R}^{512\times 512}
$$

**为什么 512 → 512 还需要投影**：虽然维度没变，但 ResNet18.layer4 的输出是为 **ImageNet 分类**学到的表示空间，和 Transformer 需要的"attention 友好"空间不一样。这个 1×1 Conv 就是一个"**适配器**"——它的权重是 ACT 训练时和 Transformer 一起学出来的，负责把 ResNet 的表示重新编排成 Transformer 能用的形式。

**形状变化**：`(1, 512, 12, 20) → (1, 512, 12, 20)`，形状不变。

**计算量**：$2\cdot 512^2\cdot 12\cdot 20\cdot 2\,\text{cam} \approx 126\,\text{MFLOPs}$。

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

**等价的 PyTorch 原生写法**（便于理解）：

```python
cam_features = cam_features.flatten(2)          # (1, 512, 240)
cam_features = cam_features.permute(2, 0, 1)    # (240, 1, 512)
```

**为什么要展平**：Transformer 处理的是序列，attention 不关心"空间位置是第几行第几列"。所以把 12×20 的 2D 网格变成长度 240 的 1D 序列。失去的空间信息由前面 §4.4.2 的 2D 位置编码补回来（位置编码里已经写好了 (i, j) 坐标）。

---

### 4.5 最终堆叠

```python
encoder_in_tokens   = torch.stack(encoder_in_tokens, axis=0)       # (482, 1, 512)
encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)   # (482, 1, 512)
```

**做了什么**：把前面构造的所有 token（1 latent + 1 state + 480 image）堆成一个长度 482 的序列；对应的位置编码也堆成同样形状。

**token / 位置编码对齐表**：

| 索引 | token 内容 | 位置编码来源 |
|---|---|---|
| `[0]` | latent token | 可学习 1D embedding（`encoder_1d_feature_pos_embed[0]`，[modeling_act.py:744](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L744)） |
| `[1]` | robot state token | 可学习 1D embedding（`encoder_1d_feature_pos_embed[1]`） |
| `[2:242]` | cam0（handeye）240 个像素 token | 2D 正弦编码（§4.4.2） |
| `[242:482]` | cam1（fixed）240 个像素 token | 2D 正弦编码（§4.4.2） |

**位置编码只加到 Q 和 K**：Transformer 里位置编码**不是加到 value 上的**，而是在每层 attention 计算前单独加到 Q 和 K 上（下一节 §5.1 详解）。这是 DETR 的标准做法：Q·K 是"算相关性"的地方，加位置编码让"相邻位置更相关"可学习；V 是"内容本身"，不该被位置信息污染。

---

## 5. 步骤 3：Transformer Encoder（4 层）

源码：[`ACTEncoderLayer`, modeling_act.py:1040](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1040)

### 5.0 这 4 层到底在做什么

Encoder 的输入是上一步构造的 482 个 token，每个 512 维。Encoder 把这 482 个 token 通过 4 层相同结构的 Transformer block "翻来覆去"做两件事：

1. **Self-Attention**：让每个 token 去"看"所有其他 token，按相关性加权平均，融合信息
2. **FFN（前馈网络）**：对每个 token 独立做一次两层 MLP，增加非线性表达

一层结束后每个 token 都"知道"了别人在说什么，4 层叠加之后每个 token 都是融合了全局上下文的特征。

**ACT 用 post-norm**：残差加完后再做 LayerNorm（与原始 Transformer 2017 论文一致，非后来流行的 pre-norm），`pre_norm=False`。

### 5.1 单层完整代码

```python
# ACTEncoderLayer.forward (modeling_act.py:1080)
skip = x
q = k = x + pos_embed                 # Q、K 加位置编码
x = self.self_attn(q, k, value=x)[0]  # 子层①：Self-Attention，V 不加位置编码
x = skip + self.dropout1(x)           # 残差连接
x = self.norm1(x)                     # post-norm LayerNorm

skip = x
x = self.linear2(self.dropout(self.activation(self.linear1(x))))  # 子层②：FFN
x = skip + self.dropout2(x)           # 残差连接
x = self.norm2(x)                     # post-norm LayerNorm

return x
```

第 $\ell$ 层的输入是 $X^{(\ell-1)} \in \mathbb{R}^{482 \times 1 \times 512}$，输出 $X^{(\ell)}$ 同形状。

---

### 5.2 子层①：多头自注意力（Self-Attention）

```python
q = k = x + pos_embed                       # (482, 1, 512)
v = x                                       # (482, 1, 512)  V 不加位置编码
x = self.self_attn(q, k, value=v)[0]        # nn.MultiheadAttention
x = skip + dropout(x)                       # 残差连接
x = self.norm1(x)                           # LayerNorm
```

**做了什么**：让每个 token（482 个）去"询问"所有其他 token，拿到一个加权平均作为自己的新表示。

多头自注意力分 4 步算。

#### Step 1：线性投影生成 Q、K、V

`nn.MultiheadAttention` 内部先对 q/k/v 各做一次线性投影，然后按 H=8 个头切分，每头维度 $d_k = D/H = 64$。

$$
Q_h = (X + P)\, W_Q^h,\quad K_h = (X + P)\, W_K^h,\quad V_h = X\, W_V^h
$$

| | 形状 | 加位置编码？ |
|---|---|---|
| Q_h | `(482, 1, 64)` 每头 | 是 |
| K_h | `(482, 1, 64)` 每头 | 是 |
| V_h | `(482, 1, 64)` 每头 | **否** |

**为什么位置编码只加到 Q 和 K，不加到 V**：
- Q·K^T 是"算相关性"的地方，加位置编码让"附近位置更相关"可学习
- V 是"被加权求和的内容本身"，不该被位置信息污染
- 这是 DETR 的标准做法

#### Step 2：计算注意力分数矩阵

对每个头 $h$：

$$
\text{score}_h = \frac{Q_h K_h^\top}{\sqrt{d_k}} \in \mathbb{R}^{482 \times 482}
$$

- 每一行 = 第 $i$ 个 token 对所有 482 个 token 的相似度分数
- 除以 $\sqrt{d_k}$ 防止分数过大导致 softmax 饱和（梯度消失）

#### Step 3：softmax 归一化 + 加权求和 V

$$
\text{head}_h = \underbrace{\text{softmax}(\text{score}_h)}_{\text{每行和=1}} V_h \in \mathbb{R}^{482 \times 1 \times 64}
$$

- softmax 把每行分数变成概率分布
- 用这个分布对 V 做加权平均，得到每个 token 的新表示

**直观理解**：如果 token $i$ 和 token $j$ 的分数很高（softmax 后权重大），那么 $i$ 的新表示里就会包含更多 $j$ 的信息。

#### Step 4：合并多头 + 输出投影

$$
\text{MHA}(X) = \underbrace{\text{concat}(\text{head}_1, \dots, \text{head}_8)}_{(482, 1, 512)}\, W_O, \quad W_O \in \mathbb{R}^{512 \times 512}
$$

8 个头并行计算出的结果（每头 64 维）拼成 512 维，再过一个 $W_O$ 输出投影。

#### Step 5：残差 + LayerNorm

```python
x = skip + dropout(x)    # 残差
x = self.norm1(x)        # LayerNorm
```

$$
X' = \text{LayerNorm}(X + \text{MHA}(X))
$$

- **残差连接**：让梯度能直接"跳"回输入，避免深层网络退化
- **LayerNorm**：对每个 token 的 512 维独立做"均值 0 方差 1"归一化，稳定训练

#### 本子层 FLOPs（单层，B=1）

| 子步骤 | 计算 | FLOPs |
|---|---|---|
| Q/K/V 投影 | `3 × 2 × 482 × 512²` | ~758 M |
| Q·K^T | `2 × 8 × 482² × 64` | ~238 M |
| softmax · V | `2 × 8 × 482² × 64` | ~238 M |
| 输出投影 W_O | `2 × 482 × 512²` | ~253 M |
| **小计** | | **≈ 1.49 GFLOPs** |

---

### 5.3 子层②：前馈网络（FFN）

```python
x = self.linear2(self.dropout(self.activation(self.linear1(x))))
x = skip + dropout(x)
x = self.norm2(x)
```

**做了什么**：对每个 token **独立**做一次两层 MLP（先扩维再压回），增加模型的非线性表达能力。

**操作分解**：

$$
\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2
$$

| 步骤 | 形状变化 | 说明 |
|---|---|---|
| `self.linear1` | `512 → 3200` | `W_1 ∈ ℝ^{3200×512}`，dim_feedforward=3200 |
| `self.activation`（ReLU） | `3200 → 3200` | 逐元素非线性 |
| `self.linear2` | `3200 → 512` | `W_2 ∈ ℝ^{512×3200}`，压回原维度 |

**为什么要先扩再压**：在高维空间更容易学到复杂的模式（有更多"计算空间"），再压回来继续传递。这是 Transformer 的"扩展-收缩"范式，类似于瑞士军刀里的可折叠工具。

**和 self-attention 的对比**：

| | self-attention | FFN |
|---|---|---|
| token 间交互 | 有（全局加权求和） | 无（每个 token 独立） |
| 主要作用 | 融合上下文 | 增加非线性表达 |

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

**关键细节**：位置编码 `pos_embed` 每一层都用**原始的那份**，不随 `x` 一起更新。这是因为每一层都要在 Q/K 处重新加一次位置编码，如果用更新后的 x，位置信息会被稀释或丢失。

**总计算量**：

| 每层 | FLOPs |
|---|---|
| 子层① Self-Attention | ~1.49 G |
| 子层② FFN | ~3.16 G |
| **单层合计** | **~4.65 G** |
| **4 层合计** | **~18.6 G** |

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

源码：[`ACTDecoderLayer`, modeling_act.py:1180](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1180)

### 6.0 这一层到底在做什么

Decoder 的任务：**从 Encoder 的 482 个上下文特征中，"读取" 100 个未来时刻需要的信息**，输出 100 个动作特征（再经过一层 nn.Linear 就是 100 帧动作了）。

**核心 trick（来自 DETR）**：
- Decoder 输入是**全零**的占位 tensor，形状 `(100, 1, 512)`
- 真正的"问题"由 **100 个可学习的 object queries**（`decoder_pos_embed`）携带
- 第 t 个 query 的含义：「未来第 t 步的动作，应该关注 encoder 的哪些信息？」

### 6.1 单层完整代码（3 个子层）

```python
# ACTDecoderLayer.forward (modeling_act.py:1238)
skip = x
q = k = x + decoder_pos_embed        # 子层①：Query Self-Attention
x = self.self_attn(q, k, value=x)[0]
x = skip + dropout(x); x = self.norm1(x)

skip = x
x = self.multihead_attn(             # 子层②：Cross-Attention (queries ↔ encoder)
    query = x + decoder_pos_embed,
    key   = encoder_out + encoder_pos_embed,
    value = encoder_out,
)[0]
x = skip + dropout(x); x = self.norm2(x)

skip = x
x = self.linear2(dropout(activation(self.linear1(x))))  # 子层③：FFN
x = skip + dropout(x); x = self.norm3(x)
```

---

### 6.2 Decoder 输入初始化

```python
decoder_in = torch.zeros((T=100, B=1, D=512))                       # 全零占位
decoder_pos_embed = self.decoder_pos_embed.weight.unsqueeze(1)      # (100, 1, 512) 可学习
```

| 张量 | 形状 | 内容 |
|---|---|---|
| `decoder_in` (x) | `(100, 1, 512)` | 全零 |
| `decoder_pos_embed` | `(100, 1, 512)` | 100 个可学习的 "object queries" |

**为什么输入是全零**：所有"问什么"的信息由 `decoder_pos_embed` 携带。每个 query $q_t$ 是训练出来的"未来第 t 步的提问模板"——比如 $q_0$ 可能代表"抓取前一瞬间要看哪儿"，$q_{99}$ 可能代表"抓取完成后要看哪儿"。这 100 个 query 向量在训练时学出来，推理时固定。

---

### 6.3 子层①：Query 自注意力（推理时近似无效）

```python
skip = x                                    # x 初始为 0
q = k = x + decoder_pos_embed               # q = k = decoder_pos_embed
x = self.self_attn(q, k, value=x)[0]        # value = 0 → 输出 = 0
x = skip + dropout(x)                       # 仍 = 0
x = self.norm1(x)                           # LayerNorm(0) = β
```

**做了什么**：理论上让 100 个 query 互相交换信息，但**第一层里近似"不起作用"**。

**分析为什么近似无效**：

| 步骤 | 值 |
|---|---|
| `value = x` | **0**（因为 x 还是全零占位） |
| `softmax · V = softmax · 0` | **0** |
| `skip + 0` | **0** |
| `norm1(0)` | **β**（LayerNorm 的偏置项） |

所以这一子层输出一个**常数向量 β**（不依赖任何输入）。真正起作用的信息流在下一步 cross-attention。

**如果 `n_decoder_layers > 1` 会怎样**：从第二层起 `x` 就不再是 0 了（携带上一层 cross-attention 的结果），这个子层才真正生效。但 ACT 默认 `n_decoder_layers=1`（保留原论文代码的历史设定，[configuration_act.py:166-168](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L166-L168)），所以这个子层在 ACT 推理中实际上是**空跑**。

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
```

**做了什么**：每个 query $q_t$（t=0..99）从 encoder 的 482 个 token 里"按相关性挑选"信息，加权平均形成自己的特征。

**和 encoder self-attention 的区别**：

| | Encoder Self-Attention | Decoder Cross-Attention |
|---|---|---|
| Q 来源 | 自己（482 个 token） | decoder（100 个 queries）|
| K、V 来源 | 自己（482 个 token） | encoder（482 个 token）|
| 用途 | 让 encoder 内部互相交流 | 让 decoder 从 encoder"读取"信息 |

**数学**：

$$
Q = (Y^{(0)}+P_{dec})\,W_Q,\quad K = (H_{enc}+P_{enc})\,W_K,\quad V = H_{enc}\,W_V
$$

$$
\text{CrossAttn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**形状变化**：

| | 形状 |
|---|---|
| Q | `(100, 1, 512)` |
| K、V | `(482, 1, 512)` |
| `QK^T` | `(100, 482)` 每行是"第 t 个 query 对 encoder 482 个位置的注意力分布" |
| softmax · V | `(100, 1, 512)` 每个 query 的新表示 |

**直观理解**：
- 第 20 个 query $q_{20}$ 可能学会"重点看 cam0 左上角区域 + 关节 state token"
- 第 80 个 query $q_{80}$ 可能学会"重点看 cam1 中间区域"
- 这些关注模式都是训练出来的，编码了"任务执行到第 t 步时应该看哪里"

#### FLOPs（单层，B=1）

| 子步骤 | 计算 | FLOPs |
|---|---|---|
| Q 投影（长度 100） | `2 × 100 × 512²` | ~52 M |
| K 投影（长度 482） | `2 × 482 × 512²` | ~253 M |
| V 投影（长度 482） | `2 × 482 × 512²` | ~253 M |
| Q·K^T | `2 × 8 × 100 × 482 × 64` | ~49 M |
| softmax · V | 同上 | ~49 M |
| 输出投影 W_O | `2 × 100 × 512²` | ~52 M |
| **小计** | | **≈ 0.71 GFLOPs** |

---

### 6.5 子层③：FFN

```python
skip = x
x = self.linear2(dropout(activation(self.linear1(x))))
x = skip + dropout(x)
x = self.norm3(x)
```

和 encoder 的 FFN 结构**完全一致**（`512 → 3200 → ReLU → 512`），只是序列长度变成 T=100：

$$
\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2
$$

**FLOPs**：$4 \cdot 100 \cdot 512 \cdot 3200 \approx 0.66\,\text{GFLOPs}$

---

### 6.6 Decoder 最终输出

```python
decoder_out = self.decoder.norm(x)          # (100, 1, 512)，最终 LayerNorm
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

源码：[modeling_act.py:975-977](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L975-L977)

```python
self.action_head = nn.Linear(D=512, A=6)
actions = self.action_head(decoder_out)     # (1, 100, 512) → (1, 100, 6)
```

**做了什么**：用一个 `nn.Linear(512 → 6)` 把每个 512 维的动作特征投影到 6 维（= SO101 的 6 个关节角度）。

**操作分解**：

| | 值 |
|---|---|
| `self.action_head` | `nn.Linear(dim_model=512, action_dim=6)` |
| 输入 H_dec | `(1, 100, 512)` |
| 输出 actions | `(1, 100, 6)` |

**数学**：

$$
\hat{a}_t = W_a\, H_{dec,t} + b_a,\quad W_a\in\mathbb{R}^{6\times 512},\quad t=0,\dots,99
$$

100 个时刻**共享同一组权重** $W_a$（batched matmul，一次算完 100 帧）。

**输出形状含义**：

| 维度 | 大小 | 含义 |
|---|---|---|
| 0 | 1 | batch（推理时 B=1） |
| 1 | 100 | 未来 100 帧（chunk_size） |
| 2 | 6 | 每帧 6 个关节的**归一化**目标角度 |

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

| 组件 | FLOPs | 说明 |
|---|---|---|
| ResNet18 × 2 摄像头 | **~24 G** | 360×640，主导 |
| 1×1 Conv 投影 (512→D) × 2 | ~0.13 G | |
| 2D 正弦 pos embed | ~0 | 无乘法密集计算 |
| Encoder 4 层 (L=482) | **~18.6 G** | 其中 FFN 占 ~12.6 G，self-attn 占 ~6 G |
| Decoder 1 层 (Q=100, KV=482) | **~1.37 G** | cross-attn 0.71 + FFN 0.66 |
| Action head | ~0.6 M | 忽略 |
| **合计** | **~44 GFLOPs** | ResNet 和 encoder FFN 是两个大头 |

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

| 被跳过的东西 | 原因 |
|---|---|
| VAE encoder（4 层 Transformer，输入 1+1+100 token）| `self.training=False`，走 else 分支，`latent = 0` |
| VAE 的 CLS token 投影、重参数化采样 | 同上 |
| L1 loss、KL loss 计算 | 推理不算损失 |
| Dropout | `eval()` 模式下等价于恒等 |
| BatchNorm 统计量更新 | `FrozenBatchNorm2d` 本就不更新 |

另外，**整个前向在 `@torch.no_grad()` 下执行**，所有中间 tensor 都不保留 grad_fn，也不会写 autograd buffer，这对峰值内存和 cache 友好度都有明显帮助。

---

## 12. 交叉引用

- 源码主体：[`ACT.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782)
- 配置默认值：[`ACTConfig`](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L37)
- Encoder 层：[`ACTEncoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1080)
- Decoder 层：[`ACTDecoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1238)
- 2D 正弦位置编码：[`ACTSinusoidalPositionEmbedding2d`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)
- 调用点：`ACTPolicy.select_action → predict_action_chunk`，[modeling_act.py:224](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L224)
- 对应论文：Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023

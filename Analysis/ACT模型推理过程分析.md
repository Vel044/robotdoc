# ACT (Action Chunking Transformer) 推理过程分析

> 本文档详细分析 ACT (Action Chunking Transformer) 模型的推理过程。
>
> 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://huggingface.co/papers/2304.13705)
> 代码: [tonyzhaozh/act](https://github.com/tonyzhaozh/act)

---

## 1. ACT 模型概述

### 1.1 核心思想

ACT 的核心思想是 **动作分块 (Action Chunking)** + **Transformer 序列建模**：

1. **动作分块**: 每次推理预测一段动作序列（如100个动作），而不是单个动作
2. **时序平滑**: 使用时序集成（Temporal Ensembling）对跨时间步的预测进行加权平均
3. **观测编码**: 使用 ResNet backbone 编码图像，使用线性层编码关节状态
4. **Transformer 解码**: 使用 cross-attention 机制融合观测信息

### 1.2 推理 vs 训练

| 阶段 | VAE Encoder | Latent | 动作输出 |
|------|-------------|--------|---------|
| 训练 | 使用 | 从动作序列采样 | 重建动作 |
| 推理 | 不使用 | 全零向量 | 直接预测 |

---

## 2. 推理调用链

```
select_action(batch)              [modeling_act.py:99]
    │
    ├── temporal_ensemble_coeff != None?
    │   ├── Yes: predict_action_chunk() → temporal_ensembler.update()
    │   └── No: predict_action_chunk() → action_queue
    │
    ▼
predict_action_chunk(batch)       [modeling_act.py:124]
    │
    ▼
model.forward(batch)               [modeling_act.py:377]
    ├── Step 1: 图像特征提取 (ResNet backbone)
    ├── Step 2: 状态特征投影 (Linear)
    ├── Step 3: Transformer Encoder
    ├── Step 4: Transformer Decoder (cross-attention)
    └── Step 5: 动作预测头
    │
    ▼
actions: [B, chunk_size, action_dim]
```

---

## 3. 入口函数: `select_action`

**文件:** `lerobot/src/lerobot/policies/act/modeling_act.py:99`

```python
@torch.no_grad()
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    """给定环境观测,选择一个动作执行

    这是对外的推理接口,每次调用返回一个动作。
    内部通过 action_queue 管理动作块,只在队列空时调用模型推理。
    """
    self.eval()

    # ===== 时序集成路径 =====
    if self.config.temporal_ensemble_coeff is not None:
        actions = self.predict_action_chunk(batch)
        action = self.temporal_ensembler.update(actions)
        return action

    # ===== 普通路径 =====
    if len(self._action_queue) == 0:
        # 预测 chunk_size 个动作,只取前 n_action_steps 个
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
        # 转置: (B, n_action_steps, action_dim) -> (n_action_steps, B, action_dim)
        self._action_queue.extend(actions.transpose(0, 1))
    return self._action_queue.popleft()
```

**参数:**
- `batch`: 包含 `"observation.images"` (图像列表) 和 `"observation.state"` (关节状态) 的字典

**返回值:**
- `Tensor`: shape=[B, action_dim], 单个动作

---

## 4. 核心推理: `predict_action_chunk`

**文件:** `lerobot/src/lerobot/policies/act/modeling_act.py:124`

```python
@torch.no_grad()
def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
    """给定环境观测,预测一段动作块"""
    self.eval()

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    actions = self.model(batch)[0]
    return actions
```

**返回值:**
- `Tensor`: shape=[B, chunk_size, action_dim], 预测的动作序列

---

## 5. 模型前向传播: `ACT.forward`

**文件:** `lerobot/src/lerobot/policies/act/modeling_act.py:377`

### 5.1 输入数据结构

```python
batch = {
    "observation.images": [
        Tensor[B, 3, 224, 224],  # 相机1图像
        Tensor[B, 3, 224, 224],  # 相机2图像
        ...
    ],
    "observation.state": Tensor[B, state_dim],  # 机器人关节状态
    # (可选)
    "observation.environment_state": Tensor[B, env_dim],
}
```

### 5.2 推理流程详解

#### Step 1: 准备 Latent (推理时全零)

```python
# 推理时: VAE encoder 不使用, latent 设为全零向量
latent_sample = torch.zeros([batch_size, self.config.latent_dim])
# latent_dim 默认=32
```

#### Step 2: 图像特征提取 (ResNet Backbone)

```python
for img in batch["observation.images"]:
    # img: (B, 3, H, W)
    cam_features = self.backbone(img)["feature_map"]
    # backbone: ResNet18/34/50
    # feature_map: (B, 512, H', W') 或 (B, 2048, H', W')

    # 添加 2D 正弦位置编码
    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)

    # 投影到 dim_model 维度
    cam_features = self.encoder_img_feat_input_proj(cam_features)
    # cam_features: (B, dim_model, H', W')

    # 重排列: (B, D, H', W') -> (H'*W', B, D)
    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
```

**ResNet backbone 详情:**
```python
# 配置
vision_backbone: "resnet18"  # 或 resnet34, resnet50
pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"

# 结构
backbone = torchvision.models.resnet18(weights=...)
# 去掉最后的 FC 层,用 IntermediateLayerGetter 取 layer4 输出
# layer4 输出: (B, 512, H/32, W/32) for ResNet18
```

#### Step 3: 状态特征投影

```python
# Robot state: (B, state_dim) -> (B, dim_model)
robot_state_embed = self.encoder_robot_state_input_proj(batch["observation.state"])

# latent: (B, latent_dim) -> (B, dim_model)
latent_embed = self.encoder_latent_input_proj(latent_sample)
```

#### Step 4: 构造 Encoder Tokens

```python
# Encoder 输入 tokens: [latent, robot_state, image_pixels]
encoder_in_tokens = [
    latent_embed,           # (B, dim_model)
    robot_state_embed,       # (B, dim_model)
    ...image_pixels...,      # (H'*W', B, dim_model) 每个像素是一个token
]
# 最终: (seq_len, B, dim_model)

# 位置编码
encoder_in_pos_embed = [
    self.encoder_1d_feature_pos_embed.weight[:1],  # latent位置
    self.encoder_1d_feature_pos_embed.weight[1:2],  # robot_state位置
    ...image_pos_embeds...,  # 每个像素位置
]
```

#### Step 5: Transformer Encoder

```python
# Transformer Encoder: 4层,标准self-attention + FFN
encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
# encoder_out: (seq_len, B, dim_model)
```

**Encoder 结构:**
```python
ACTEncoder:
    - 4层 ACTEncoderLayer
    - 每层: MultiHeadSelfAttention + LayerNorm + FeedForward + LayerNorm
    - 配置: dim_model=512, n_heads=8, dim_feedforward=3200
```

#### Step 6: Transformer Decoder (Cross-Attention)

```python
# Decoder 输入: learnable query tokens (初始为全零)
decoder_in = torch.zeros((chunk_size, B, dim_model))
# chunk_size=100, 即预测100个动作

# Decoder: cross-attention 到 encoder 输出
decoder_out = self.decoder(
    decoder_in,           # queries: (100, B, D)
    encoder_out,          # keys/values: (seq_len, B, D)
    encoder_pos_embed,    # encoder位置编码
    decoder_pos_embed,    # decoder位置编码(learnable)
)
# decoder_out: (100, B, dim_model)
```

**Decoder 结构:**
```python
ACTDecoder:
    - 1层 ACTDecoderLayer (原ACT代码有bug,实际只用第1层)
    - 每层: SelfAttention + LayerNorm + CrossAttention + LayerNorm + FeedForward + LayerNorm
```

#### Step 7: 动作预测

```python
# 转置: (100, B, D) -> (B, 100, D)
decoder_out = decoder_out.transpose(0, 1)

# 动作头: dim_model -> action_dim
actions = self.action_head(decoder_out)
# actions: (B, chunk_size, action_dim)
# action_dim = 6 for SO101 (6个关节)
```

---

## 6. 时序集成: `ACTTemporalEnsembler`

**文件:** `lerobot/src/lerobot/policies/act/modeling_act.py:164`

### 6.1 核心思想

对当前时刻之前预测的动作序列进行**指数加权平均**,得到更平滑的动作输出。

### 6.2 权重计算

```
wᵢ = exp(-temporal_ensemble_coeff * i)

其中 i=0 是最旧的动作, i=chunk_size-1 是最新的动作
```

**示例 (temporal_ensemble_coeff=0.01):**
```python
chunk_size = 10
coeff = 0.01
weights = exp(-coeff * torch.arange(chunk_size))
# tensor([1.0000, 0.9900, 0.9802, 0.9704, 0.9608, ...])
# 旧动作权重更高
```

### 6.3 在线更新算法

```python
def update(self, actions: Tensor) -> Tensor:
    """
    actions: (B, chunk_size, action_dim)
    返回: (B, action_dim), 当前时刻的集成动作
    """
    if self.ensembled_actions is None:
        # 第一次: 直接使用预测的动作
        self.ensembled_actions = actions.clone()
    else:
        # 在线更新: 对前 chunk_size-1 个动作进行加权平均
        # w_i * old_avg + (1-w_i) * new_action
        self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
        self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
        self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
        # 拼接最新的动作
        self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)

    # 取出第一个动作返回,并更新状态
    action = self.ensembled_actions[:, 0]
    self.ensembled_actions = self.ensembled_actions[:, 1:]
    self.ensembled_actions_count -= 1
    return action
```

---

## 7. 推理模式下的数据流

```
输入 batch
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ predict_action_chunk(batch)                                 │
│     │                                                      │
│     ▼                                                      │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ 图像列表 (如 [img1, img2])                              │  │
│ │     │                                                  │  │
│ │     ▼                                                  │  │
│ │ ResNet18 backbone                                      │  │
│ │     │                                                  │  │
│ │     ▼                                                  │  │
│ │ feature_map: (B, 512, 7, 7) × 2相机                   │  │
│ │     │                                                  │  │
│ │     ▼                                                  │  │
│ │ 2D Sinusoidal Pos Embed                               │  │
│ │     │                                                  │  │
│ │     ▼                                                  │  │
│ │ 投影 + 重排: (B, 512, 7, 7) -> (49, B, 512) × 2       │  │
│ └───────────────────────────────────────────────────────┘  │
│     │                                                      │
│     ▼                                                      │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ 关节状态 (B, 6)                                        │  │
│ │     │                                                  │  │
│ │     ▼                                                  │  │
│ │ Linear(6 -> 512)                                      │  │
│ │     │                                                  │
│ │     ▼                                                  │  │
│ │ state_embed: (B, 512)                                  │  │
│ └───────────────────────────────────────────────────────┘  │
│     │                                                      │
│     ▼                                                      │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ Latent (推理时全零)                                    │  │
│ │ Linear(32 -> 512)                                     │  │
│ └───────────────────────────────────────────────────────┘  │
│     │                                                      │
│     ▼                                                      │
│ 拼接: [latent_embed, state_embed, img1_tokens(49), img2_tokens(49)] │
│      = (seq_len=101, B, 512)                              │
│     │                                                      │
│     ▼                                                      │
│ Transformer Encoder (4层)                                 │
│     │                                                      │
│     ▼                                                      │
│ encoder_out: (101, B, 512)                                │
│     │                                                      │
│     ▼                                                      │
│ Transformer Decoder (1层)                                  │
│     │ Decoder queries: 全零 (100, B, 512)                │
│     │ Cross-attention to encoder_out                     │
│     │                                                      │
│     ▼                                                      │
│ decoder_out: (100, B, 512)                                │
│     │                                                      │
│     ▼                                                      │
│ action_head: Linear(512 -> 6)                             │
│     │                                                      │
│     ▼                                                      │
│ actions: (B, 100, 6)  ← 100个动作,每个6维(6个关节)        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
select_action:
    │
    ├── temporal_ensemble=True?
    │   └── Yes: 时序集成,返回加权平均后的动作
    │
    └── temporal_ensemble=False?
        └── No: action_queue管理,每次返回队列头部一个动作
```

---

## 8. 关键配置参数

**文件:** `lerobot/src/lerobot/policies/act/configuration_act.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 100 | 每次推理预测的动作数量 |
| `n_action_steps` | 100 | 实际执行的动作数量 (≤chunk_size) |
| `dim_model` | 512 | Transformer 隐藏层维度 |
| `n_heads` | 8 | Multi-head attention 的头数 |
| `dim_feedforward` | 3200 | FFN 隐藏层维度 |
| `n_encoder_layers` | 4 | Encoder 层数 |
| `n_decoder_layers` | 1 | Decoder 层数 (原ACT有bug,实际只用1层) |
| `use_vae` | True | 是否使用 VAE (推理时=False) |
| `latent_dim` | 32 | VAE 潜在空间维度 |
| `vision_backbone` | "resnet18" | 图像backbone |
| `temporal_ensemble_coeff` | None | 时序集成系数 (None=不使用) |

---

## 9. 源码文件路径

| 文件 | 说明 |
|------|------|
| `lerobot/src/lerobot/policies/act/modeling_act.py` | ACT模型实现 |
| `lerobot/src/lerobot/policies/act/configuration_act.py` | 配置文件 |
| `lerobot/src/lerobot/policies/act/processor_act.py` | 预处理/后处理 |
| `lerobot/src/lerobot/policies/pretrained.py` | PreTrainedPolicy基类 |

---

## 10. 与 record_loop 的对接

```
record_loop() 中的推理阶段
    │
    ▼
predict_action(observation_frame, policy, ...)
    │
    ▼
preprocessor(observation)
    │
    ▼
policy.select_action(observation)
    │  ← ACTPolicy.select_action()
    │
    ├── action_queue 有内容? → 直接返回队列头部
    │
    └── action_queue 为空?
            │
            ▼
        predict_action_chunk(batch)
            │
            ▼
        model.forward(batch)
            │
            ▼
        actions: (B, chunk_size, action_dim)
            │
            ▼
    postprocessor(action)
        │
        ▼
    返回单个 action 给 robot.send_action()
```

---

## 11. PyTorch 计算详解

### 11.1 Transformer Encoder 计算

```python
# 输入: (seq_len, batch, dim_model) = (101, B, 512)

# Self-Attention
q = k = x + pos_embed  # (101, B, 512)
v = x                  # (101, B, 512)
attn_output, _ = nn.MultiheadAttention(512, 8)(q, k, v)
# attn_output: (101, B, 512)

# FFN
x = nn.Linear(512, 3200)(x)
x = F.relu(x)
x = nn.Linear(3200, 512)(x)
# x: (101, B, 512)
```

### 11.2 Transformer Decoder 计算

```python
# Decoder 输入 (queries): 全零
decoder_in: (100, B, 512)

# Self-Attention (decoder自己的自注意力)
q = k = decoder_in + decoder_pos_embed
self_attn_out, _ = nn.MultiheadAttention(512, 8)(q, k, decoder_in)
# self_attn_out: (100, B, 512)

# Cross-Attention (看encoder输出)
q = self_attn_out + decoder_pos_embed  # decoder侧
k = v = encoder_out + encoder_pos_embed  # encoder侧
cross_attn_out, _ = nn.MultiheadAttention(512, 8)(q, k, v)
# cross_attn_out: (100, B, 512)

# FFN
ffn_out = FFN(cross_attn_out)
# ffn_out: (100, B, 512)
```

### 11.3 动作头计算

```python
# 输入: (B, 100, 512)
# 输出: (B, 100, 6)
actions = nn.Linear(512, 6)(decoder_out)
```

---

## 12. 总结

ACT 推理过程:
1. **图像编码**: ResNet backbone 提取图像特征,添加 2D 位置编码
2. **状态编码**: 线性层投影关节状态
3. **Latent**: 推理时为全零向量,不使用 VAE encoder
4. **Encoder**: Transformer encoder 融合所有观测信息
5. **Decoder**: Transformer decoder (cross-attention) 生成动作序列
6. **输出**: 动作头映射到动作空间 (6维 for SO101)

关键设计:
- **动作分块**: 一次预测多个动作,提高时序一致性
- **时序集成**: 对跨时间步预测加权平均,提高平滑性
- **Cross-attention**: Decoder 通过 cross-attention 访问 Encoder 的输出,实现条件生成

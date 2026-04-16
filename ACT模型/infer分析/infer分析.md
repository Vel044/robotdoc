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
| `B` | batch size | `1` |
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

## 1. 顶层调用链

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

### 1.1 推理时 `ACT.forward` 完整执行的代码

下面这段是把 [modeling_act.py:782-979](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782-L979) 的 `ACT.forward` 按 **SO101 + 2 摄像头 + `eval()`** 这次命令的实际配置"展平"的结果：所有永远走不到的训练分支、`env_state` 分支、`else batch_size` 分支、训练期 `assert` 全部删除，剩下的每一行在推理时都会被真实执行。后面的 §2–§6 只对这段代码做数学和 FLOPs 展开，不再重复贴代码。

```python
# 关闭梯度计算，因为推理时不需要反向传播
@torch.no_grad()
def forward(self, batch):
    # 确定 batch_size（推理时 batch 里有 "observation.images"）
    batch_size = batch["observation.images"][0].shape[0]

    # ── 步骤 1：latent（推理时直接全零，不走 VAE encoder） ─────────
    mu = log_sigma_x2 = None
    latent_sample = torch.zeros(
        [batch_size, self.config.latent_dim],
        dtype=torch.float32,
    ).to(batch["observation.state"].device)

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
        (self.config.chunk_size, batch_size, self.config.dim_model),
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

## 2. 步骤 1：latent（推理时为全零）

### 2.1 数学表达



$$
z = \mathbf{0} \in \mathbb{R}^{B \times Z} = \mathbb{R}^{1 \times 32}
$$

### 2.2 VAE 在推理时到底用不用得到？—— **完全用不到**

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

## 3. 步骤 2：构造 Encoder 输入 token

源码：[modeling_act.py:898-948](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L898-L948)

Encoder 的输入是一个序列：

$$
X_{enc}^{(0)} = [\underbrace{x_{latent}}_{1},\; \underbrace{x_{state}}_{1},\; \underbrace{x_{cam0,1}, \dots, x_{cam0,240}}_{240},\; \underbrace{x_{cam1,1}, \dots, x_{cam1,240}}_{240}] \in \mathbb{R}^{L_{enc\_seq} \times B \times D}
$$

其中 $L_{enc\_seq} = 482$。下面 3.1 / 3.2 / 3.3 / 3.4 小节分别对 latent token、robot state token、图像 token 和最终堆叠做数学细节展开（代码已在 §1.1 一并贴过）。

### 3.1 Latent token

```python
encoder_in_tokens = [self.encoder_latent_input_proj(z)]       # (1, B, D)
```

线性投影：

$$
x_{latent} = W_{lat}\, z + b_{lat}, \quad W_{lat} \in \mathbb{R}^{D \times Z},\; D{=}512,\; Z{=}32
$$

因 $z=\mathbf{0}$，结果恰好是偏置 $b_{lat}$。**FLOPs ≈ 2·D·Z ≈ 33k**（可忽略）。

### 3.2 Robot state token

```python
encoder_in_tokens.append(
    self.encoder_robot_state_input_proj(batch["observation.state"])
)                                                             # (B, D) → append
```

$$
x_{state} = W_{state}\, s + b_{state}, \quad s \in \mathbb{R}^{B \times S},\; W_{state} \in \mathbb{R}^{D \times S}
$$

即把 6 维的关节位置 $s$ 投影到 512 维。注意 `observation.state` 已经经过 `normalize_inputs`（`MEAN_STD` 归一化）。

### 3.3 图像 token（每路摄像头）

对 `handeye` 和 `fixed` 两路图像依次做：

```python
cam_features = self.backbone(img)["feature_map"]               # ResNet18.layer4
cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)  # 2D 正弦位置编码
cam_features = self.encoder_img_feat_input_proj(cam_features)  # 1×1 Conv: 512→D
cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
```

源码：[modeling_act.py:919-943](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L919-L943)

#### 3.3.1 ResNet18 backbone

用 [IntermediateLayerGetter](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L702)（torchvision）只拿到 `layer4` 的输出。ResNet18 的管道：

$$
\text{img} \xrightarrow{\text{conv1 7×7, s2}} \xrightarrow{\text{maxpool s2}} \xrightarrow{\text{layer1}} \xrightarrow{\text{layer2, s2}} \xrightarrow{\text{layer3, s2}} \xrightarrow{\text{layer4, s2}} \text{feature map}
$$

空间下采样总因子 $=2^5=32$：

$$
\underbrace{(B,3,360,640)}_{\text{input}} \longrightarrow \underbrace{(B,512,12,20)}_{\text{layer4 output}}
$$

每一层都是 `conv → FrozenBatchNorm2d → ReLU` 的组合，对 BasicBlock 还有残差。推理阶段 BN 已经被 fuse 成仿射：

$$
y = \gamma\cdot \frac{x-\mu_{BN}}{\sqrt{\sigma^2_{BN}+\epsilon}} + \beta
$$

这在 `FrozenBatchNorm2d` 里写死、不更新统计量，因此它等价于一个带仿射的 1×1 线性层，能被编译器融合进 conv。

**近似 FLOPs（ResNet18，输入 360×640）**：约 `12 GFLOPs` × 2 摄像头 = **~24 GFLOPs**，是推理里最"重"的部分。

#### 3.3.2 2D 正弦位置编码

源码：[`ACTSinusoidalPositionEmbedding2d`, modeling_act.py:1345](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)

对特征图的每个空间位置 $(y,x)\in[1,h]\times[1,w]$，先归一化到 $[0,2\pi]$：

$$
\tilde{y} = \frac{y}{h} \cdot 2\pi, \qquad \tilde{x} = \frac{x}{w} \cdot 2\pi
$$

然后生成 `dim_model/2 = 256` 维的频率向量 $\omega_i = 10000^{2\lfloor i/2\rfloor/d},\ d{=}256$，对 y/x 分别计算：

$$
\text{PE}_y[i] = \begin{cases}\sin(\tilde{y}/\omega_i) & i\text{ even}\\ \cos(\tilde{y}/\omega_i) & i\text{ odd}\end{cases}
\quad\text{同理 } \text{PE}_x
$$

最后 $\text{pos\_embed} = \text{concat}(\text{PE}_y, \text{PE}_x) \in \mathbb{R}^{1\times D\times h\times w}$，与 cam_features 同形状。**这一部分无可学参数**，计算量可忽略。

#### 3.3.3 1×1 卷积投影：512 → D

```python
self.encoder_img_feat_input_proj = nn.Conv2d(512, D=512, kernel_size=1)
```

数学上就是对每个空间位置 $(i,j)$ 做一个 $D\times C_{res}$ 矩阵乘：

$$
\text{cam}_{ij}^{\text{out}} = W_{img}\, \text{cam}_{ij}^{\text{in}} + b_{img}, \quad W_{img}\in\mathbb{R}^{D\times C_{res}}=\mathbb{R}^{512\times 512}
$$

**FLOPs**：$2 \cdot D\cdot C_{res} \cdot h\cdot w \cdot N_{cam} \approx 2\cdot 512^2\cdot 240\cdot 2 \approx 126\,\text{MFLOPs}$。

#### 3.3.4 展平为 token 序列

```python
cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
# (B, 512, 12, 20) → (240, B, 512)
```

把 240 个像素变成 240 个 token。两路摄像头拼到一起就是 480 个 image tokens。

### 3.4 最终堆叠

```python
encoder_in_tokens   = torch.stack(encoder_in_tokens, axis=0)       # (482, B, D)
encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)   # (482, 1, D)
```

位置编码由两部分拼成：

- 前 2 个 1D token（latent、state）用**可学习嵌入** `self.encoder_1d_feature_pos_embed.weight`（[modeling_act.py:744](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L744)）
- 后 480 个 2D token 用上面的正弦编码

注意 Transformer 里位置编码不是加到 value 上的，而是在每层的 Q/K 前加一次（见下一节）。

---

## 4. 步骤 3：Transformer Encoder（4 层）

源码：[`ACTEncoderLayer`, modeling_act.py:1040](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1040)

`pre_norm=False`（post-norm 模式，与原始 Transformer 一致）。第 $\ell$ 层的输入记作 $X^{(\ell-1)}\in\mathbb{R}^{L_{enc\_seq}\times B\times D}$，位置编码记作 $P\in\mathbb{R}^{L_{enc\_seq}\times 1\times D}$。

### 4.1 多头自注意力子层

```python
q = k = x + pos_embed                       # (482, B, D)
v = x                                       # (482, B, D)  注意：V 不加 pos
x = self.self_attn(q, k, value=v)[0]        # nn.MultiheadAttention
x = skip + dropout(x)                       # 残差
x = self.norm1(x)                           # post-norm 的 LayerNorm
```

数学上，对每个 head $h\in[1,H]$：

$$
Q_h = (X^{(\ell-1)}+P)\, W_Q^h,\quad K_h = (X^{(\ell-1)}+P)\, W_K^h,\quad V_h = X^{(\ell-1)}\, W_V^h
$$

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h,\quad d_k = D/H = 64
$$

$$
\text{MHA}(X^{(\ell-1)}) = \text{concat}(\text{head}_1,\dots,\text{head}_H)\, W_O
$$

$$
X' = \text{LayerNorm}\!\left(X^{(\ell-1)} + \text{MHA}(X^{(\ell-1)})\right)
$$

**关键点**：位置编码**只加到 Q 和 K，不加到 V**。这样注意力分数"看到"位置信息，但被加权求和的内容仍是原始 token。这是 DETR 的标准做法。

**FLOPs 估算**（单层、推理 `B=1`）：

- QKV 投影：$3\cdot 2\cdot L_{enc\_seq}\cdot D^2 = 6\cdot 482\cdot 512^2 \approx 758\,\text{MFLOPs}$
- $QK^\top$：$2\cdot H\cdot L_{enc\_seq}^2\cdot d_k = 2\cdot 8\cdot 482^2\cdot 64 \approx 238\,\text{MFLOPs}$
- softmax × V：$\approx 238\,\text{MFLOPs}$
- 输出投影 $W_O$：$2\cdot L_{enc\_seq}\cdot D^2 \approx 253\,\text{MFLOPs}$
- 小计 **≈ 1.49 GFLOPs / 层**

### 4.2 前馈子层（FFN）

```python
x = linear2(dropout(activation(linear1(x))))
x = skip + dropout(x)
x = self.norm2(x)
```

$$
\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2, \quad W_1\in\mathbb{R}^{F\times D},\ W_2\in\mathbb{R}^{D\times F}
$$

$F=3200$，所以每层 FFN 的 FLOPs：

$$
2\cdot L_{enc\_seq}\cdot (D\cdot F + F\cdot D) = 4\cdot 482\cdot 512\cdot 3200 \approx 3.16\,\text{GFLOPs}
$$

> **Encoder 单层 ≈ 4.65 GFLOPs，4 层 ≈ 18.6 GFLOPs**。

### 4.3 Encoder 最终输出

```python
encoder_out = self.encoder(...)             # (482, B, D)
```

数学记号：

$$
H_{enc} = \text{Enc}(X_{enc}^{(0)}; P_{enc}) \in \mathbb{R}^{L_{enc\_seq}\times B\times D}
$$

这是融合了「latent（零）+ 关节状态 + 两路图像每像素」的上下文特征。

---

## 5. 步骤 4：Transformer Decoder（DETR 风格，1 层）

源码：[`ACTDecoderLayer`, modeling_act.py:1180](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1180)

### 5.1 初始化 decoder 输入

```python
decoder_in = torch.zeros((T=100, B, D=512))
decoder_pos_embed = self.decoder_pos_embed.weight.unsqueeze(1)   # (100, 1, D)
```

**核心 trick**：decoder 的输入张量是全零，真正的信息由**可学习的位置编码**（object queries）携带。这 100 个 query 向量 $\{q_t\}_{t=1}^{100}\subset\mathbb{R}^D$ 是模型训练出来的"未来第 t 步应该问什么"的模板。

### 5.2 子层 1：Query 自注意力

```python
skip = x                                    # x 初始为 0
q = k = x + decoder_pos_embed               # q = k = decoder_pos_embed
x = self.self_attn(q, k, value=x)[0]        # 首层 value=0 → 输出=0
x = skip + dropout(x)                       # 仍 = 0
x = self.norm1(x)                           # LayerNorm(0)
```

第一层里 `value=x=0`，所以自注意力子层的输出也是 0（`softmax·V = softmax·0 = 0`），残差后仍然是 0，只是经过一次 LayerNorm 变成常数向量。

> **为什么不是完全退化？** 因为 `n_decoder_layers=1`，自注意力子层对后续**起步值很低**，但 LayerNorm 的 β 仍然能带入偏置。更重要的影响来自下一步的交叉注意力。

公式上：

$$
Y^{(0)} = \text{LayerNorm}\!\left(\underbrace{0}_{\text{value}} + \underbrace{\text{MHA}(q,k,v{=}0)}_{=\,0}\right) \;\;(\text{第一层})
$$

**注意**：如果 `n_decoder_layers > 1`，第二层的 `value` 就不再是 0，这个路径才真正生效。默认 `n_decoder_layers=1` 是与原论文代码的 bug 保持一致（见 [configuration_act.py:166-168](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L166-L168)）。

### 5.3 子层 2：交叉注意力（Encoder ↔ Decoder）

**这是真正产生信息流的一步**。

```python
x = self.multihead_attn(
    query = x + decoder_pos_embed,          # (100, B, D)
    key   = encoder_out + encoder_pos_embed,# (482, B, D)
    value = encoder_out,                    # (482, B, D)
)[0]
x = skip + dropout(x)
x = self.norm2(x)
```

数学上（省略多头下标）：

$$
Q = (Y^{(0)}+P_{dec})\,W_Q,\quad K = (H_{enc}+P_{enc})\,W_K,\quad V = H_{enc}\,W_V
$$

$$
\text{CrossAttn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \in \mathbb{R}^{T\times B\times D}
$$

$$
Y^{(1)} = \text{LayerNorm}\!\left(Y^{(0)} + \text{CrossAttn}\right)
$$

每个 query $q_t$ 对 encoder 的 482 个 token 做加权读取，形成一个"这个时刻需要看哪里"的注意力分布。直观上 $q_t$ 就是"第 t 个未来动作要执行时，机器人应该关注图像的哪些像素、关节状态的哪些维度"。

**FLOPs（B=1，单层）**：

- QKV 投影（Q 长度 T=100，KV 长度 L=482）：$2\cdot T\cdot D^2 + 2\cdot 2\cdot L\cdot D^2 = 2\cdot 100\cdot 512^2 + 4\cdot 482\cdot 512^2 \approx 505 + 505 = 558\,\text{MFLOPs}$
  - 其实是 $2\cdot T\cdot D^2$（Q）$+ 2\cdot L\cdot D^2$（K）$+ 2\cdot L\cdot D^2$（V）$≈ 52+253+253 = 558$ MFLOPs
- $QK^\top$：$2\cdot H\cdot T\cdot L\cdot d_k = 2\cdot 8\cdot 100\cdot 482\cdot 64 \approx 49\,\text{MFLOPs}$
- softmax × V：同上 ≈ 49 MFLOPs
- 输出投影：$2\cdot T\cdot D^2 \approx 52\,\text{MFLOPs}$
- **小计 ≈ 0.71 GFLOPs**

### 5.4 子层 3：FFN

与 encoder 相同，但序列长度只有 $T=100$：

$$
2\cdot T\cdot (D\cdot F + F\cdot D) = 4\cdot 100\cdot 512\cdot 3200 \approx 0.66\,\text{GFLOPs}
$$

最后：

```python
decoder_out = self.decoder.norm(x)          # LayerNorm
decoder_out = decoder_out.transpose(0, 1)   # (B, T, D)
```

$$
H_{dec} \in \mathbb{R}^{B\times T\times D}
$$

> **Decoder 单层合计 ≈ 1.37 GFLOPs**，只有 1 层。

---

## 6. 步骤 5：Action Head

源码：[modeling_act.py:975-977](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L975-L977)

```python
self.action_head = nn.Linear(D, A)
actions = self.action_head(decoder_out)     # (B, T, A)
```

$$
\hat{a}_t = W_a\, H_{dec,t} + b_a,\quad W_a\in\mathbb{R}^{A\times D} = \mathbb{R}^{6\times 512},\ t=1,\dots,100
$$

**FLOPs**：$2\cdot T\cdot D\cdot A \approx 614\,\text{k FLOPs}$，可忽略。

返回值：`actions.shape == (1, 100, 6)`，下游还要经过 `unnormalize_outputs` 把 `MEAN_STD` 归一化反过来才是真正的关节目标。

---

## 7. 后处理：action chunk 队列

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

## 8. 整体计算总量估算（单次 `predict_action_chunk`，B=1）

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

## 9. 一次推理从"数学总览"回看

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

## 10. 推理时"省掉"的部分

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

## 11. 交叉引用

- 源码主体：[`ACT.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L782)
- 配置默认值：[`ACTConfig`](../../../lerobot/src/lerobot/policies/act/configuration_act.py#L37)
- Encoder 层：[`ACTEncoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1080)
- Decoder 层：[`ACTDecoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1238)
- 2D 正弦位置编码：[`ACTSinusoidalPositionEmbedding2d`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1345)
- 调用点：`ACTPolicy.select_action → predict_action_chunk`，[modeling_act.py:224](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L224)
- 对应论文：Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023

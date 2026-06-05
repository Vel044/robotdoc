# Transformer Encoder 推理链路

本文档从 [ACT推理.md](./ACT推理.md) 中拆出，专门分析 ACT 推理中的 `self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)`：

```python
# ACT.forward 中调用
encoder_out = self.encoder(
    encoder_in_tokens,          # (482, 1, 512)，token 内容
    pos_embed=encoder_in_pos_embed,  # (482, 1, 512)，对应位置编码
)
# encoder_out: (482, 1, 512)
```

对应主流程：[ACT推理.md §6](./ACT推理.md#6-步骤-3transformer-encoder4-层)。

---

## 1. 输入输出

| 参数 | 形状 | 内容 |
| --- | --- | --- |
| `encoder_in_tokens` / `x` | `(482, 1, 512)` | 1 个 latent token + 1 个 state token + 480 个 image token |
| `encoder_in_pos_embed` / `pos_embed` | `(482, 1, 512)` | 与 token 一一对应的位置编码 |
| `encoder_out` / `H_enc` | `(482, 1, 512)` | Encoder 4 层输出，形状不变，但每个 token 已融合全局上下文 |

符号含义：

```text
482 = 1 latent + 1 state + 480 image
1   = batch size
512 = Transformer 隐藏维度 dim_model
```

Encoder 的核心作用不是改变 shape，而是让 482 个 token 通过注意力相互交换信息。

---

## 2. 整体结构

ACT 这里的 Encoder 有 4 层，每层结构相同：

```text
输入 x
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
FFN
  ↓
Add & Norm
  ↓
输出 x
```

4 层叠加：

```python
# ACTEncoder.forward
for layer in self.layers:                   # 4 层
    x = layer(x, pos_embed=pos_embed)        # 每层都重新使用同一份位置编码
encoder_out = x                              # (482, 1, 512)
```

形状始终保持：

```text
第 0 层输入:  (482, 1, 512)
第 1 层输出:  (482, 1, 512)
第 2 层输出:  (482, 1, 512)
第 3 层输出:  (482, 1, 512)
第 4 层输出:  (482, 1, 512) = H_enc
```

---

## 3. 单层 EncoderLayer 源码

下面是推理时单层 `ACTEncoderLayer.forward` 的关键代码，按当前 ACT 配置理解即可。ACT 默认 `pre_norm=False`，因此走 **post-norm**：残差相加后再做 LayerNorm。

```python
class ACTEncoderLayer(nn.Module):
    def forward(
        self,
        x,                                      # 当前层输入，形状 (482, 1, 512)
        pos_embed: Tensor | None = None,       # 位置编码，形状 (482, 1, 512)
        key_padding_mask: Tensor | None = None,# padding mask，本实验没有使用
    ) -> Tensor:
        skip = x                               # 保存残差分支，形状 (482, 1, 512)

        if self.pre_norm:
            x = self.norm1(x)                  # 本实验 pre_norm=False，所以不走这里

        # Query 和 Key 加位置编码，Value 不加位置编码
        # Q/K 用来算相关性，所以需要知道 token 的位置
        # V 表示内容本身，所以保留原始内容特征
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(
            q,                                 # Query，形状 (482, 1, 512)
            k,                                 # Key，形状 (482, 1, 512)
            value=x,                           # Value，形状 (482, 1, 512)
            key_padding_mask=key_padding_mask,
        )[0]                                   # attention 输出，形状 (482, 1, 512)
        x = skip + self.dropout1(x)            # 残差连接，形状仍为 (482, 1, 512)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)                  # post-norm：attention 残差后归一化
            skip = x                           # 保存 FFN 子层的残差输入

        # FFN：Linear(512→3200) → ReLU → Dropout → Linear(3200→512)
        x = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )
        x = skip + self.dropout2(x)            # FFN 残差连接

        if not self.pre_norm:
            x = self.norm2(x)                  # post-norm：FFN 残差后归一化

        return x                               # 当前层输出，形状 (482, 1, 512)
```

---

## 4. Self-Attention 子层

### 4.1 Q、K、V 的来源

```python
q = k = x + pos_embed     # (482, 1, 512)
v = x                     # (482, 1, 512)
```

为什么位置编码只加到 Q/K：

```text
Q/K 用来算 token 之间的相关性，因此需要位置信息；
V 是被加权求和的内容，不额外加入位置编码，避免污染内容特征。
```

### 4.2 多头切分

配置：

```text
dim_model = 512
n_heads   = 8
每头维度  = 512 / 8 = 64
```

内部会把 512 维拆成 8 个 head：

```text
Q: (482, 1, 512) → 8 个 (482, 1, 64)
K: (482, 1, 512) → 8 个 (482, 1, 64)
V: (482, 1, 512) → 8 个 (482, 1, 64)
```

### 4.3 注意力计算：每一步公式和形状

下面按**单个 head** 展开。多头注意力一共有 8 个 head，每个 head 的维度是 64。

#### 第 1 步：Q/K/V 投影并切成多头

输入：

```text
q: (482, 1, 512)
k: (482, 1, 512)
v: (482, 1, 512)
```

`MultiheadAttention` 内部先做 3 个线性投影，和 `nn.Linear` 一样是“矩阵乘法 + bias”：

```text
Q = q · W_Q^T + b_Q    → (482, 1, 512)
K = k · W_K^T + b_K    → (482, 1, 512)
V = v · W_V^T + b_V    → (482, 1, 512)
```

再按 8 个 head 切开：

```text
Q_h: (482, 64)
K_h: (482, 64)
V_h: (482, 64)
```

这里下标 `h` 表示“某一个 head”。每个 head 都独立做下面的计算。

#### 第 2 步：相似度矩阵

```text
score = Q_h · K_h^T / √64
```

形状变化：

```text
Q_h:   (482, 64)
K_h^T: (64, 482)

score = (482,64) × (64,482) = (482,482)
```

`score[i,j]` 的含义：

```text
第 i 个 query token 和第 j 个 key token 的内积相似度。
```

所以 `(482,482)` 不是一次内积的结果，而是 482 个 query token 与 482 个 key token 两两内积后组成的相似度矩阵。

#### 第 3 步：softmax 权重

```text
weight = softmax(score)
```

形状不变：

```text
score:  (482,482)
weight: (482,482)
```

softmax 沿最后一维做，也就是对每一行的 482 个分数归一化：

```text
weight[i,j] = exp(score[i,j]) / Σ_k exp(score[i,k])
```

归一化后：

```text
Σ_j weight[i,j] = 1
```

也就是说，对于第 `i` 个 token，`weight[i,:]` 是它看全部 482 个 token 的注意力分布。

#### 第 4 步：对 V 做加权求和

```text
head_out = weight · V_h
```

形状变化：

```text
weight: (482,482)
V_h:    (482,64)

head_out = (482,482) × (482,64) = (482,64)
```

对第 `i` 个 token 来说：

```text
head_out[i] = Σ_j weight[i,j] · V_h[j]
```

这一步不是再算相似度，而是用注意力权重对所有 value 向量做加权平均，得到第 `i` 个 token 的上下文向量。

#### 第 5 步：8 个 head 拼回 512 维

8 个 head 每个输出 `(482,64)`：

```text
head_out_1: (482,64)
head_out_2: (482,64)
...
head_out_8: (482,64)
```

拼接：

```text
concat(head_out_1 ... head_out_8) = (482,512)
```

再经过输出投影：

```text
attn_out = concat · W_o + b_o
```

最终恢复 batch 维：

```text
attn_out: (482, 1, 512)
```

直观解释：

```text
Self-Attention 让每个 token 都去看全部 482 个 token。
比如 state token 可以看图像 token，图像 token 也可以看 state token。
```

### 4.4 Attention 后的 Add & Norm

代码：

```python
x = skip + self.dropout1(attn_out)
x = self.norm1(x)
```

形状：

```text
skip:     (482, 1, 512)
attn_out: (482, 1, 512)

skip + attn_out → (482, 1, 512)
LayerNorm       → (482, 1, 512)
```

推理时模型处于 `eval()`，dropout 等价于恒等映射；但结构上仍然保留 dropout 和残差路径。

LayerNorm 对每个 token 的 512 维独立归一化：

```text
mean = 512 个维度的均值
std  = 512 个维度的标准差
x_norm = (x - mean) / (std + ε)
output = γ × x_norm + β
```

其中 `γ`、`β` 是可学习参数，形状都是 `(512,)`。

---

## 5. FFN 子层

FFN 是每个 token 独立执行的两层 MLP。它不让 token 之间互相交换信息，token 之间的信息交换已经在 Self-Attention 里完成；FFN 只负责把每个 token 自己的 512 维特征做非线性变换。

```python
x = self.linear2(
    self.dropout(
        self.activation(
            self.linear1(x)
        )
    )
)
```

### 5.1 输入

```text
x: (482, 1, 512)
```

含义：

```text
482 = token 数
1   = batch size
512 = 每个 token 的特征维度
```

FFN 会对 482 个 token 分别执行同一组线性层。不同 token 之间不会在 FFN 里相乘或加权。

### 5.2 第 1 步：Linear(512 → 3200)

```text
linear1_out = x · W1^T + b1
```

参数形状：

```text
W1: (3200, 512)
b1: (3200,)
```

形状变化：

```text
x:           (482, 1, 512)
W1^T:        (512, 3200)

linear1_out: (482, 1, 3200)
```

对单个 token 来说，就是 3200 次内积：

```text
输入 token: (512,)
第 k 个输出 = 这个 512 维 token 与 W1 第 k 行做内积 + b1[k]
最终得到: (3200,)
```

所以 `Linear` 前向不是外积，而是矩阵乘法；矩阵乘法里的每个输出元素都是一次内积加一个偏置。

### 5.3 第 2 步：ReLU 激活

```text
relu_out = ReLU(linear1_out) = max(0, linear1_out)
```

形状不变：

```text
linear1_out: (482, 1, 3200)
relu_out:    (482, 1, 3200)
```

ReLU 是纯计算，没有可学习参数：

```text
正数保留
负数变成 0
```

它的作用是引入非线性。没有 ReLU，两层 Linear 可以合并成一层 Linear；有了 ReLU，中间会出现“截断”，模型才能表达更复杂的函数。

### 5.4 第 3 步：Linear(3200 → 512)

```text
ffn_out = relu_out · W2^T + b2
```

参数形状：

```text
W2: (512, 3200)
b2: (512,)
```

形状变化：

```text
relu_out: (482, 1, 3200)
W2^T:     (3200, 512)

ffn_out:  (482, 1, 512)
```

对单个 token 来说，就是把 3200 维中间特征再压回 512 维：

```text
输入 token: (3200,)
第 k 个输出 = 这个 3200 维向量与 W2 第 k 行做内积 + b2[k]
最终得到: (512,)
```

### 5.5 第 4 步：FFN 后的 Add & Norm

代码：

```python
x = skip + self.dropout2(ffn_out)
x = self.norm2(x)
```

推理时 dropout 等价于恒等映射，所以形状可以直接看成：

```text
skip:    (482, 1, 512)
ffn_out: (482, 1, 512)

skip + ffn_out → (482, 1, 512)
LayerNorm      → (482, 1, 512)
```

LayerNorm 仍然是对每个 token 的 512 维独立做：

```text
mean = 512 个维度的均值
std  = 512 个维度的标准差
x_norm = (x - mean) / (std + ε)
output = γ × x_norm + β
```

其中 `γ`、`β` 是可学习参数，形状都是 `(512,)`。

### 5.6 FFN 总形状流

```text
(482, 1, 512)
  ↓ Linear(512→3200)
(482, 1, 3200)
  ↓ ReLU
(482, 1, 3200)
  ↓ Linear(3200→512)
(482, 1, 512)
  ↓ 残差 + LayerNorm
(482, 1, 512)
```

为什么要先扩维再压回：

```text
如果只有 Linear，没有 ReLU，多层线性变换仍然等价于一层线性变换。
FFN 中间加入 ReLU 后，模型才能表达非线性关系。
3200 > 512，给每个 token 更大的中间表达空间。
```

---

## 6. 计算量估算

单层 Encoder，`B=1`，`L=482`，`D=512`，`H=8`：

| 子模块 | FLOPs 近似值 | 说明 |
| --- | --- | --- |
| Q/K/V 投影 | `3 × 2 × 482 × 512² ≈ 758M` | 三个线性投影 |
| Q·K^T | `2 × 8 × 482² × 64 ≈ 238M` | 注意力分数 |
| softmax·V | `2 × 8 × 482² × 64 ≈ 238M` | 加权求和 |
| 输出投影 | `2 × 482 × 512² ≈ 253M` | 多头拼回后的线性层 |
| Self-Attention 小计 | `≈ 1.49G` | 单层 |
| FFN | `≈ 3.16G` | `512→3200→512` |
| 单层合计 | `≈ 4.65G` | attention + FFN |
| 4 层合计 | `≈ 18.6G` | Encoder 总计算量 |

FFN 是 Encoder 中计算量最大的部分，比 Self-Attention 更重。

---

## 7. 输出 H_enc 的含义

最终：

```text
H_enc = Encoder(X_enc, P_enc)
H_enc.shape = (482, 1, 512)
```

`H_enc` 仍然有 482 个 token，但每个 token 都融合了其他 token 的上下文：

```text
H_enc[0]      latent token 位，融合了视觉和关节状态
H_enc[1]      state token 位，融合了视觉和 latent
H_enc[2:482]  image token 位，融合了其他图像位置和机器人状态
```

Decoder 后续会把 `H_enc` 当作 `memory`：

```python
decoder_out = self.decoder(
    decoder_in,                         # (100, 1, 512)
    encoder_out,                        # H_enc，(482, 1, 512)
    encoder_pos_embed=encoder_in_pos_embed,
    decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
)
```

一句话总结：

```text
Encoder 把当前观测编码成全局上下文 H_enc；
Decoder 再根据 H_enc 生成未来 100 步动作。
```

---

## 8. 交叉引用

- 主推理文档：[ACT推理.md](./ACT推理.md)
- 对应结构图：[TransformerEncoder结构.svg](./TransformerEncoder结构.svg)
- Encoder 层源码：[`ACTEncoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1080)
- PyTorch 多头注意力：[`torch.nn.MultiheadAttention`](../../pytorch/torch/nn/modules/activation.py#L1091)
- PyTorch attention functional：[`multi_head_attention_forward`](../../pytorch/torch/nn/functional.py#L6228)

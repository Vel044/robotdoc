# Transformer Decoder 推理链路

本文档从 [ACT推理.md](./ACT推理.md) 中拆出，专门分析 ACT 推理中的 `self.decoder(...)`：

```python
# ACT.forward 中调用
decoder_in = torch.zeros((100, 1, 512))          # 100 个动作 query 的内容占位，全零

decoder_out = self.decoder(
    decoder_in,                                  # (100, 1, 512)，全零 query 内容
    encoder_out,                                 # H_enc，(482, 1, 512)，Encoder 输出 memory
    encoder_pos_embed=encoder_in_pos_embed,      # P_enc，(482, 1, 512)，Encoder 位置编码
    decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
                                                 # P_dec，(100, 1, 512)，可学习动作 query 编码
)
# decoder_out: (100, 1, 512)

decoder_out = decoder_out.transpose(0, 1)
# decoder_out: (1, 100, 512)
```

对应主流程：[ACT推理.md §7](./ACT推理.md#7-步骤-4transformer-decoderdetr-风格1-层)。

---

## 1. 输入输出

| 参数 | 形状 | 内容 |
| --- | --- | --- |
| `decoder_in` / `x` | `(100, 1, 512)` | 100 个未来动作 query 的内容输入，推理时全零 |
| `decoder_pos_embed` / `P_dec` | `(100, 1, 512)` | 来自 `self.decoder_pos_embed.weight` 的可学习模型参数，100 个 query 向量对应未来 100 个动作步 |
| `encoder_out` / `H_enc` | `(482, 1, 512)` | Encoder 输出的观测上下文 memory |
| `encoder_pos_embed` / `P_enc` | `(482, 1, 512)` | Encoder 侧 482 个 token 的位置编码 |
| `decoder_out` / `H_dec` | `(100, 1, 512)` | Decoder 输出的动作特征 |
| `decoder_out.transpose(0, 1)` | `(1, 100, 512)` | 送入 Action Head 前的动作特征 |

符号含义：

```text
100 = chunk_size，一次预测未来 100 帧动作
482 = Encoder memory 长度 = 1 latent + 1 state + 480 image
1   = batch size
512 = Transformer 隐藏维度 dim_model
```

Decoder 的核心作用是：用 100 个未来动作 query 去查询 Encoder 的 482 个观测 token，得到 100 个动作特征。

`P_dec` 不是从 observation、state 或图像算出来的，而是模型训练时学到的参数；加载 checkpoint 后推理时固定使用。

---

## 2. 整体结构

ACT 当前默认 `n_decoder_layers=1`，所以 Decoder 只有 1 层：

```text
decoder_in: zeros(100,1,512)
P_dec:      learnable(100,1,512)
H_enc:      (482,1,512)
P_enc:      (482,1,512)
  ↓
Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention
  ↓
Add & Norm
  ↓
FFN
  ↓
Add & Norm
  ↓
Decoder final LayerNorm
  ↓
H_dec: (100,1,512)
  ↓ transpose
(1,100,512)
```

和 Encoder 最大的区别：

```text
Encoder Self-Attention：482 个观测 token 彼此看
Decoder Cross-Attention：100 个动作 query 去看 482 个观测 memory
```

---

## 3. 单层 DecoderLayer 源码

下面是推理时单层 `ACTDecoderLayer.forward` 的关键代码。ACT 默认 `pre_norm=False`，因此走 **post-norm**：残差相加后再做 LayerNorm。

```python
class ACTDecoderLayer(nn.Module):
    def forward(
        self,
        x: Tensor,                          # 当前 decoder token，形状 (100, 1, 512)
        encoder_out: Tensor,                # Encoder memory，形状 (482, 1, 512)
        decoder_pos_embed: Tensor | None = None, # P_dec，形状 (100, 1, 512)
        encoder_pos_embed: Tensor | None = None, # P_enc，形状 (482, 1, 512)
    ) -> Tensor:
        # 子层 1：Self-Attention，100 个动作 query 之间相互注意
        skip = x                            # 保存残差分支，形状 (100, 1, 512)
        q = k = x + decoder_pos_embed       # Q/K 加动作 query 位置编码
        x = self.self_attn(
            q,                              # Query，形状 (100, 1, 512)
            k,                              # Key，形状 (100, 1, 512)
            value=x,                        # Value，形状 (100, 1, 512)
        )[0]                                # 输出形状 (100, 1, 512)
        x = skip + self.dropout1(x)         # 残差连接，形状仍为 (100, 1, 512)
        x = self.norm1(x)                   # post-norm
        skip = x                            # 保存给 Cross-Attention 残差分支

        # 子层 2：Cross-Attention，动作 query 查询 Encoder memory
        x = self.multihead_attn(
            query=x + decoder_pos_embed,    # Q，形状 (100, 1, 512)
            key=encoder_out + encoder_pos_embed, # K，形状 (482, 1, 512)
            value=encoder_out,              # V，形状 (482, 1, 512)
        )[0]                                # 输出形状 (100, 1, 512)
        x = skip + self.dropout2(x)         # 残差连接
        x = self.norm2(x)                   # post-norm
        skip = x                            # 保存给 FFN 残差分支

        # 子层 3：FFN，每个动作 query 独立做 512→3200→512
        x = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )
        x = skip + self.dropout3(x)         # 残差连接
        x = self.norm3(x)                   # post-norm

        return x                            # 当前层输出，形状 (100, 1, 512)
```

---

## 4. Decoder Self-Attention 子层

这一节的计算过程**参考前面 Encoder 的 Self-Attention**，详见 [TransformerEncoder推理.md §4](./TransformerEncoder推理.md#4-self-attention-子层)。

Decoder 的 Self-Attention 和 Encoder 的 Self-Attention 是同一种 `MultiheadAttention` 计算：都是 `Q·K^T → softmax → weight·V`。区别是这里的序列长度是 100，输入来源是动作 query；Encoder 那边的序列长度是 482，输入来源是观测 token。

### 4.1 Q、K、V 的来源

```python
q = k = x + decoder_pos_embed   # (100, 1, 512)
v = x                           # (100, 1, 512)
```

推理进入第 1 层时：

```text
x = decoder_in = 全零 (100, 1, 512)
decoder_pos_embed = P_dec = 可学习参数 (100, 1, 512)
```

因此：

```text
Q/K 的输入主要来自 P_dec
V 的输入是全零 x
```

### 4.2 注意力计算形状

按单个 head 展开，每头维度为 64：

```text
Q_h: (100,64)
K_h: (100,64)
V_h: (100,64)
```

相似度矩阵：

```text
score = Q_h · K_h^T / √64

Q_h:   (100,64)
K_h^T: (64,100)
score: (100,100)
```

softmax：

```text
weight = softmax(score)
weight: (100,100)
```

加权求和：

```text
head_out = weight · V_h

weight:  (100,100)
V_h:     (100,64)
head_out:(100,64)
```

8 个 head 拼接后：

```text
concat 8 heads → (100,512)
输出投影 W_o  → (100,1,512)
```

### 4.3 为什么推理首层近似空跑

首层进入 self-attention 时 `V = x = 0`：

```text
head_out = weight · V_h = weight · 0 = 0
```

所以 self-attention 的输出主要仍是 0，残差后也是 0，经过 `LayerNorm(0)` 后得到的是 LayerNorm 的可学习偏置 `β`。由于 ACT 默认只有 1 层 Decoder，这个 self-attention 子层对观测信息没有贡献，真正读取观测的是下一步 Cross-Attention。

---

## 5. Cross-Attention 子层

Cross-Attention 是 Decoder 真正做事的一步：100 个动作 query 去看 Encoder 的 482 个 memory token。

### 5.1 Q、K、V 的来源

```python
Q_in = x + decoder_pos_embed          # (100, 1, 512)
K_in = encoder_out + encoder_pos_embed# (482, 1, 512)
V_in = encoder_out                    # (482, 1, 512)
```

含义：

```text
Q 来自 Decoder：第 t 个动作 query 想问“未来第 t 步该关注什么”
K 来自 Encoder：482 个观测 token 带上位置编码，用来被匹配
V 来自 Encoder：482 个观测 token 的内容本身，用来被加权求和
```

### 5.2 第 1 步：Q/K/V 投影并切成多头

`MultiheadAttention` 内部先做 3 个线性投影：

```text
Q = Q_in · W_Q^T + b_Q    → (100, 1, 512)
K = K_in · W_K^T + b_K    → (482, 1, 512)
V = V_in · W_V^T + b_V    → (482, 1, 512)
```

切成 8 个 head 后，单个 head 的形状：

```text
Q_h: (100,64)
K_h: (482,64)
V_h: (482,64)
```

### 5.3 第 2 步：相似度矩阵

```text
score = Q_h · K_h^T / √64
```

形状变化：

```text
Q_h:   (100,64)
K_h^T: (64,482)

score = (100,64) × (64,482) = (100,482)
```

`score[t,j]` 的含义：

```text
第 t 个动作 query 和第 j 个 Encoder memory token 的内积相似度。
```

所以 `(100,482)` 表示：100 个未来动作步分别对 482 个观测 token 打分。

### 5.4 第 3 步：softmax 权重

```text
weight = softmax(score)
```

形状不变：

```text
score:  (100,482)
weight: (100,482)
```

softmax 沿最后一维做，也就是对每个动作 query 的 482 个观测 token 分数归一化：

```text
weight[t,j] = exp(score[t,j]) / Σ_k exp(score[t,k])
```

归一化后：

```text
Σ_j weight[t,j] = 1
```

### 5.5 第 4 步：对 Encoder V 做加权求和

```text
head_out = weight · V_h
```

形状变化：

```text
weight: (100,482)
V_h:    (482,64)

head_out = (100,482) × (482,64) = (100,64)
```

对第 `t` 个动作 query 来说：

```text
head_out[t] = Σ_j weight[t,j] · V_h[j]
```

这一步不是再算相似度，而是用注意力权重对 Encoder 的 482 个 value 向量做加权平均，得到第 `t` 个未来动作步的上下文向量。

### 5.6 第 5 步：8 个 head 拼回 512 维

8 个 head 每个输出 `(100,64)`：

```text
head_out_1: (100,64)
head_out_2: (100,64)
...
head_out_8: (100,64)
```

拼接：

```text
concat(head_out_1 ... head_out_8) = (100,512)
```

再经过输出投影：

```text
cross_out = concat · W_o^T + b_o
```

最终恢复 batch 维：

```text
cross_out: (100, 1, 512)
```

---

## 6. Cross-Attention 后的 Add & Norm

代码：

```python
x = skip + self.dropout2(cross_out)
x = self.norm2(x)
```

推理时 dropout 等价于恒等映射：

```text
skip:      (100, 1, 512)
cross_out: (100, 1, 512)

skip + cross_out → (100, 1, 512)
LayerNorm        → (100, 1, 512)
```

LayerNorm 对每个动作 query 的 512 维独立归一化：

```text
mean = 512 个维度的均值
std  = 512 个维度的标准差
x_norm = (x - mean) / (std + ε)
output = γ × x_norm + β
```

其中 `γ`、`β` 是可学习参数，形状都是 `(512,)`。

---

## 7. FFN 子层

FFN 和 Encoder 里的 FFN 结构相同，只是 token 数从 482 变成 100。

### 7.1 输入

```text
x: (100, 1, 512)
```

### 7.2 Linear(512 → 3200)

```text
linear1_out = x · W1^T + b1
```

形状变化：

```text
x:           (100, 1, 512)
W1^T:        (512, 3200)
linear1_out: (100, 1, 3200)
```

### 7.3 ReLU

```text
relu_out = max(0, linear1_out)
```

形状不变：

```text
relu_out: (100, 1, 3200)
```

### 7.4 Linear(3200 → 512)

```text
ffn_out = relu_out · W2^T + b2
```

形状变化：

```text
relu_out: (100, 1, 3200)
W2^T:     (3200, 512)
ffn_out:  (100, 1, 512)
```

### 7.5 FFN 后的 Add & Norm

```text
skip:    (100, 1, 512)
ffn_out: (100, 1, 512)

skip + ffn_out → (100, 1, 512)
LayerNorm      → (100, 1, 512)
```

---

## 8. Decoder 最终输出

`ACTDecoder.forward` 在所有层结束后还有一个最终 LayerNorm：

```python
x = self.norm(x)
```

形状不变：

```text
x: (100, 1, 512)
```

回到 `ACT.forward` 后转置：

```python
decoder_out = decoder_out.transpose(0, 1)
```

形状变化：

```text
(100, 1, 512) → (1, 100, 512)
```

最终：

```text
H_dec: (1, 100, 512)
```

`H_dec[0,t]` 是未来第 `t` 帧动作的 512 维内部表示，后续送入 Action Head：

```text
Action Head: Linear(512→6)
(1,100,512) → (1,100,6)
```

---

## 9. 计算量估算

单层 Decoder，`B=1`，`T=100`，`L_enc=482`，`D=512`，`H=8`：

| 子模块 | FLOPs 近似值 | 说明 |
| --- | --- | --- |
| Query Self-Attention | 很小且首层近似无效 | `V=x=0`，信息不来自观测 |
| Cross-Attention Q 投影 | `2 × 100 × 512² ≈ 52M` | query 长度 100 |
| Cross-Attention K/V 投影 | `2 × 2 × 482 × 512² ≈ 506M` | memory 长度 482 |
| Cross-Attention Q·K^T | `2 × 8 × 100 × 482 × 64 ≈ 49M` | 注意力分数 |
| Cross-Attention softmax·V | `≈ 49M` | 加权求和 |
| Cross-Attention 输出投影 | `2 × 100 × 512² ≈ 52M` | 多头拼回 |
| Cross-Attention 小计 | `≈ 0.71G` | 单层 |
| FFN | `≈ 0.66G` | `100 × 512→3200→512` |
| Decoder 单层合计 | `≈ 1.37G` | Cross-Attention + FFN |

---

## 10. 输出 H_dec 的含义

最终：

```text
H_dec = Decoder(zeros, H_enc, P_dec, P_enc)
H_dec.shape = (1, 100, 512)
```

含义：

```text
H_dec[0,0]   第 0 帧未来动作的内部特征
H_dec[0,1]   第 1 帧未来动作的内部特征
...
H_dec[0,99]  第 99 帧未来动作的内部特征
```

一句话总结：

```text
Decoder 用 100 个可学习动作 query，从 Encoder 的 482 个观测 memory 中读取信息，
生成 100 个动作特征，再交给 Action Head 变成 (1,100,6) 的动作 chunk。
```

---

## 11. 交叉引用

- 主推理文档：[ACT推理.md](./ACT推理.md)
- 对应结构图：[TransformerDecoder结构.svg](./TransformerDecoder结构.svg)
- Decoder 层源码：[`ACTDecoderLayer.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1295)
- Decoder 容器源码：[`ACTDecoder.forward`](../../../lerobot/src/lerobot/policies/act/modeling_act.py#L1208)
- Encoder 详解：[TransformerEncoder推理.md](./TransformerEncoder推理.md)
- PyTorch 多头注意力：[`torch.nn.MultiheadAttention`](../../pytorch/torch/nn/modules/activation.py#L1091)
- PyTorch attention functional：[`multi_head_attention_forward`](../../pytorch/torch/nn/functional.py#L6228)

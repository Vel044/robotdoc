# ResNet18 视觉特征提取链路

本文档从 ACT 推理主文档中拆出，专门分析 `self.backbone(img)["feature_map"]` 这一段：

```python
# forward 里调用（modeling_act.py:926）：
cam_features = self.backbone(img)["feature_map"]
# img:          (1, 3, 360, 640)  归一化后的 RGB 图像
# cam_features: (1, 512, 12, 20)  每个空间位置一个 512 维特征向量
```

对应主流程：[ACT推理.md §4.4.1](./ACT推理.md)。

---

## 1. 在 ACT 里的定义与调用

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
# IntermediateLayerGetter：包装一下，让 forward 只跑到 layer4 就停，不走 avgpool 和 fc 分类头
self.backbone = IntermediateLayerGetter(
    backbone_model, return_layers={"layer4": "feature_map"}
)
# 返回的是 dict：{"feature_map": Tensor(B,512,h,w)}，所以下面要用 ["feature_map"] 取出来

# forward 里调用（modeling_act.py:926）：
cam_features = self.backbone(img)["feature_map"]
# img:          (1, 3, 360, 640)  归一化后的 RGB 图像
# cam_features: (1, 512, 12, 20)  每个空间位置一个 512 维特征向量
```

这里的 `self.backbone` 是 torchvision 的 `IntermediateLayerGetter`。它接收完整的 ResNet18，但只返回指定中间层输出。`return_layers={"layer4": "feature_map"}` 的意思是：执行到 `layer4` 后，把 `layer4` 的输出放进字典，键名叫 `feature_map`。

原版 ResNet18 是 ImageNet 分类网络，`layer4` 后面还有：

- `avgpool`：全局平均池化，把 `(512, 12, 20)` 压成 `(512, 1, 1)`
- `fc`：线性分类头，把 512 维映射到 1000 个 ImageNet 类别

ACT 不需要分类结果，它要的是带空间位置的视觉特征图，所以在 `layer4` 截断。

---

## 2. 总调用链

```text
self.backbone(img)                                          # modeling_act.py:926
  │
  │  self.backbone 是 IntermediateLayerGetter 实例           # torchvision/models/_utils.py
  │  调用 IntermediateLayerGetter.forward(img)：
  │    for name, module in self.items():
  │        x = module(x)          ← 逐个执行下面的子模块
  │
  ├── x = Conv2d.forward(x)                                 # nn.Conv2d(3→64, 7×7, stride=2)
  ├── x = FrozenBatchNorm2d.forward(x)                       # x * scale + bias
  ├── x = ReLU.forward(x)                                    # max(0, x)
  ├── x = MaxPool2d.forward(x)                               # 3×3 窗口取 max，stride=2
  ├── x = layer1.forward(x)                                  # nn.Sequential.forward()
  │     ├── x = BasicBlock.forward(x)                        # resnet.py:89
  │     │     ├── x' = Conv2d.forward(x)                     # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')          # 固定仿射归一化
  │     │     ├── x' = ReLU.forward(x')                      # 逐元素 max(0,x)
  │     │     ├── x' = Conv2d.forward(x')                    # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')          # 固定仿射归一化
  │     │     ├── shortcut = x                               # 形状相同，残差支路直传
  │     │     └── x = ReLU.forward(x' + shortcut)             # 主支路 + 残差支路
  │     └── x = BasicBlock.forward(x)                        # 同上
  ├── x = layer2.forward(x)                                  # nn.Sequential.forward()
  │     ├── x = BasicBlock.forward(x)                        # 有 downsample
  │     │     ├── x' = Conv2d.forward(x)                     # 3×3 conv, stride=2
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')          # 固定仿射归一化
  │     │     ├── x' = ReLU.forward(x')                      # 逐元素 max(0,x)
  │     │     ├── x' = Conv2d.forward(x')                    # 3×3 conv
  │     │     ├── x' = FrozenBatchNorm2d.forward(x')          # 固定仿射归一化
  │     │     ├── shortcut = downsample(x)                    # conv1x1(stride=2) + BN
  │     │     └── x = ReLU.forward(x' + shortcut)             # 主支路 + 对齐后的残差支路
  │     └── x = BasicBlock.forward(x)                        # 无 downsample
  ├── x = layer3.forward(x)                                  # 同 layer2 结构
  └── x = layer4.forward(x)                                  # 同 layer2 结构
        ↑ 这里捕获输出，返回 {"feature_map": Tensor(1, 512, 12, 20)}
```

---

## 3. 形状总览

输入是单路摄像头图像：

```text
img: (1, 3, 360, 640)
```

ResNet18 到 `layer4` 的形状变化：

| 模块 | 输出形状 | 说明 |
| --- | --- | --- |
| 输入图像 | `(1, 3, 360, 640)` | batch=1，RGB 三通道 |
| `conv1` | `(1, 64, 180, 320)` | 7×7 卷积，stride=2，空间减半 |
| `bn1` | `(1, 64, 180, 320)` | 冻结 BN，形状不变 |
| `relu` | `(1, 64, 180, 320)` | 负数置零，形状不变 |
| `maxpool` | `(1, 64, 90, 160)` | 3×3 池化，stride=2，空间再次减半 |
| `layer1` | `(1, 64, 90, 160)` | 2 个 BasicBlock，通道和空间不变 |
| `layer2` | `(1, 128, 45, 80)` | 第 1 个 block stride=2，通道 64→128 |
| `layer3` | `(1, 256, 23, 40)` | 第 1 个 block stride=2，通道 128→256 |
| `layer4` | `(1, 512, 12, 20)` | 第 1 个 block stride=2，通道 256→512 |

总下采样因子是 32：

```text
conv1 stride=2
maxpool stride=2
layer2 stride=2
layer3 stride=2
layer4 stride=2
总计：2 × 2 × 2 × 2 × 2 = 32
```

所以 `360/32≈12`，`640/32=20`，最终每路摄像头提供 `12×20=240` 个空间位置，每个位置是 512 维视觉特征。

---

## 4. 每一步详解

### 4.1 `conv1`：入口卷积

`conv1` 是 ResNet 给第一层卷积起的变量名，实际类型是 `nn.Conv2d`。

```text
权重: W ∈ ℝ^{64 × 3 × 7 × 7}     ← 64 个 7×7×3 滤波器
偏置: 无                         ← bias=False，后面的 BN 可以吸收偏置
输入: x ∈ ℝ^{1 × 3 × 360 × 640}  ← 归一化后的 RGB 图像
参数: stride=2, padding=3
输出: y ∈ ℝ^{1 × 64 × 180 × 320}
```

对输出的每个 batch、输出通道、空间位置做下面的乘加：

$$
y[k,m,n] =
\sum_{c=0}^{2}
\sum_{i=0}^{6}
\sum_{j=0}^{6}
W[k,c,i,j]\cdot x[c,2m+i-3,2n+j-3]
$$

含义逐项拆开：

- `k`：第几个输出通道，也就是第几个卷积核，范围 `0..63`
- `c`：输入 RGB 通道，范围 `0..2`
- `i,j`：7×7 卷积核内部坐标
- `2m,2n`：因为 `stride=2`，输出移动一步，输入窗口移动两步
- `-3`：因为 `padding=3`，坐标要把补零边界算进去

空间尺寸计算：

$$
H_{out}=\left\lfloor\frac{H_{in}+2p-kernel}{stride}\right\rfloor+1
=\left\lfloor\frac{360+6-7}{2}\right\rfloor+1=180
$$

`W_out` 同理：

$$
W_{out}=\left\lfloor\frac{640+6-7}{2}\right\rfloor+1=320
$$

每个输出像素需要 `3×7×7=147` 次乘加。`conv1` 的作用是把原始 RGB 像素转换成 64 个低级视觉响应，例如边缘、颜色块和纹理方向。

### 4.2 `bn1`：冻结批归一化

ACT 创建 ResNet18 时指定：

```python
backbone_model = torchvision.models.resnet18(
    weights="IMAGENET1K_V1",       # 加载 ImageNet 预训练权重
    norm_layer=FrozenBatchNorm2d,  # 把 ResNet 里的 BN 全部替换成冻结版
)
```

`FrozenBatchNorm2d` 在推理时不更新 `running_mean` 和 `running_var`。它等价于一个逐通道固定仿射变换：

$$
y_c = \gamma_c \cdot \frac{x_c-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}}+\beta_c
$$

源码实现通常会把上式合并成一次乘加：

$$
\text{scale}_c = \frac{\gamma_c}{\sqrt{\sigma_c^2+\varepsilon}}
$$

$$
\text{bias}_c = \beta_c-\mu_c\cdot\text{scale}_c
$$

$$
y_c=x_c\cdot\text{scale}_c+\text{bias}_c
$$

形状保持 `(1, 64, 180, 320)`。这里的 `c` 是通道号，每个通道有自己的一组 `scale` 和 `bias`，同一通道的所有空间位置共享这组数。

### 4.3 `relu`：非线性激活

```text
y = max(0, x)
```

逐元素规则：

- 输入大于 0：原样保留
- 输入小于等于 0：置为 0

形状保持 `(1, 64, 180, 320)`。如果没有 ReLU，多层卷积和线性层叠起来仍然等价于一个更大的线性变换；ReLU 让网络能够表达非线性视觉模式。

### 4.4 `maxpool`：最大池化下采样

`maxpool` 是：

```text
nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
```

对每个通道独立做 3×3 窗口最大值：

$$
y[m,n]=\max_{i,j\in\{0,1,2\}} x[2m+i-1,2n+j-1]
$$

输出形状：

```text
(1, 64, 180, 320) → (1, 64, 90, 160)
```

它保留局部区域里响应最强的特征，同时把空间尺寸减半。

### 4.5 `layer1` 到 `layer4`：残差 stage

ResNet18 的四个 stage 都是 `nn.Sequential(BasicBlock × 2)`。区别在于通道数和第一个 block 是否下采样：

| stage | 输入形状 | 输出形状 | 第一个 block stride | 通道变化 |
| --- | --- | --- | --- | --- |
| `layer1` | `(1, 64, 90, 160)` | `(1, 64, 90, 160)` | 1 | 64→64 |
| `layer2` | `(1, 64, 90, 160)` | `(1, 128, 45, 80)` | 2 | 64→128 |
| `layer3` | `(1, 128, 45, 80)` | `(1, 256, 23, 40)` | 2 | 128→256 |
| `layer4` | `(1, 256, 23, 40)` | `(1, 512, 12, 20)` | 2 | 256→512 |

每个 `BasicBlock` 的主支路：

$$
\text{out}
=\text{Conv}_{3\times3}(x)
\to \text{FrozenBN}
\to \text{ReLU}
\to \text{Conv}_{3\times3}
\to \text{FrozenBN}
$$

每个 `BasicBlock` 的残差输出：

$$
\text{output}=\text{ReLU}(\text{out}+\text{shortcut}(x))
$$

`shortcut(x)` 有两种情况：

- 形状不变时：`shortcut(x)=x`，直接把输入加回主支路
- 形状变化时：`shortcut(x)=FrozenBN(Conv1x1(x))`，用 1×1 卷积对齐通道数和空间尺寸

残差连接的意义是让 block 学习“在输入基础上补充什么”，而不是每一层都从零学习完整映射。这让网络更深时也容易训练。

---

## 5. BasicBlock 逐行逻辑

ResNet18 的 `BasicBlock.forward` 可以压缩成下面的推理路径：

```python
identity = x
# identity 保存残差支路输入；如果后面不需要 downsample，它会原样加回主支路。

out = self.conv1(x)
# 第一层 3×3 卷积；可能 stride=1 保持空间尺寸，也可能 stride=2 做下采样。

out = self.bn1(out)
# 冻结 BN；按通道执行 out * scale + bias。

out = self.relu(out)
# 逐元素 ReLU；负数清零，引入非线性。

out = self.conv2(out)
# 第二层 3×3 卷积；stride 通常为 1，继续提取局部特征。

out = self.bn2(out)
# 冻结 BN；再次把通道分布拉回稳定范围。

if self.downsample is not None:
    identity = self.downsample(x)
    # 当空间尺寸或通道数变化时，残差支路也必须变成同样形状。
    # downsample 通常是 Conv2d(1×1, stride=2) + FrozenBatchNorm2d。

out += identity
# 主支路输出和残差支路逐元素相加；两者形状必须完全一致。

out = self.relu(out)
# 相加后再做一次 ReLU，得到 block 最终输出。
```

以 `layer2` 第一个 block 为例：

```text
输入 x:        (1, 64, 90, 160)
主支路 conv1:  (1, 128, 45, 80)    ← 3×3, stride=2, 通道 64→128
主支路 conv2:  (1, 128, 45, 80)    ← 3×3, stride=1
残差支路:      (1, 128, 45, 80)    ← 1×1, stride=2, 通道 64→128
相加后输出:    (1, 128, 45, 80)
```

如果没有残差支路的 1×1 卷积，`(1,64,90,160)` 不能和 `(1,128,45,80)` 相加，形状对不上。

---

## 6. 输出给 ACT 的含义

`layer4` 输出：

```text
feature_map: (1, 512, 12, 20)
```

含义：

- `1`：batch size，推理时单帧
- `512`：每个空间位置的视觉语义特征维度
- `12×20`：输入图像下采样 32 倍后的空间网格

在 ACT 主流程中，这个输出还会继续走三步：

```python
cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
# 生成同形状的 2D 正弦位置编码：(1, 512, 12, 20)

cam_features = self.encoder_img_feat_input_proj(cam_features)
# 1×1 Conv 投影：(1, 512, 12, 20) → (1, 512, 12, 20)

cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
# 展平成 token 序列：(1, 512, 12, 20) → (240, 1, 512)
```

所以单路摄像头最终贡献 240 个 image token；本次配置有 `handeye` 和 `fixed` 两路摄像头，共贡献 480 个 image token。

---

## 7. 计算量

单路 `360×640` 图像跑 ResNet18 到 `layer4`，粗估约 12 GFLOPs。两路摄像头：

```text
12 GFLOPs/路 × 2 路 ≈ 24 GFLOPs
```

这部分是 `predict_action_chunk` 的主要开销之一，占整次 ACT 推理计算量的 50% 左右。对树莓派 5 这种 ARM CPU-only 环境，ResNet18 backbone 是最值得优先优化或裁剪的模块之一。

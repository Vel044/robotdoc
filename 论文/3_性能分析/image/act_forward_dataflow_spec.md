# 队列为空时一次完整 ACT 推理数据流图制图说明

## 目标

生成一张用于论文 3.2 节的正式结构图，说明当 `_action_queue` 为空时，一次完整 ACT 推理从 observation 到舵机目标的全过程。

这张图要回答三个问题：

- 输入如何从相机图像和 state 变成模型 batch。
- ACT 模型推理时真正参与计算的结构是什么。
- 输出的 100 帧归一化动作如何进入队列，并最终反归一化为当前舵机目标。

## 输出文件

建议输出为矢量图：

- `robotdoc/论文/3_性能分析/image/act_forward_dataflow.pdf`
- 可先生成 SVG，再转 PDF；不要输出低分辨率位图。

图宽按论文单栏正文宽度设计。推荐横向图，宽高比约 `1.8:1` 到 `2.2:1`。

## 必须表达的事实

当前 ACT 配置：

- 两路相机输入，分辨率 `640×360`
- batch size `B=1`
- `chunk_size = n_action_steps = 100`
- `action_dim = 6`
- 一次完整模型输出 `(1, 100, 6)`

预处理：

- 两路图像来自 observation，原始格式是 HWC `uint8`
- 图像转为 CHW `float32`
- 图像数值归一化到 `[0,1]`
- 增加 batch 维，得到每路图像 `(1, 3, 360, 640)`
- 图像经过 checkpoint 对应的 mean/std 归一化
- state 从 `(6,)` 变为 `(1,6)`，同样经过 mean/std 归一化

模型结构：

- 两路图像分别进入 ResNet18
- 取 ResNet18 `layer4` 特征
- 每路图像得到 `512×12×20` 特征图
- 两路图像共形成 `2×12×20 = 480` 个视觉 token
- 视觉特征加入 2D 位置编码，并经过 `1×1` 投影
- state 经过投影形成 state token
- 推理态 `eval()` 下训练用 VAE encoder 不参与前向
- latent 使用全零向量，形成 latent token
- 视觉 token、state token、latent token 拼接为 `482` 个 token
- token 维度为 `512`
- Transformer Encoder 共 `4` 层
- Transformer Decoder 共 `1` 层
- Decoder 使用 `100` 个 action query
- action head 输出归一化动作 chunk：`(1,100,6)`

后处理：

- 将 100 帧归一化动作写入 `_action_queue`
- 当前帧通过 `popleft()` 取出
- `postprocessor` 将归一化动作反归一化为真实舵机目标
- 返回当前动作 `(6,)`

## 推荐布局

使用三层水平分区，避免箭头交叉。

第一层：输入与预处理

从左到右：

1. `2 路图像 HWC uint8`
2. `CHW float32 + batch`
3. `图像 mean/std 归一化`

同时在这一层或旁边放 state 输入：

1. `state (6,)`
2. `state mean/std 归一化`

第二层：ACT.forward 模型结构

建议分成上下两条支路，最后汇合到 token 拼接：

视觉支路：

1. `两路 ResNet18 layer4`
2. `512×12×20 ×2`
3. `2D 位置编码 + 1×1 投影`
4. `480 视觉 tokens`

state / latent 支路：

1. `state 投影`
2. `latent = 0`
3. 小注释：`推理态 VAE encoder 不运行`
4. `state token + latent token`

汇合后从左到右：

1. `拼接 482 tokens`
2. `Transformer Encoder ×4`
3. `Transformer Decoder ×1`
4. `100 action queries`
5. `action head`
6. `归一化动作 (1,100,6)`

第三层：队列与后处理

从左到右或从右到左均可，但必须清晰：

1. `_action_queue.extend`
2. `缓存 100 帧`
3. `popleft 当前帧`
4. `postprocessor 反归一化`
5. `舵机目标 (6,)`

建议让模型输出在右侧向下进入第三层，然后第三层横向排布，不要画长距离回环线穿过模型节点。

## 视觉风格

颜色建议：

- 输入 observation：蓝色边框、浅蓝底。
- 预处理：橙色边框、浅橙底。
- 模型主体：青蓝色边框、浅青底。
- 队列与后处理：绿色边框、浅绿底。
- `latent=0 / VAE encoder 不运行`：可以用浅黄色或灰色注释框，强调它不是运行中的重计算模块。

要求：

- 箭头不穿过任何文字或节点。
- 箭头不要重叠。
- 三个层次之间要有明显留白。
- 每个节点最多两行文字。
- 代码名太长时用短标签，正文会补充细节。
- 不要把 “VAE encoder” 画成参与推理的模型模块。

## 推荐图中文字

图题：

`队列为空时一次完整 ACT 推理的数据流与模型结构`

建议节点文字：

- `2 路图像`
- `HWC uint8`
- `CHW float32`
- `batch`
- `mean/std`
- `ResNet18 layer4`
- `2D pos + 1×1 proj`
- `480 visual tokens`
- `state token`
- `latent = 0`
- `482 tokens`
- `Encoder ×4`
- `Decoder ×1`
- `action head`
- `(1,100,6)`
- `_action_queue`
- `postprocessor`
- `servo target (6,)`

不要出现：

- `CVAE Encoder`
- `CVAE encoder`
- `trunk`
- `trunk size`

如果必须写 VAE 相关内容，只能写：

`推理态 VAE encoder 不运行，latent=0`

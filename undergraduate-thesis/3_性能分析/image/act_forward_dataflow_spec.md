# 图 2：ACT 推理流程三层图

这张图重点讲 **需要推理时 ACT 里面发生了什么**。快速路径只要一句话带过，不要画 action chunk 队列细节。

## 第一层：快速路径，简单带过

这一层只表达：

```text
_action_queue 非空
    -> 直接 popleft 取 1 帧动作
    -> 返回执行
```

不要在这里画 `chunk_size`，不要画 100 帧队列，不要展开后续 99 帧。

## 第二层：`predict_action()` 包装层

这一层只画模型外面的三步：

```text
原始 observation
    |
预处理
    - numpy -> Tensor
    - 图像 HWC -> CHW
    - uint8 -> float32 / 255
    - 加 batch 维
    - preprocessor 做 state/image mean-std 归一化
    |
select_action()
    - 队列为空时调用 ACT.forward
    - 得到归一化动作
    |
后处理
    - postprocessor 反归一化
    - 返回 1 帧舵机动作 (6,)
```

包装层不要画太复杂。它的作用就是：

```text
预处理 -> select_action -> 后处理
```

## 第三层：`ACT.forward()` 模型内部

这一层是图的重点，要一块块讲清楚。

输入：

```text
state:  (1,6)
images: 2 × (1,3,360,640)
```

模型内部可以按 5 个模块画：

### 1. latent 分支

```text
latent = 0
VAE encoder 推理时不运行
latent 投影到 512 维
```

注意：不要画成 `CVAE encoder` 在参与推理。

### 2. state 分支

```text
state (1,6)
    -> Linear 投影
    -> state token (512维)
```

### 3. 图像分支

```text
两路图像
    -> ResNet18 layer4
    -> 每路输出 (1,512,12,20)
    -> 两路共 480 个视觉 token
    -> 2D 位置编码
    -> 1×1 投影
```

视觉分支是主要计算来源，可以在图里稍微突出。

### 4. Transformer 编码/解码

```text
latent token + state token + 480 image tokens
    -> 拼接成 482 tokens，每个 512 维
    -> Transformer Encoder ×4
    -> Transformer Decoder ×1
       使用 100 个 action query
```

### 5. 动作输出

```text
action head
    -> 输出 action chunk: (1,100,6)
    -> select_action 取当前 1 帧
```

这里可以标注：

- `100` 对应 `chunk_size`
- `6` 对应 `action_dim`
- 最终控制循环只执行其中当前 1 帧动作

## 推荐画法

画成上下三层：

1. 顶层很小：`队列非空 -> 直接取 1 帧`
2. 中层中等：`预处理 -> select_action -> 后处理`
3. 底层最大：`ACT.forward` 内部结构

图的视觉重心应该放在第三层 ACT 模型内部，而不是动作队列。

## 不要出现

- `trunk`
- `trunk size`
- `CVAE encoder`
- 把 VAE encoder 画成推理时运行

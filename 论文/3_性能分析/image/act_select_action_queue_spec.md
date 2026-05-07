# ACT `select_action` 动作队列路径图制图说明

## 目标

生成一张用于论文 3.2 节的正式流程图，说明 ACT 在线推理时 `predict_action()` / `select_action()` 的两条路径：

- 队列非空时，大多数控制帧直接从 `_action_queue` 取缓存动作，不重新跑模型。
- 队列为空时，才执行一次完整 ACT 推理，生成 `chunk_size=100` 帧动作并写入队列。

这张图的核心结论是：在线控制循环中的多数帧不被完整模型推理阻塞，`chunk_size` 是把一次重推理的开销分摊到后续动作帧上。

## 输出文件

建议输出为矢量图：

- `robotdoc/论文/3_性能分析/image/act_select_action_queue.pdf`
- 可先生成 SVG，再转 PDF；不要输出低分辨率位图。

图宽按论文单栏正文宽度设计，推荐横向图，宽高比约 `2.2:1` 到 `2.6:1`。

## 必须表达的事实

当前配置：

- `chunk_size = n_action_steps = 100`
- `action_dim = 6`
- 一次完整模型推理输出 `(B, chunk_size, action_dim) = (1, 100, 6)`
- 完整推理后，队列缓存 100 帧归一化动作，当前帧立刻弹出第 0 帧返回，后续 99 帧走快速路径。

快速路径必须明确：

- `_action_queue` 非空
- `_action_queue.popleft()`
- `postprocessor` 反归一化
- 返回当前舵机目标
- 跳过：numpy 到 Tensor 转换、`preprocessor`、`policy.select_action()`、`predict_action_chunk()`、`ACT.forward()`

完整路径必须明确：

- `_action_queue` 为空
- numpy observation，包括两路图像和 state
- Tensor / 维度转换
- `preprocessor`
- `policy.select_action()`
- `predict_action_chunk()`
- `ACT.forward()`
- 写入 100 帧动作队列
- 弹出第 0 帧
- `postprocessor` 反归一化
- 返回当前舵机目标

## 推荐布局

使用“泳道图”，不要画成复杂网络图。

左侧放两个公共节点：

1. `predict_action()`
2. `检查 _action_queue`

从 `检查 _action_queue` 分成上下两条不交叉的路径：

上方绿色泳道：`队列非空：快速路径`

节点从左到右：

1. `_action_queue.popleft()`
2. `缓存动作`
3. `postprocessor 反归一化`
4. `返回当前动作 (6,)`

上方泳道加一个浅灰小注释：

`跳过 Tensor 转换、preprocessor 与 ACT.forward`

下方橙色泳道：`队列为空：完整推理路径`

节点从左到右，然后可在右侧下折一行，保持箭头不交叉：

1. `numpy obs`
2. `Tensor / 维度转换`
3. `preprocessor`
4. `policy.select_action()`
5. `predict_action_chunk()`
6. `ACT.forward()`
7. `写入 100 帧队列`
8. `弹出第 0 帧`
9. `postprocessor 反归一化`
10. `返回当前动作 (6,)`

如果下方节点太多，可以合并为：

- `select_action → predict_action_chunk`
- `ACT.forward 输出 (1,100,6)`
- `extend 队列 + popleft 第0帧`

## 视觉风格

颜色建议：

- 公共入口和判断节点：蓝色边框、浅蓝底。
- 快速路径：绿色边框、浅绿色底。
- 完整推理路径：橙色边框、浅橙色底。
- 注释文字：灰色。

要求：

- 所有箭头尽量直线或正交折线，严禁箭头互相重叠。
- 文字不要压住箭头。
- 节点之间留足横向间距。
- 不要使用过长英文函数名撑破节点；长函数名可以拆行。
- 不要让图看起来像程序调用栈，本图重点是“两条运行路径”和“多数帧走快速路径”。

## 推荐图中文字

图题：

`ACT 在线推理的动作队列快速路径与完整推理路径`

可出现在图内的短标签：

- `多数帧`
- `队列非空`
- `队列为空`
- `完整推理一次`
- `缓存 100 帧动作`
- `后续 99 帧快速取用`

不要出现：

- `trunk size`
- `trunk`
- `CVAE encoder`

统一使用：

- `chunk_size`
- `VAE encoder 不运行` 只在第二张模型结构图中说明即可，本图可以不写。

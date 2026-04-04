## 1. ACT ：算法 / 模型逻辑

ACT模型 | 斯坦福ALOHA团队 | Action Chunking Transformer
致力于**低价值**硬件发挥出**高精度**操作的性能

不管用什么模型都是经历下面的循环

下面省略了一些细节代码，只保留核心部分
```Python
@safe_stop_image_writer
def record_loop():
    """
    核心控制循环：负责观测获取、动作推理、指令发送及数据保存。
    该循环体现了：观测 -> 推理 -> 动作 -> 执行 -> 等待 的完整链路。
    """
    # 检查数据集帧率与请求帧率是否匹配，确保控制频率一致性
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # 如果提供了策略模型（如 ACT），在开始前重置其内部状态（如 Transformer 的缓存）
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter() # 记录回合开始的绝对时间

    # 主循环：在指定的回合时间内持续运行
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter() # 记录本帧循环的开始时间

        # 检查是否通过键盘等事件触发了提前退出
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # --- ① 获取观测 (Observation) ---
        # 链路：机器人硬件 -> 内核驱动缓冲区 -> Python dict
        observation = robot.get_observation()

        # 如果需要保存数据或执行策略，将原始观测转换为数据集定义的特征格式
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        # --- ② 推理/获取动作 (Action Generation) ---
        if policy is not None:
            # 自主模式：调用神经网络模型（如 ACT）预测动作值
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            # 将张量结果映射回机器人的动作特征键值对
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
   
        # --- ③ 动作发送 (Send Action) ---
        # 链路：Python -> os.write -> TTY/ACM -> USB Controller (DMA) -> 机器人控制器
        # 注意：此处 send_action 在数据提交给内核/硬件队列后即返回，与物理传输并行
        sent_action = robot.send_action(action)

        # --- 数据记录 (Logging) ---
        if dataset is not None:
            # 构建动作帧并将其与观测帧合并，存入数据集缓冲区
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)
        # --- ④ 频率对齐 (Busy Wait) ---
        # 计算本轮逻辑消耗的时间
        dt_s = time.perf_counter() - start_loop_t
        # 忙等：等待直到填满 1/fps 的时长，确保控制频率稳定，--dataset.fps int：Limit the frames per second. (default: 30)，默认 FPS 是 30。
        # 此时硬件正在通过 DMA 并行发送动作包
        busy_wait(1 / fps - dt_s)
        # 更新回合已持续的时间
        timestamp = time.perf_counter() - start_episode_t
```
观测->推理->动作->执行->...的循环

**FPS如果大的话有可能会达到USB发送->推理->发送的循环的极限**

感觉可以从**提升USB发送速度->提升FPS->提升模型速度**来对整体进行优化


```python
def predict_action():
    # 浅拷贝观测数据，避免修改原始数据源
    observation = copy(observation)
    
    with (
        # 开启推理模式：禁用梯度计算，减少内存消耗并加速计算
        torch.inference_mode(),
        # 如果使用 CUDA 且开启了混合精度(AMP)，则启用自动混合精度上下文
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # 遍历所有观测特征（如图像、关节角度等），将其转换为 PyTorch 格式
        for name in observation:
            # 将 NumPy 数组转换为 PyTorch 张量
            observation[name] = torch.from_numpy(observation[name])
            
            # 针对图像数据的特殊预处理：
            if "image" in name:
                # 归一化：将像素值从 [0, 255] 映射到 [0, 1.0] 的 float32 类型
                observation[name] = observation[name].type(torch.float32) / 255
                # 维度转换：将 (H, W, C) 转换为 PyTorch 要求的 (C, H, W) 格式
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            
            # 增加批次维度 (Batch Dimension)，将形状变为 (1, C, H, W)
            observation[name] = observation[name].unsqueeze(0)
            # 将数据移动到计算设备（如 CPU、CUDA 或 MPS）
            observation[name] = observation[name].to(device)

        # 注入任务描述和机器人类型信息（用于多任务策略）
        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # --- 核心推理阶段 ---
        # 调用策略模型的 select_action 方法，根据当前观测计算动作
        # 对于 ACT 模型，这会通过 Transformer 的编码器-解码器生成动作序列
        action = policy.select_action(observation)

        # 移除推理结果中的批次维度
        action = action.squeeze(0)

        # 将生成的动作移动回 CPU 内存，以便后续通过串口(os.write)发送给硬件
        action = action.to("cpu")

    return action

```

模型在 `lerobot/src/lerobot/policies/act/configuration_act.py` 和 `lerobot/src/lerobot/policies/act/modeling_act.py`

参考论文《使用低成本硬件学习细粒度双手动操作》（https://huggingface.co/papers/2304.13705）。


![动作分块](Picture/动作分块ActionThunking.png)


1. 上半部分：Action Chunking（动作分块）

- 一次预测，一段动作（t=0的时候预测0-4的动作，t=4的时候预测4-8的动作）
- 固定间隔更新：机器人会依次执行这 4 个动作。直到 t=4 时，模型才会再次进行观测并预测下一个动作块。
- 局限性：这种方式虽然减少了推理频率，但在两个动作块的交界处（例如 t=3 和 t=4 之间）可能会出现动作不连贯或抖动。


2. 下半部分：Action Chunking + Temporal Ensemble（时序集成）

是 ACT 实现高精度和平滑度的核心考虑。它不再是每隔几步更新一次，而是**每一帧都更新，但利用重叠的预测**，其实是**用计算压力换动作平滑**了。

- 滑动窗口式的预测：
  - 在 t=0 时，预测一组序列。
  - 在 t=1 时，再次观测并预测一组新的序列。注意图中 t=1 的序列相对于 t=0 向右滑动了一格。
- 指数加权平均（加权求和）
  - 系统会给这些重叠的动作分配权重（如图中显示的 [0.5, 0.3, 0.2, 0.1]）。越早预测的动作权重越高，通过加权平均得到最终执行的动作。

![ACT架构](Picture/ACT架构.png)

1. 左侧：训练阶段的 CVAE 编码器 (Encoder)

- 输入：包含当前时刻的机器人**关节位置**（joints）和完整的**专家动作序列**（action sequence）。
- 输出：压缩成一个风格变量 z（style variable）。
- 这个编码器**仅在训练时使用**，推理的时候Encoder会被删除

2. 右侧：推理阶段的策略网络 (Decoder/Policy)

- 输入：多视角图像、关节状态、风格变量 z。
- 通过一个Transformer-Encoder和一个Decoder
- 输出：一次性预测出一个动作分块（action sequence/chunk），未来几步的。

![训练和推理](Picture/训练和推理.png)

#### 训练
1. 训练集：图片+动作序列（本质是模仿学习）
2. 推理变量z：将动作序列和关节数据映射到隐藏维度（512维）。经过带有 4 个自注意力块的 Transformer 编码器，推导出 z 的均值和标准差，并通过重参数化技巧采样出 z。
3. 预测动作序列：图像通过 ResNet18 提取特征并平坦化（flatten）。视觉特征、关节数据和 z 一起进入策略的 Transformer 编码器。Transformer 解码器（带有 7 个交叉注意力块）输出预测的动作序列。

#### 推理
- 与训练时的“预测”部分几乎一致，唯一区别是 z 直接设为 0。
- 模型根据当前的实时观测（图像和关节），一次性预测未来的一组动作序列。



#### 这是其中一个模型，也可以换成Diffusion/Pi0等模型。

### 内核编译安装

现在是可以原生编译安装了

![内核编译安装](Picture/内核编译安装.png)



## 2. ACT 模型推理原理详解


### 2.1. 输入：环境观测数据（Observation）

#### 图像输入：这当前模型中就是两个摄像头的图像输入
#### 关节状态输入：当前模型中就是机械臂的关节角度和速度
#### 风格变量 z：在推理时，ACT 不再使用训练时的 CVAE 编码器。为了保持确定性，它直接给解码器喂入一个全为 0 的向量（即先验分布的均值）


### 2.2. 模型处理 （模型处理从输入到输出的过程在这里）


### 2.3. 动作输出（Action Output）

模型输出一个长度为 k（通常为 100）的动作序列。这个序列包含了接下来 k 个时间步的关节目标位置 。

### 2.4. 时间集成（Temporal Ensembling）

就和之前的示意图一样，有一个滑动窗口的重叠覆盖和加权平均的过程，目的是为了动作平滑。



## 3. 对其推理时间进行详细解析

这边用的推理参数是每次推理 **100** 个动作，推理一次要使用**1.6s**，执行的**FPS大约为20**，所以推理一次执行**5s**

```bash
FPS: 0.59 | Obs: 17.13ms | Infer: 1678.52ms | Act: 1.27ms | Wait: 0.04ms
FPS: 17.04 | Obs: 5.57ms | Infer: 49.94ms | Act: 2.88ms | Wait: 0.11ms
FPS: 18.75 | Obs: 5.31ms | Infer: 46.50ms | Act: 1.51ms | Wait: 0.02ms
FPS: 18.67 | Obs: 5.59ms | Infer: 44.82ms | Act: 2.23ms | Wait: 0.43ms
FPS: 19.90 | Obs: 6.21ms | Infer: 41.77ms | Act: 1.97ms | Wait: 0.22ms
FPS: 19.23 | Obs: 6.21ms | Infer: 43.23ms | Act: 2.54ms | Wait: 0.01ms
FPS: 2.82 | Obs: 6.51ms | Infer: 344.63ms | Act: 2.43ms | Wait: 0.01ms
FPS: 19.74 | Obs: 6.18ms | Infer: 41.71ms | Act: 2.10ms | Wait: 0.62ms
FPS: 18.77 | Obs: 6.09ms | Infer: 44.71ms | Act: 2.45ms | Wait: 0.01ms
FPS: 19.31 | Obs: 5.65ms | Infer: 43.72ms | Act: 2.41ms | Wait: 0.01ms
FPS: 20.90 | Obs: 4.63ms | Infer: 40.70ms | Act: 2.13ms | Wait: 0.01ms
FPS: 9.39 | Obs: 6.02ms | Infer: 98.50ms | Act: 1.95ms | Wait: 0.01ms
FPS: 20.95 | Obs: 4.70ms | Infer: 40.59ms | Act: 1.54ms | Wait: 0.86ms
FPS: 19.84 | Obs: 4.80ms | Infer: 43.67ms | Act: 1.68ms | Wait: 0.22ms
FPS: 18.96 | Obs: 5.01ms | Infer: 44.95ms | Act: 2.75ms | Wait: 0.01ms
FPS: 19.99 | Obs: 5.22ms | Infer: 43.46ms | Act: 1.22ms | Wait: 0.11ms
FPS: 9.07 | Obs: 5.72ms | Infer: 102.71ms | Act: 1.81ms | Wait: 0.01ms
FPS: 20.91 | Obs: 5.94ms | Infer: 39.41ms | Act: 1.91ms | Wait: 0.55ms
FPS: 19.99 | Obs: 5.32ms | Infer: 42.01ms | Act: 2.52ms | Wait: 0.16ms
FPS: 19.84 | Obs: 6.20ms | Infer: 41.96ms | Act: 2.23ms | Wait: 0.01ms
FPS: 20.33 | Obs: 5.00ms | Infer: 42.18ms | Act: 1.92ms | Wait: 0.01ms
FPS: 8.68 | Obs: 3.51ms | Infer: 108.95ms | Act: 2.72ms | Wait: 0.01ms
FPS: 19.95 | Obs: 5.39ms | Infer: 42.72ms | Act: 1.51ms | Wait: 0.49ms
```

执行的帧率在20左右，所以5s进行一次大infer，轻帧有一个很大的infer时间，因为哪怕不推理，他还是要把一大堆数据做处理，



## 4. 具体怎么预测100帧动作

```py
def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
    """预测一个完整的动作块 (Action Chunk)。"""

    # 1. 标准化输入 (归一化到 Gaussian 分布)
    batch = self.normalize_inputs(batch)
    # ... (处理图像键值)

    # 2. 模型前向传播 (Heavy Lifting)
    # 这里的 self.model 就是下面的 ACT 类,会创建一个ACT实例
    actions = self.model(batch)[0] 

    # 3. 反标准化输出 (恢复到机器人的物理单位，如弧度)
    actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
    return actions

```

ACT 的架构在深度学习中属于 **Encoder-Decoder（编码器-解码器）** 架构。它的核心任务是实现一个映射函数 ：

Action一百个点=F(Images,JointState,Latent 潜变量 z)

z在训练时，它捕捉动作的随机性（比如抓杯子时手抖动的幅度）。
在推理时（正如你之前看到的 latent_sample = torch.zeros），它通常被置为0。这意味着要求 F 输出**最标准、最确定**的动作。


模型在训练：[VAE Encoder] + [Transformer Encoder] + [Transformer Decoder]
模型在推理：[Transformer Encoder] + [Transformer Decoder]

### ACT 推理模型结构
### 第一阶段：数据向量化与序列构建 (Tokenization)


### 第二阶段：编码器 (Encoder) —— 自注意力机制 (Self-Attention)

Encoder 的作用是计算特征之间的**相关性矩阵**，并进行加权聚合。

**Encoder 输出**：一个形状为 [N,512] 的矩阵（我们记为 **Memory**）。它包含了经过全局推理后的环境理解。


### 第三阶段：解码器 (Decoder) —— 交叉注意力机制 (Cross-Attention)

这是 ACT 实现“一次预测 100 帧”的核心。Decoder 的结构与 Encoder 略有不同。

1. **输入是什么？ (Learned Queries)**
* 输入是一个**固定参数矩阵**，代码中叫 `decoder_pos_embed`，形状为[100,512] 。
* 这 100 个向量在推理开始时是**初始化状态**，它们不包含任何具体的图像信息，只包含“我代表第几个时间步”的位置信息。我们称之为 **Object Queries**。


2. **并行计算**：
* 注意，这 100 个 Query 是作为一个矩阵 [100,512] 一起参与矩阵乘法的。
* 因此，第 0 帧到第 99 帧的特征提取是**同时**完成的，不存在先后依赖。


3. **Decoder 输出**：
* 经过多层 Cross-Attention 后，那 100 个原本只有位置信息的向量，吸纳了图像和状态信息，变成了 100 个**包含了动作意图的高维特征向量**。

### 第四阶段：预测头 (Prediction Head) —— 线性映射

这是最后一步，将高维特征映射回物理空间。

* **输入**：Decoder 输出的矩阵，形状[100,512] 。
* **运算**：一个简单的全连接层（MLP）。
* 公式：Y=XWT+b
* 其中 W 的形状是 [14,512]（（假设机械臂有 14 个自由度）。


* **输出**：形状[100,14] 。
* 这就是 100 帧的关节角度数据。


这就是为什么它能一次性算出 100 帧：因为在数学上，这就是一个**批量的矩阵乘法操作**，100 个时间步在矩阵运算中是并行的维度。


模型参数：vision_backbone: str = "resnet18"：视觉皮层用 `ResNet18`，是一种经典的CNN网络
输入：一张 480 x 640 的 RGB 图片（几十万个像素点）。
处理：通过 18 层卷积运算，提取出边缘、纹理、形状等信息。
输出：一个 15 x 20 x 512 的特征图（Feature Map）。
后续：这个特征图会被展平，变成 300 个长度为 512 的 Token，喂给 Transformer

n_encoder_layers: int = 4：Encoder 有 `4` 层。
n_decoder_layers: int = 1：Decoder 有 `1` 层。

n_heads: int = 8：多头注意力有 `8` 个头。Transformer 内部的 512 维总线被拆分成了 8 组，每组 64 维。

dropout: float = 0.1：丢弃率为 `10%`。
optimizer_lr: float = 1e-5：学习率（步长）。

encoder的矩阵运算是4层是串行，但是矩阵运算torchcpu自动会并行
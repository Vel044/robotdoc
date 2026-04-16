# motors_bus 异步读/写设计分析

> 关于 `lerobot/src/lerobot/motors/motors_bus.py` 里为什么**读**有异步远期规划、
> 而**写**完全没有对应规划的完整说明。

## 一、结论速览

| 维度 | `sync_read` | `sync_write` |
|------|-------------|--------------|
| 上游是否留有异步占位 | ✅ 有（`_async_read` TODO） | ❌ 完全没有 |
| 规划采用的异步形式 | 流水线 pipelining | — |
| 是否依赖后台线程 | 否 | — |
| 主要收益 | 隐藏串口往返延迟 | — |
| 主要代价 | policy 用的观测延迟 1 帧 | — |
| 当前在本 fork 的状态 | 仅注释占位，未实现 | — |

一句话：**上游对"读"留了条优化路径**（以牺牲 1 帧观测新鲜度换取吞吐），
**对"写"没有任何优化路径**，因为做异步写对这个系统既不安全也不解决真实瓶颈。

---

## 二、上游 TODO 原文

位于 upstream `huggingface/lerobot` 的 [motors_bus.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py)
（本仓库已同步到 `_setup_sync_reader` 与 `sync_write` 之间）：

```python
# TODO(aliberts, pkooij): Implementing something like this could get even much faster read times if need be.
# Would have to handle the logic of checking if a packet has been sent previously though but doable.
# This could be at the cost of increase latency between the moment the data is produced by the motors and
# the moment it is used by a policy.
# def _async_read(self, motor_ids: list[int], address: int, length: int):
#     if self.sync_reader.start_address != address or self.sync_reader.data_length != length or ...:
#         self._setup_sync_reader(motor_ids, address, length)
#     else:
#         self.sync_reader.rxPacket()
#         self.sync_reader.txPacket()
#
#     for id_ in motor_ids:
#         value = self.sync_reader.getData(id_, address, length)
```

TODO 只针对 **read**，`sync_write` 定义前后都没有类似注释或占位方法。

---

## 三、为什么"异步读"可以有规划？

### 1. 上游方案的本质：流水线，不是多线程

上游规划里的"异步"跟 camera 的 `async_read` 后台线程模式**完全不一样**：

- **Camera async_read**（`cameras/opencv/camera_opencv.py:_read_loop`）：
  后台线程死循环地 `cv2.VideoCapture.read()`，主线程调用 `async_read()` 直接取最新帧。
  每台相机独占一个 USB 口，后台线程不会和任何东西抢总线。

- **Motors _async_read (TODO)**：
  **没有后台线程**。主线程里把 Feetech SDK 的 `txRxPacket()`（发请求 + 等回包）
  拆成两步：
  1. 本次调用先 `rxPacket()` **接收上一次调用发出的请求的回包**
  2. 然后 `txPacket()` **发出这一次的新请求**
  3. 下次调用再来收这一次的回包

  本质是**把串口往返的等待时间隐藏到两次调用之间**（流水线 / pipelining）。

### 2. 为什么这种流水线对读安全可行

- **读请求是幂等的**：重复读 `Present_Position` 不会产生任何副作用。
  即使预取的数据因为参数变了要丢弃，也没有任何物理后果。
- **bus 依然是单线程访问**：没有并发，没有锁，不会和 `sync_write` 撞车，
  因为整个流水线只发生在主线程的 `_async_read` 一个方法里。
- **粒度可控**：TODO 注释里明确写了"要判断上次发的包参数（地址、长度等）
  跟这次是否一致"，不一致就退化成普通 `_sync_read`。

### 3. 代价：观测延迟 1 帧

TODO 注释的这句是关键：

> This could be at the cost of increase latency between the moment the data
> is produced by the motors and the moment it is used by a policy.

也就是说，主线程第 N 次调用 `_async_read` 拿到的是**第 N-1 次请求时电机的状态**。
对 policy 来说，"此刻的观测"其实比实际晚了一整帧。

在树莓派 5 + ACT 这种推理主导（inference ~1200ms/frame at cs=1）的场景下，
省下那点串口时间（1-2ms）**远小于**推理延迟本身，且多出 1 帧观测滞后反而会
让 policy 的控制质量下降。所以上游只留了 TODO 没实现，也合情合理。

---

## 四、为什么"异步写"没有任何规划？

这是本文档的核心。**不是上游忘了，是不能做。**

### 1. 对"异步"的澄清：机器人什么时候开始动？

一个关键的认知纠偏——`sync_write` 和假想的 `async_write`，**机器人开始运动的
时刻完全相同**，差别只是 Python 线程阻塞多少毫秒：

| 方式 | Python 线程阻塞 | 舵机开始运动的时刻 |
|------|----------------|--------------------|
| `sync_write` | 阻塞 ~1-2ms 直到串口字节发完 | 收到数据包就开始动 |
| 假想 `async_write` | 0 ms，立即返回 | 收到数据包就开始动（完全不变） |

所以常见误区"异步写会让机器人一边推理一边动"**是错的**——不管同步异步，
机器人都在 `send_action` 那一刻开始动。要解决"推理期间世界变化"的问题，
真正的手段是 **chunked action**（一次预测 N 步未来动作，cs > 1），
不是改写 bus 层。

### 2. 真正不能做异步写的原因

#### 原因 A：串口是半双工共享资源（决定性因素）

Feetech 舵机走的是 TTL 半双工串行线，所有电机**共用一根物理总线**。
同一时刻总线上只能有一个事务在进行。

- camera async_read 能做，是因为**每台相机独占自己的 USB 接口**，
  后台线程不停 read 对别人没有任何影响。
- motors 如果开后台线程异步写，主线程同时在 `sync_read`，
  两个请求会在同一根线上撞车，导致：
  - 字节流交错 → 回包校验失败 → 整帧数据废掉
  - 或者某个电机收到了畸形包 → 直接忽略（最好情况）或误动作（灾难）

如果加锁保护，两个线程轮流持锁 → 退化成纯串行 → 等于没异步，
还白白多了锁竞争和上下文切换。

> **对比**：上游的 `_async_read` pipelining 之所以成立，是因为它**全程在主线程里**，
> 不和任何其他总线事务并发。它隐藏的是"请求-响应"的等待时间，不是 Python 层的阻塞。

#### 原因 B：读改写依赖链被打破

看 [so101_follower.py](/Users/vel/Work/RobotOS/Lerobot/lerobot/src/lerobot/robots/so101_follower/so101_follower.py) 里的 `send_action`：

```
goal_pos = parse(action)
if max_relative_target is not None:
    present_pos = bus.sync_read("Present_Position")   # 读当前位置
    goal_pos   = ensure_safe_goal_position(goal, present_pos)   # 限幅
bus.sync_write("Goal_Position", goal_pos)             # 写目标位置
```

这是一条**强依赖链**：read → 计算 → write。如果 write 异步化：

- 下一帧的 `sync_read("Present_Position")` 可能和上一帧尚未完成的后台 `async_write`
  在总线上冲突（原因 A）。
- 下一帧读到的 `Present_Position` 是电机正在执行前一个 goal 的中间状态，
  `ensure_safe_goal_position` 的限幅基准变得不确定。

也就是说，安全限幅的前提是"读-写"严格串行。异步写一旦引入，整个安全机制的
正确性假设就塌了。

#### 原因 C：错误暴露时机延后（安全性）

`_sync_write` 失败时立即抛 `ConnectionError`：

```python
if not self._is_comm_success(comm) and raise_on_error:
    raise ConnectionError(f"{err_msg} ...")
```

调用方（`robot.send_action`）能立刻知道"这一帧指令没送达"，可以停机、
重试或报警。

如果换成异步写：
- 主线程 `send_action` 已经返回成功，policy 继续往后算
- 后台线程几十毫秒后才发现包掉了
- 此时机械臂可能卡在某个半执行的姿态，policy 还在基于"以为执行成功了"
  的假设输出下一条指令
- 错误反馈回到用户代码可能已经过了好几帧

对机器人安全场景这是**非常忌讳的错误延迟暴露**。

#### 原因 D：省的时间不在瓶颈上

这是工程层面的现实考虑。看当前的实际耗时分布（见
[analysis/timing_stats.csv](/Users/vel/Work/RobotOS/Lerobot/lerobot/analysis/timing_stats.csv)）：

| chunk_size | inference ms/frame | sync_write ms/frame |
|-----------:|-------------------:|--------------------:|
|          1 |              ~1200 |                 ~1-2 |
|         10 |               ~140 |                 ~1-2 |
|        100 |                ~20 |                 ~1-2 |

异步化能省下的最多就是那 1-2ms 的 `txPacket` 阻塞时间。但：

- cs=1 时推理占了 99% 时间，省 2ms 没有意义
- cs=100 时外层还有 FPS 限制（比如 30Hz = 33ms/帧），写完要 sleep
  等下一帧才触发下一条动作，省 2ms 更没意义

**只有纯实时控制系统**（没有视觉、没有大模型，频率 500Hz+）才需要
抠这 2ms。在 lerobot 的使用场景里完全不是瓶颈。

### 3. 反面：如果硬要做异步写，成本有多大？

综合以上，最小可行的异步写需要：

1. 一个专用的写线程 + 写队列
2. 一把覆盖整个 bus 的读写锁（否则原因 A）
3. 一套错误回调机制把后台失败汇报回主线程（原因 C）
4. `send_action` 内部的 read-modify-write 要改成"读完直接算完写完"
   保持原子性，等于主线程还是要阻塞一次（原因 B），异步化只对"不带安全限幅
   的直接 write"生效，收益进一步被挤压

最终能异步化的只是 `max_relative_target=None` 这一条冷门路径上的 1-2ms，
代价是整个 bus 层引入并发复杂度。ROI 显然是负的，所以上游连 TODO 都没留。

---

## 五、对本项目（树莓派 5 + ACT）的实际建议

基于上面的分析，对你的 use case：

1. **不要试图实现 `_async_write`**。收益 < 2ms，会把 `max_relative_target` 的
   安全机制搞脆弱。
2. **也不一定要实现上游的 `_async_read` 规划**。在 cs>1 的场景里，
   观测新鲜度（实时性）比吞吐更重要；而 cs=1 时推理本身是瓶颈，
   省下的串口时间可以忽略。
3. **真正有效的优化方向**：
   - 增大 chunk_size（把推理延迟摊到多帧上，见 `analysis/timing_chart.*.png`）
   - 推理侧优化（模型量化、ONNX/NCNN 移植、ARM NEON）
   - 如果要做 bus 层优化，优先是**减少每帧 `sync_read` 次数**
     （例如把安全限幅的 read 和观测的 read 合并为一次）
4. **如果将来真的需要**，可以先实现上游 TODO 里的 pipelined `_async_read`，
   观察是否能稳定省下串口时间。异步写不建议碰。

---

## 六、参考位置

- 本 fork 的占位注释：`lerobot/src/lerobot/motors/motors_bus.py`
  （`_setup_sync_reader` 方法之后、`sync_write` 方法之前）
- 上游原始 TODO：https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py
- 安全限幅逻辑：`lerobot/src/lerobot/robots/so101_follower/so101_follower.py`
  的 `send_action` 方法
- Camera async_read 后台线程对照组：
  `lerobot/src/lerobot/cameras/opencv/camera_opencv.py` 的 `_read_loop`
- 性能数据支撑：`lerobot/analysis/timing_stats.csv`、
  `lerobot/analysis/chart1_time_pct.png` 等

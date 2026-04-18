# Async Inference:推理与执行流水线化分析

> 为什么"让机器人执行 chunk 的同时推理下一个 chunk"听起来很美,但在
> 树莓派 5 + cs=100 + ~2s 推理延迟的场景下,收益会被大幅抵消?
>
> 本文从时间线出发,一步步展示 async inference 的工作机制、隐含代价,
> 以及在本项目当前硬件条件下的可行性。

---

## 一、问题的起点

看 [timing_stats.csv](../../lerobot/analysis/timing_stats.csv) 里 `chunk_size=100` 的典型一帧:

```
推理 (inference) → 执行 (action/wait) → 推理 → 执行 → ...
     ~2000 ms         ~3300 ms         ~2000 ms   ~3300 ms
```

**直观念头**:执行那 3.3 秒机器人只是在走路,CPU 很闲,为什么不趁这段时间
**提前把下一个 chunk 推理出来**?

这就是 async inference 的出发点。lerobot 官方在
[scripts/server/robot_client.py](../../lerobot/src/lerobot/scripts/server/robot_client.py)
和 [scripts/server/policy_server.py](../../lerobot/src/lerobot/scripts/server/policy_server.py)
里已经实现了这个机制。本文先讲它怎么工作,再讲它**什么时候真的有用**。

---

## 二、同步 vs 异步:时间线对比

### 同步模式(当前 `record.py` 的行为)

以 30 FPS、cs=100、推理延迟 2s 为例:

```
时刻(s): 0         2        5.3       7.3      10.6      12.6
         │         │         │         │         │         │
         ├─推理 A──┤├───执行 chunk A(100帧)──┤├─推理 B──┤├───执行 chunk B──┤
         │    2s   │        3.3s           │    2s   │        3.3s
         │         │                       │         │
         └─ CPU 忙 ┘└─ CPU 空闲(IO + sleep)─┘

一个完整循环 = 推理 + 执行 = 5.3s / 100 帧
实际 FPS = 100 / 5.3 ≈ 19 FPS(低于目标 30 FPS)
```

问题很明显:**执行那 3.3s 里 CPU 几乎没事做**,浪费了算力。

### 异步模式(async inference 理想图)

```
时刻(s): 0       2       4       6       8      10      12
         │       │       │       │       │       │       │
主执行:  ├───执行 chunk A──┤├──执行 chunk B──┤├──执行 chunk C──┤
         │      3.3s      │      3.3s       │      3.3s
后台推理:├推理B┤  ├推理C┤  ├推理D┤  ├推理E┤
          2s      2s       2s      2s

稳态 FPS = 30(由执行速度决定,不再受推理拖累)
```

听起来完美:**吞吐从 19 FPS 涨到 30 FPS,提升 ~60%**。

但是——**这张图成立的前提是"一次推理产出的 100 帧全都有用"**。
而这正是本文第五节要戳破的幻觉。

---

## 三、async inference 的架构(lerobot 官方实现)

lerobot 的做法是把"推理"和"执行"拆成两个进程,用 gRPC 通信:

```
┌──────────── Robot Client(主线程)────────────┐    ┌─── Policy Server ───┐
│                                                │    │                      │
│  control_loop():                               │    │                      │
│    while running:                              │    │                      │
│      ① if action_queue 有东西:                 │    │                      │
│           pop 一个 action → 执行               │    │                      │
│                                                │    │                      │
│      ② if 队列剩余 ≤ chunk_size_threshold:     │    │                      │
│           采观测 ─── gRPC 发出 ───────────────►│    │  收到观测             │
│           (不等待,立即回到 ①)                 │    │    ↓                │
│                                                │    │  policy.predict()   │
│      ③ sleep 到下一个 tick(维持 FPS)          │    │  (~2s on RPi5)      │
│                                                │    │    ↓                │
│  后台线程 receive_actions():                   │◄───│  返回 100 帧动作    │
│    while running:                              │    │                      │
│      从 gRPC 收 action_chunk                  │    │                      │
│      aggregate 到 action_queue                 │    │                      │
│                                                │    │                      │
└────────────────────────────────────────────────┘    └──────────────────────┘
```

两个进程各跑各的,通过一个线程安全的 `action_queue` 交换数据。

### 3.1 核心数据结构（helpers.py）

理解两个文件之前,先看贯穿始终的数据结构:

```python
# ─── 时间基准（所有带时间信息的数据都继承它）───
@dataclass
class TimedData:
    timestamp: float   # unix 时间戳，用于跨进程计算网络延迟
    timestep: int      # 全局帧号，用于去重和 temporal ensemble 对齐

# ─── 带时间的 action（server → client）───
@dataclass
class TimedAction(TimedData):
    action: Action     # Tensor, shape=(action_dim,)，各关节目标位置（度）

# ─── 带时间的观测（client → server）───
@dataclass
class TimedObservation(TimedData):
    observation: RawObservation  # dict[str, Tensor/ndarray]
                                 #   "observation.images.cam_high": (H,W,3) uint8
                                 #   "observation.state": (action_dim,) float64
    must_go: bool = False        # True 时 server 立刻推理，不排队

# ─── client 发给 server 的初始化配置 ───
@dataclass
class RemotePolicyConfig:
    policy_type: str              # "act"
    pretrained_name_or_path: str  # 模型路径
    lerobot_features: dict        # 输入输出特征描述
    actions_per_chunk: int        # = chunk_size，每次推理返回多少帧
    device: str = "cpu"           # 推理设备
```

### 3.2 RobotClient 初始化（robot_client.py `__init__`）

Client 启动时连接硬件、建立 gRPC channel、初始化状态:

```python
class RobotClient:
    def __init__(self, config: RobotClientConfig):
        # 1. 连接机器人硬件（打开串口、初始化电机和相机）
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        # 2. 建立 gRPC channel，用于和 policy_server 通信
        self.stub = services_pb2_grpc.AsyncInferenceStub(
            grpc.insecure_channel(self.server_address, ...)
        )

        # 3. 关键状态变量
        self.latest_action = -1        # 已执行到的最新帧号（-1=还没执行任何帧）
        self.action_chunk_size = -1    # server 每次 chunk 的帧数（首次收到后更新）

        # action_queue: 两线程共享的缓冲，存 TimedAction
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()

        # start_barrier: 保证主线程和后台线程同时启动
        self.start_barrier = threading.Barrier(2)

        # must_go: 队列空时强制推理的开关
        self.must_go = threading.Event()
        self.must_go.set()
```

### 3.3 握手流程（client `start()` → server `Ready()` + `SendPolicyInstructions()`）

Client 调用 `start()` 完成 gRPC 握手和模型加载:

```python
# ── client 端 ──
def start(self):
    # 第一步：握手，确认 server 已就绪
    self.stub.Ready(services_pb2.Empty())

    # 第二步：把模型配置发给 server
    policy_config_bytes = pickle.dumps(self.policy_config)
    self.stub.SendPolicyInstructions(
        services_pb2.PolicySetup(data=policy_config_bytes)
    )
```

```python
# ── server 端 ──
def SendPolicyInstructions(self, request, context):
    policy_specs = pickle.loads(request.data)   # → RemotePolicyConfig

    # 根据类型（如 "act"）拿到 Policy 子类，加载模型权重
    policy_class = get_policy_class(self.policy_type)
    self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
    self.policy.to(self.device)                 # 移到 cpu/cuda/mps
```

### 3.4 主控制循环（client `control_loop()`）

这是整个 client 的心脏。每个 tick（~33ms @30FPS）执行一次:

```python
def control_loop(self, task, verbose=False):
    self.start_barrier.wait()  # 等后台线程就绪

    while self.running:
        tick_start = time.perf_counter()

        # ① 从 action_queue pop 一帧发给舵机
        if self.actions_available():
            self.control_loop_action(verbose)

        # ② 队列剩余不足时，采观测发给 server 触发推理
        if self._ready_to_send_observation():
            self.control_loop_observation(task, verbose)

        # ③ sleep 补齐帧间隔，维持恒定 FPS
        time.sleep(max(0, self.config.environment_dt - elapsed))
```

### 3.5 执行 action（client `control_loop_action()`）

```python
def control_loop_action(self, verbose=False):
    with self.action_queue_lock:
        timed_action = self.action_queue.get_nowait()  # 非阻塞取队首

    # tensor → dict{"shoulder_pan": 45.0, ...} → 串口 sync_write → 舵机运动
    self.robot.send_action(
        self._action_tensor_to_action_dict(timed_action.get_action())
    )

    with self.latest_action_lock:
        self.latest_action = timed_action.get_timestep()  # 更新已执行帧号
```

### 3.6 采观测并发送给 server（client `control_loop_observation()` → server `SendObservations()`）

```python
# ── client 端 ──
def control_loop_observation(self, task, verbose=False):
    # 1. 从硬件采集一帧观测（相机 async_read + 舵机 sync_read）
    raw_observation = self.robot.get_observation()
    raw_observation["task"] = task

    # 2. 打包成 TimedObservation，附上时间戳和当前帧号
    observation = TimedObservation(
        timestamp=time.time(),
        observation=raw_observation,
        timestep=max(self.latest_action, 0),  # 告诉 server "在第 N 帧后采的"
    )

    # 3. must_go 标志：队列空 + 事件已 set → 强制推理
    with self.action_queue_lock:
        observation.must_go = self.must_go.is_set() and self.action_queue.empty()

    # 4. 非阻塞发送（不等推理结果）
    self.send_observation(observation)
```

```python
# ── server 端 ──
def SendObservations(self, request_iterator, context):
    # 拼接分块字节 → pickle 反序列化为 TimedObservation
    received_bytes = receive_bytes_in_chunks(request_iterator, ...)
    timed_observation = pickle.loads(received_bytes)

    # 尝试入队（可能被过滤：帧号重复 / 和上次太像）
    self._enqueue_observation(timed_observation)
```

### 3.7 server 端推理并返回（server `GetActions()`）

```python
# ── server 端 ──
def GetActions(self, request, context):
    try:
        # 阻塞等待 observation_queue 里有一条观测（超时返回空包）
        obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)

        # 完整推理流水线：预处理 → predict → 后处理
        action_chunk = self._predict_action_chunk(obs)  # → list[TimedAction]

        # pickle 序列化后返回给 client
        return services_pb2.Actions(data=pickle.dumps(action_chunk))

    except Empty:
        return services_pb2.Empty()  # 超时无观测，返回空包
```

`_predict_action_chunk` 内部:

```python
def _predict_action_chunk(self, observation_t):
    # 1. 预处理：RawObservation → policy 格式（缩放、归一化、加 batch 维度）
    observation = self._prepare_observation(observation_t)

    # 2. 推理
    action_tensor = self._get_action_chunk(observation)
    #   → policy.predict_action_chunk(observation)
    #   → shape: (1, chunk_size, action_dim) 截取前 actions_per_chunk 帧

    # 3. 后处理：CPU 化 + 包装成 TimedAction 列表
    action_tensor = action_tensor.cpu().squeeze(0)   # (chunk_size, action_dim)
    return self._time_action_chunk(
        observation_t.get_timestamp(),    # 基准时间戳
        list(action_tensor),              # 每帧一个 Tensor
        observation_t.get_timestep()      # 基准帧号
    )
```

### 3.8 后台线程拉取 action（client `receive_actions()`）

```python
# ── client 端后台线程 ──
def receive_actions(self):
    self.start_barrier.wait()

    while self.running:
        # 阻塞等待 server 返回（server 推理完才回复）
        actions_chunk = self.stub.GetActions(services_pb2.Empty())

        if len(actions_chunk.data) == 0:
            continue  # server 超时返回空包，重试

        timed_actions = pickle.loads(actions_chunk.data)  # → list[TimedAction]
        self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

        # temporal ensemble：和队列里已有的 action 融合
        self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)

        # 收到 action 后重置 must_go（为下次空队列做准备）
        self.must_go.set()
```

---

## 四、三个关键设计

### 1. `chunk_size_threshold`——不等队列空就提前推理

```python
# ── client 端 ──
def _ready_to_send_observation(self):
    """触发条件：队列剩余比例 ≤ threshold"""
    with self.action_queue_lock:
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

cs=100、`threshold=0.5` 时:**队列剩 50 个 action 就开始推下一个 chunk**。
设计意图是让"新 chunk 到货的时刻"正好接上"旧 chunk 耗尽的时刻",避免 stall。

注意：`action_chunk_size` 初始为 `-1`，首次 chunk 到货前 `qsize/(-1)` 为负数，
自动 ≤ threshold，等价于"一开始就触发推理"。这是正确的冷启动行为。

### 2. Temporal Ensemble——多 chunk 融合

新 chunk 到达时,不是简单替换旧队列,而是**按时间戳对齐**。
核心实现在 `_aggregate_action_queues`:

```python
# ── client 端 ──
def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
    # aggregate_fn(x1, x2): x1=旧预测, x2=新预测。默认取 x2（新覆盖旧）
    # 典型替代：weighted_average，对同一时刻多次预测加权平均，减少抖动

    current_action_queue = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}

    for new_action in incoming_actions:
        # 情况 1：帧号 ≤ 已执行帧号 → 已过期，直接丢弃
        if new_action.get_timestep() <= self.latest_action:
            continue

        # 情况 2：帧号不在旧队列 → 纯未来帧，直接入队
        elif new_action.get_timestep() not in current_action_queue:
            future_queue.put(new_action)

        # 情况 3：帧号重叠 → temporal ensemble，融合两次预测
        else:
            future_queue.put(TimedAction(
                timestep=new_action.get_timestep(),
                action=aggregate_fn(
                    current_action_queue[new_action.get_timestep()],  # 旧的
                    new_action.get_action()                            # 新的
                ),
            ))

    self.action_queue = future_queue  # 原子替换
```

图示:

```
队列里的 chunk A:  [f60, f61, ..., f99]        (chunk A 的尾巴,40 帧)
                       ↓   ↓         ↓
新来的 chunk B:    [f60, f61, ..., f99, f100, ..., f129]
                       ↓   ↓         ↓    ↓           ↓
融合后:            [avg, avg, ..., avg, f100, ..., f129]
                    └── 和老预测加权平均 ──┘└── 纯新增 ──┘
```

这是 ACT 原论文提出的 temporal ensemble——不同 chunk 对**同一未来时刻**的多次预测
取加权平均,作用是让动作更平滑、减少 chunk 切换时的 jitter。

### 3. `must_go` 事件——防止队列真的空掉

如果推理实在赶不上,队列会空。`must_go` 是兜底机制,涉及三个位置:

```python
# ── client 端：采观测时设置 must_go ──
def control_loop_observation(self, task, verbose=False):
    ...
    with self.action_queue_lock:
        # 仅当 must_go 事件已 set 且队列为空时，标记强制推理
        observation.must_go = self.must_go.is_set() and self.action_queue.empty()

    self.send_observation(observation)  # must_go=True 随观测发给 server

    if observation.must_go:
        self.must_go.clear()  # 用完就清，避免后续每帧都强制推理
```

```python
# ── server 端：入队时检查 must_go ──
def _enqueue_observation(self, obs):
    if (
        obs.must_go                    # ← 强制推理！跳过所有 sanity check
        or self.last_processed_obs is None
        or self._obs_sanity_checks(obs, self.last_processed_obs)
    ):
        if self.observation_queue.full():
            self.observation_queue.get_nowait()  # 先清掉旧的
        self.observation_queue.put(obs)
        return True
    return False
```

```python
# ── client 端：收到新 chunk 后重置 must_go ──
def receive_actions(self):
    ...
    self._aggregate_action_queues(timed_actions, ...)
    self.must_go.set()  # 重置，为下一次空队列做准备
```

如果 `must_go` 频繁被触发,说明推理速度跟不上执行速度——参数没调好或硬件瓶颈。

---

## 五、核心陷阱:chunk 的"有效部分"远小于想象

**这是 async inference 最不被讨论的一点**,也是决定它在你硬件上能不能生效的关键。

### 场景设定

- `chunk_size=100`
- `threshold=0.5`(队列剩 50 触发)
- 执行频率 `F=30 FPS`
- 推理延迟 `T_infer=2s`

### 详细时间线

```
时刻:  t=0       t=1.67s     t=3.33s     t=3.67s
frame:  0          50          100         110
        │          │           │           │
主线程: ├──执行 chunk A[0..99]──┤
                   ↑                        ↑
                  采 O_50                   chunk B 到货
                   │                        │
后台:              ├────2s 推理─────────────┤
                                            │
              chunk B 时间戳 = [50, 51, ..., 149]
              主线程此刻所在帧 = 110(推理期间走了 60 帧)
```

### Chunk B 的 100 帧命运分解

| chunk B 时间戳段 | 帧数 | 状态 | 命运 |
|---|---:|---|---|
| `[50, 109]` | 60 帧 | 主线程已走过 | **直接丢弃**(过期) |
| `[110, 149]` | 40 帧 | 纯未来 | 入队 |

**注意:这里 chunk A 和 chunk B 的时间戳没有重叠**——因为 chunk A 只覆盖 `[0, 99]`,
chunk B 从 `50` 开始但主线程到货时已在 `110`。Temporal ensemble **一次都没触发**,
chunk B 那 60 帧预测完全被丢掉。

---

### 三种推理速度下的完整时间线

> **统一参数**：`N=100, threshold=0.5(→触发阈值=50帧), F=30 FPS`
>
> 图示：每格=10帧 | `████` 正常执行 | `░░░░` stall(队列空,机器人停住) | 推理在后台并行

---

#### 情况 A：T_infer = 30 帧(1 s) ✅ 永不 stall，稳态 30 FPS

```
帧:    0    10   20   30   40   50   60   70   80   90  100  110  120  130  140  150
       │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
执行:  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ──→
推理:                            ↑P1 ─────────────↓           ↑P2 ─────────────↓
                                 采O_50 (30帧)   到货f80       采O_100 (30帧)  到货f130
队列:  100   90   80   70   60  [50]  40   30  [70]  60  [50]  40   30  [70]  60  [50]
                               触发          合并+70  触发          合并+70
```

- 到货时队列剩余：50 − 30 = **+20 帧**（有余量）
- Chunk B 时间戳 `[50..149]`，到货时 frame=80：
  - `[50..79]` 30 帧过期 → 丢弃
  - `[80..99]` 20 帧与老队列重叠 → **temporal ensemble 有效**
  - `[100..149]` 50 帧纯新增 → 入队
- 合并后 70 帧，走 20 帧触发下轮，如此循环。

---

#### 情况 B：T_infer = 50 帧(1.67 s) ⚠️ 临界，无 ensemble，偶发抖动

```
帧:    0    10   20   30   40   50   60   70   80   90  100  110  120  130  140  150
       │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
执行:  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ──→
推理:                            ↑P1 ─────────────────────────────↓
                                 采O_50        (50帧)            到货f100
                                                                  ↑P2(立刻触发，因为50帧=阈值)
队列:  100   90   80   70   60  [50]  40   30   20   10  [50]  40   30   20   10  [50]
                               触发                     刚好空   立触P2              刚好空
```

- 到货时队列剩余：50 − 50 = **0 帧**（临界，刚好耗尽）
- Chunk B 时间戳 `[50..149]`，到货时 frame=100：
  - `[50..99]` 50 帧过期 → 丢弃
  - `[100..149]` 50 帧纯新增 → 入队，且 50=阈值，**立刻触发 P2**
- **temporal ensemble 完全失效**：老队列恰好空，无任何重叠帧。
- 实际中单帧时序抖动可能导致偶发 1 帧 stall（紧急靠 `must_go` 兜底）。

---

#### 情况 C：T_infer = 70 帧(2.33 s) ⚠️ 周期性 stall，实际 ≈ 21.4 FPS

> **关键机制**：stall 期间机器人停止执行动作，`latest_action` 冻结，
> chunk 的"过期"判断基于 `latest_action`（见
> [robot_client.py:414](../../lerobot/src/lerobot/scripts/server/robot_client.py#L414):
> `timestep=max(latest_action, 0)`），**因此 stall 的帧不计入过期**。

```
帧:    0    10   20   30   40   50   60   70   80   90  100  110  120  130  140  150  160  170  180  190
       │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
执行:  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ░░░░ ░░░░ ████ ████ ████ ████ ████ ░░░░ ░░░░ ████→
       ←── chunk A 全部 100 帧 ───→  └stall┘  ←──── chunk B 有效 50 帧 ─────→   └stall┘
                                      20帧                                          20帧
推理:                            ↑P1 ─────────────────────────────────────────────────↓
                                 采O_50                        (70帧)                到货f120
                                                                                     ↑P2(立刻触发)
执行序号(latest_action):
       0    10   20   30   40   50   60   70   80   90   99   99  100  110  120  130  140  149  149  150
                               触发时=50                   ↑空!  冻结!  到货  执行chunk B         ↑空!  冻结!
```

- P1 触发时 `latest_action=50`，采集 O_50，chunk B 时间戳 `[50..149]`
- P1 到货时（frame 120 实时）`latest_action=99`（chunk A 最后一帧，stall 期冻结）
- 过期判断：`timestep <= latest_action=99`，即 `[50..99]` = **50 帧过期**（只有执行期）
- **stall 的 20 帧不过期**，`[100..149]` = **50 帧有效**（不是 30 帧！）
- 50 帧入队 = 阈值，立触 P2

**稳态（每轮固定循环，从 chunk B 起）：**

```
           ←──────────── 推理 P2(70帧) ────────────→
 执行 50 帧                    stall 20 帧               执行 50 帧                    stall 20 帧
 ████████████████████████       ░░░░░░░░               ████████████████████████       ░░░░░░░░
 │                     │        │      │                │                     │        │      │
f120                  f170     f170   f190             f190                  f240     f240   f260
 (到货 50 帧,立触 P2)  (队列空) (等推理) (到货 50 帧)
 ├────────────────────────── 70 帧 / 周期 ──────────────────────────────────────────────────────→
```

- stall 的 20 帧中 `latest_action` 冻结 → 到货 50 帧过期 0，有效 50 帧 → 立触 → 循环
- 实际 FPS = 50 帧执行 ÷ (70 帧周期 / 30 FPS) = **≈ 21.4 FPS**

---

#### 三种情况汇总

| 情况 | T_infer | stall 时长 | 有效帧(stall不过期) | Temporal Ensemble | 实际 FPS |
|---|---:|---:|---:|---|---:|
| A | 30 帧(1 s) | 无 | 50+20=70 帧 | ✅ 20 帧重叠 | **30** |
| B | 50 帧(1.67 s) | 偶发 1 帧 | 50 帧 | ❌ 无 | **≈30** |
| C | 70 帧(2.33 s) | 20 帧/周期 | 50 帧 | ❌ 无 | **≈21.4** |

不等式 `T_infer ≤ N × threshold / F`（即 `N ≥ T_infer × F / threshold`）的验证：

不 stall 条件：推理时间 ≤ 触发阈值队列的执行时间 = (N × threshold) / F = (100 × 0.5) / 30 = **1.67 s**

| 情况 | T_infer | 对比阈值执行时间 | 结果 |
|---|---:|---:|---|
| A | 1 s | 1 < 1.67 s | ✅ 不 stall，有余量 |
| B | 1.67 s | 1.67 = 1.67 s | ⚠️ 临界，偶发单帧 |
| C | 2.33 s | 2.33 > 1.67 s | ❌ stall 20 帧/周期 |

---

## 六、可行性不等式

把上面的观察一般化,写成一个简洁的条件:

> **稳态条件**:一次推理产出的新增有效帧数 ≥ 推理期间消耗的帧数
>
> `N - T_infer × F ≥ T_infer × F`
>
> → **`N ≥ 2 × T_infer × F`**

符号:
- `N` = chunk_size(训练期固定,运行时不能改)
- `T_infer` = 一次推理耗时(秒)
- `F` = 执行频率(FPS)

### 直观解释

- `T_infer × F` = 推理一次期间主线程"往前走"了多少帧
- 新 chunk 的前 `T_infer × F` 帧会因为"过期"被丢掉
- 剩下的 `N - T_infer × F` 帧必须足够让主线程"撑"到下一次推理到货
- 下一次推理到货又要 `T_infer`,期间又要消耗 `T_infer × F` 帧
- 所以要求:`N - T_infer × F ≥ T_infer × F`

如果这个不等式不成立,async inference 会周期性地 stall,实际 FPS 会退化到
 `F_actual ≈ (N - T_infer × F) / T_infer`(每次推理只能支撑这么多帧的执行)。

### 代入本项目数据

假设 `T_infer=2s`、`F=30`(来自
[timing_stats.csv](../../lerobot/analysis/timing_stats.csv)):

| chunk_size N | `2 × T_infer × F` | 满足? | 稳态 FPS(理论) |
|---:|---:|---|---:|
| 50  | 120 | 否 | ~(50-60)/2 = **跌到负数,必 stall** |
| 100 | 120 | 否 | ~(100-60)/2 = **20 FPS** |
| 120 | 120 | 勉强 | ~30 FPS(无余量,容易抖) |
| 150 | 120 | 是 | 30 FPS(有余量) |
| 200 | 120 | 是 | 30 FPS(很稳) |

**结论**:cs=100 + 2s 推理在 30 FPS 下**不可行**——理论吞吐只有 20 FPS,比同步模式
(~19 FPS)**几乎没提升**,反而多了并发和 CPU 争抢的成本。

---

## 七、什么时候 async inference 才真的赚?

不等式 `N ≥ 2 × T_infer × F` 给出了三个旋钮,任意一个调到位都能让 async 生效:

### 方案 A:把推理变快(推荐)

让 `T_infer` 下降,是不等式左侧最有效的调整:

- ACT int8 量化 → T_infer 可能从 2s 降到 ~800ms
- ONNX/NCNN 部署 + ARM NEON → 再砍一半
- 目标:`T_infer ≤ N / (2F) = 100 / 60 ≈ 1.67s`

如果能做到 `T_infer=1s`,那么 `2 × 1 × 30 = 60 < 100`,cs=100 直接可用。

### 方案 B:加大 chunk_size(需要重训)

ACT 的 `chunk_size` 是**训练期固定**的(模型结构里写死了 decoder 的序列长度),
运行时改不了。如果要走这条路,必须:

1. 重新训一个 cs=200 的模型
2. 验证 cs 变大后 policy 质量不降(过长的 open-loop 可能导致误差累积)

代价较大。

### 方案 C:降低执行频率

把 `F` 从 30 FPS 降到 15 FPS:`2 × 2 × 15 = 60 < 100`,cs=100 够用。
但机器人反应会变慢,可能影响任务成功率。

### 方案 D:放弃 async(最保守)

在 cs=100 + 2s 推理的当前条件下,同步模式的 ~19 FPS 可能就是现实上限。
把精力放在推理优化(方案 A)上,推理变快之后无论同步异步都受益。

---

## 八、树莓派 5 单机的额外考量

假设不等式满足了,async inference 还有一层**CPU 争抢**问题:

树莓派 5 只有 4 个 ARM Cortex-A76 核心。ACT 推理靠 PyTorch 的 OpenMP
(见 [PyTorch_OpenMP_调用链详解.md](./PyTorch_OpenMP_调用链详解.md))
默认会吃满所有核心。如果 `policy_server` 和 `robot_client` 同机运行:

| 进程 | CPU 需求 |
|---|---|
| policy_server | 推理时 4 核全占(OpenMP 默认) |
| robot_client | 控制循环 + 串口 IO + 相机读取,~1 核 |

两者重叠时会互相挤,推理线程被迫让出时间片 → `T_infer` 变大 → 不等式更难满足。

**缓解措施**:

1. **绑核**:`taskset -c 0-2 python policy_server.py` 让推理只用 3 核,
   `taskset -c 3 python robot_client.py` 把控制循环钉在剩下那核
2. **限制 torch 线程**:policy_server 里设 `torch.set_num_threads(3)`
3. **禁用 OpenCV 的内部多线程**:`cv2.setNumThreads(1)` 避免相机读取跟推理抢核

即便如此,3 核推理比 4 核慢 ~25%,`T_infer` 会相应增加,不等式的余量被进一步吃掉。

---

## 九、对本项目的实际建议

按性价比排序:

1. **先把推理变快**(方案 A):ACT int8/ONNX 移植是一次性投入,同步异步都受益。
   这是本项目最大的优化空间——20 FPS → 理论可达 100 FPS+。

2. **推理变快之后再考虑 async**:如果 `T_infer` 能降到 1s 以下,cs=100 直接
   满足不等式,此时 async 能稳定带来 ~60% 吞吐提升,值得投入。

3. **如果短期必须用 async**:考虑**训一个 cs=200 的 ACT 模型**配 async,
   这条路可行但工作量大(重新训、重新评估、重新调 temporal ensemble 权重)。

4. **现阶段 cs=100 不建议上 async**:实测吞吐大概率不如同步,反而引入
   双进程、gRPC、队列同步、CPU 争抢等一堆复杂度,ROI 为负。

5. **真正的底线**:先跑一遍官方 demo 实测,用和
   [plot_timing.py](../../lerobot/analysis/plot_timing.py) 同样的指标对比 cs=100
   下 async vs 同步的 FPS。如果不等式分析是对的,async 应该不如同步——这能
   作为反向验证,省得在错的路径上投入优化工作。

---

## 十、参考

- lerobot 官方实现:
  - [robot_client.py](../../lerobot/src/lerobot/scripts/server/robot_client.py)
  - [policy_server.py](../../lerobot/src/lerobot/scripts/server/policy_server.py)
  - [helpers.py](../../lerobot/src/lerobot/scripts/server/helpers.py)(内含 aggregate_fn、TimedAction 等)
- 性能数据:[timing_stats.csv](../../lerobot/analysis/timing_stats.csv)、
  [chart3_ms_per_frame.png](../../lerobot/analysis/chart3_ms_per_frame.png)
- 相关分析:
  - [motors_bus_异步读写设计分析.md](./motors_bus_异步读写设计分析.md)(为什么 bus 层不能做异步写)
  - [ACT模型推理过程分析.md](./ACT模型推理过程分析.md)(chunk_size 和 n_action_steps 的关系)
  - [predict_action时间分析.md](./predict_action时间分析.md)(推理耗时分解)
  - [PyTorch_OpenMP_调用链详解.md](./PyTorch_OpenMP_调用链详解.md)(CPU 核心占用分析)

---

## 附录:一张图总结

```
           ┌────────────────────────────────────────┐
           │  想要 async inference 真正带来吞吐提升  │
           └──────────────────┬─────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   N ≥ 2 × T_infer × F         │
              └───────┬─────────┬─────────┬───┘
                      │         │         │
             变大 N   │  变小 T │  变小 F │
                      ▼         ▼         ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ 重训 cs  │ │ 量化/ONNX│ │ 降 FPS   │
              │ 更大模型 │ │ 减小模型 │ │          │
              │ 工作量大 │ │ 一次投入 │ │ 反应变慢 │
              │ 质量风险 │ │ 同步异步 │ │ 任务受影响│
              │          │ │ 都受益   │ │          │
              └──────────┘ └──────────┘ └──────────┘
                              ▲
                              │
                              └── 推荐优先级最高
```

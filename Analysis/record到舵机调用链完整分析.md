# record.py → scservo_sdk 完整调用链分析

本文档追踪从 `python -m lerobot.record` 指令到舵机底层 SDK 的每一层调用，说明每个函数的参数含义和作用。
**分析命令**:
```bash
python -m lerobot.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12254705 \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=R07254705 \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}" \
    --dataset.single_task="Put the bottle into the black basket." \
    --policy.path=/home/vel/so101-bottle/last/pretrained_model \
    --dataset.repo_id=${HF_USER}/eval_so101_bottle --dataset.push_to_hub=false \
```

---

## 总览：调用链示意图

```
python -m lerobot.record
    ↓
record(cfg: RecordConfig)                    [lerobot/src/lerobot/record.py]
    ↓
make_robot_from_config(cfg.robot)            [lerobot/src/lerobot/robots/utils.py]
    ↓
SO101Follower(config)                        [lerobot/src/lerobot/robots/so101_follower/so101_follower.py]
    │ 内部持有 FeetechMotorsBus (负责串口通信)
    │ 内部持有 cameras (dict: 相机实例)
    ↓
robot.connect()
    ├─ bus.connect() → port_handler.openPort() → 打开串口
    ├─ bus.calibrate() → 校准(可选)
    └─ cam.connect() → 连接相机

[每帧循环 ~33ms @ 30fps]
    ↓
┌─ 观测阶段 ─────────────────────────────────────────────────────────────
│  robot.get_observation()
│      ├─ bus.sync_read("Present_Position") → 串口读 → 各舵机返回位置
│      └─ cam.async_read() → 从后台线程缓存获取最新图像
│          ↓
│  robot_observation_processor(obs) → 观测后处理
│          ↓
│  build_dataset_frame(features, obs_processed, prefix="observation")
│          ↓
│  observation_frame (用于策略推理或写入数据集)
└────────────────────────────────────────────────────────────────────────

┌─ 推理阶段 (二选一) ──────────────────────────────────────────────────────
│
│  [A] POLICY 路径 (有策略时):
│      predict_action(observation_frame, policy, ...)
│          ├─ preprocessor(observation) → 输入预处理
│          ├─ policy.select_action(observation) → 模型推理
│          └─ postprocessor(action) → 输出后处理
│          ↓
│      act_processed_policy: {"shoulder_pan.pos": 0.7, ...}
│
│  [B] TELEOP 路径 (有示教手时):
│      teleop.get_action()
│          └─ bus.sync_read("Present_Position") → 读示教手当前角度
│          ↓
│      teleop_action_processor((act, obs)) → 示教动作处理
│          ↓
│      act_processed_teleop: {"shoulder_pan.pos": 0.5, ...}
│
└────────────────────────────────────────────────────────────────────────

┌─ 动作阶段 ─────────────────────────────────────────────────────────────
│  robot_action_processor((action, obs)) → 动作处理
│          ↓
│  robot.send_action(robot_action_to_send)
│      └─ bus.sync_write("Goal_Position", goal_pos)
│          ├─ _unnormalize() → 物理值 → 原始字节
│          ├─ _encode_sign() → 符号位处理
│          └─ sync_writer.txPacket() → 串口写 → 舵机收到目标位置
│          ↓
│  dataset.add_frame(frame) → 写入数据集
│          ↓
│  log_rerun_data(observation, action) → 可视化(可选)
└────────────────────────────────────────────────────────────────────────

┌─ 等待阶段 ─────────────────────────────────────────────────────────────
│  busy_wait(1/fps - dt_s) → 等待以维持目标帧率
└────────────────────────────────────────────────────────────────────────
```

---

## 第 1 层：record.py — 记录主入口

**文件：** `lerobot/src/lerobot/record.py`

### `RecordConfig` 完整定义

```python
# record.py:199-229
@dataclass
class RecordConfig:
    robot: RobotConfig                                      # 机器人硬件配置（串口、舵机型号等）
    dataset: DatasetRecordConfig                            # 数据集配置（repo_id、fps、episode数量等）
    teleop: TeleoperatorConfig | None = None                # 示教设备配置（示教手类型、串口等）
    policy: PreTrainedConfig | None = None                  # 策略配置（模型路径、设备等）
    display_data: bool = False                              # 是否开启rerun可视化
    play_sounds: bool = True                                # 是否使用语音播报事件
    resume: bool = False                                    # 是否从已有数据集继续录制

    def __post_init__(self):
        # HACK: 再次解析命令行以获取预训练模型路径
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """使解析器支持 --policy.path=local/dir 从本地加载配置"""
        return ["policy"]
```

### `DatasetRecordConfig` 完整定义

```python
# record.py:156-197
@dataclass
class DatasetRecordConfig:
    repo_id: str                                            # 数据集仓库ID，如 "lerobot/test"
    single_task: str                                        # 任务描述，如 "Put the bottle into the black basket."
    root: str | Path | None = None                          # 数据集存储根目录（默认 ~/.cache/huggingface/lerobot）
    fps: int = 30                                           # 目标帧率
    episode_time_s: int | float = 60                        # 每个episode录制时长（秒）
    reset_time_s: int | float = 60                          # 重置环境等待时长（秒）
    num_episodes: int = 50                                  # 总录制episode数
    video: bool = True                                      # 是否将帧编码为视频
    push_to_hub: bool = True                                # 是否上传到HuggingFace Hub
    private: bool = False                                   # 是否私有仓库
    tags: list[str] | None = None                           # Hub标签
    num_image_writer_processes: int = 0                     # 图片写子进程数（0=只用线程）
    num_image_writer_threads_per_camera: int = 4            # 每相机图片写线程数
    video_encoding_batch_size: int = 1                      # 视频编码批大小
    rename_map: dict[str, str] = field(default_factory=dict)  # 观测/动作重命名映射

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")
```

### 本命令下各配置的实际值

**`cfg.robot` (SO101FollowerConfig)：**
```python
SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="R12254705",
    disable_torque_on_disconnect=True,
    max_relative_target=None,
    cameras={
        "handeye": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=30),
        "fixed":   OpenCVCameraConfig(index_or_path=2, width=640, height=360, fps=30),
    },
    use_degrees=False,
    calibration_dir=None,
)
```

**`cfg.teleop` (SO101LeaderConfig)：**
```python
SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="R07254705",
    use_degrees=False,
    calibration_dir=None,
)
```

**`cfg.dataset` (DatasetRecordConfig)：**
```python
DatasetRecordConfig(
    repo_id="${HF_USER}/eval_so101_bottle",
    single_task="Put the bottle into the black basket.",
    root=None, fps=30, episode_time_s=60, reset_time_s=60,
    num_episodes=50, video=True, push_to_hub=False,
    num_image_writer_processes=0, num_image_writer_threads_per_camera=4,
    video_encoding_batch_size=1, rename_map={},
)
```

**`cfg.policy` (PreTrainedConfig)：** 从 `/home/vel/so101-bottle/last/pretrained_model` 加载的模型配置。

### 入口函数 `record()`

```python
# record.py:500
@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
```

`cfg` 即上面描述的完整配置对象。

**关键执行步骤：**

```python
# record.py:509
robot = make_robot_from_config(cfg.robot)   # 实例化机器人(SO101Follower)

# record.py:510
teleop = make_teleoperator_from_config(cfg.teleop)  # 实例化示教手(SO101Leader)

# record.py:512
# 创建三大处理器: 示教动作处理、机器人动作处理、观测处理
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# record.py:572
robot.connect()   # 打开串口、连接相机、写入配置

# record.py:573-574
if teleop is not None:
    teleop.connect()  # 连接示教手

# record.py:590-643
while recorded_episodes < cfg.dataset.num_episodes:
    record_loop(robot, ...)      # 核心录制循环,运行 episode_time_s 秒
    record_loop(..., reset_time_s)  # 重置循环(可选)
    dataset.save_episode()       # 保存episode到磁盘

# record.py:647-649
robot.disconnect()
teleop.disconnect()
```

---

### 核心循环 `record_loop()`

```python
# lerobot/src/lerobot/record.py:263
@safe_stop_image_writer  # 装饰器: 确保图像写入器正确停止（即使异常也会调用 dataset.stop_image_writer()）
def record_loop(
    robot: Robot,                        # SO101Follower实例 — 被控制的从臂机器人
    events: dict,                        # 键盘事件字典，结构如下
    fps: int,                            # 目标帧率(Hz)，本命令为 30
    teleop_action_processor: RobotProcessorPipeline,  # 示教动作后处理管道（默认为恒等变换）
    robot_action_processor: RobotProcessorPipeline,    # 机器人动作后处理管道（默认为恒等变换）
    robot_observation_processor: RobotProcessorPipeline,  # 观测后处理管道（默认为恒等变换）
    dataset: LeRobotDataset | None = None,  # 数据集实例；None 时不记录数据（重置阶段）
    teleop: Teleoperator | list[Teleoperator] | None = None,  # 遥控设备；单个/多个/None
    policy: PreTrainedPolicy | None = None,  # 预训练策略模型；None 时不推理（重置阶段）
    preprocessor: PolicyProcessorPipeline | None = None,  # 策略输入预处理器（归一化、设备转移等）
    postprocessor: PolicyProcessorPipeline | None = None,  # 策略输出后处理器（反归一化等）
    control_time_s: int | None = None,   # 本轮循环的总控制时长（秒）；录制=60, 重置=60
    single_task: str | None = None,      # 任务描述字符串，如 "Put the bottle into the black basket."
    display_data: bool = False,          # 是否用 rerun 可视化观测和动作
) -> None:                               # 无返回值；数据通过 dataset.add_frame() 副作用写入
```

**`events` 字典结构**（由 `init_keyboard_listener()` 创建）：

| 键 | 类型 | 含义 | 触发方式 |
|----|------|------|----------|
| `"exit_early"` | `bool` | 提前结束当前 episode | 按 → 键 |
| `"rerecord_episode"` | `bool` | 重录当前 episode | 按 ← 键 |
| `"stop_recording"` | `bool` | 停止全部录制 | 按 Esc 键 |

**三个 Processor Pipeline 参数详解**（由 `make_default_processors()` 创建，默认都是恒等变换）：

| 参数 | 泛型签名 | 输入 | 输出 | 作用 |
|------|----------|------|------|------|
| `teleop_action_processor` | `RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]` | `(动作dict, 观测dict)` 二元组 | 动作 `dict` | 对 teleop 原始动作做处理（默认不修改） |
| `robot_action_processor` | `RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]` | `(动作dict, 观测dict)` 二元组 | 动作 `dict` | 对下发给机器人的动作做处理（默认不修改） |
| `robot_observation_processor` | `RobotProcessorPipeline[RobotObservation, RobotObservation]` | 观测 `dict` | 观测 `dict` | 对观测做处理（默认不修改） |

> 其中 `RobotAction = dict[str, Any]`，`RobotObservation = dict[str, Any]`（定义在 `processor/core.py:L40-L42`）

**循环内每帧执行流程（每帧预算 33.33ms @ 30fps）：**

#### 1. 观测阶段 (Observation)

#### 1. 观测阶段 (Observation)

```python
# lerobot/src/lerobot/record.py:339
obs = robot.get_observation()
# ── 返回值 obs: RobotObservation (dict[str, Any]) ──
# {
#     "shoulder_pan.pos": 0.5,     # float，关节1水平旋转角度（归一化值，范围-100~100）
#     "shoulder_lift.pos": -0.3,   # float，关节2肩部抬起角度
#     "elbow_flex.pos": 0.1,       # float，关节3肘部弯曲角度
#     "wrist_flex.pos": 0.2,       # float，关节4腕部弯曲角度
#     "wrist_roll.pos": 0.0,       # float，关节5腕部旋转角度
#     "gripper.pos": 50.0,         # float，夹爪开度（范围0~100）
#     "handeye": np.ndarray,       # shape=(360, 640, 3)，手眼相机最新帧（BGR uint8）
#     "fixed": np.ndarray,         # shape=(360, 640, 3)，固定相机最新帧（BGR uint8）
# }

obs_processed = robot_observation_processor(obs)
# ── 输入: obs (dict) ── 上面 robot.get_observation() 的返回值
# ── 输出: obs_processed (dict) ── 默认与 obs 相同（恒等变换）
# ── 作用: 预留扩展点，未来可加归一化、滤波等

observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
# ── 参数:
#   dataset.features: dict — 数据集特征定义，描述每个字段的 dtype/shape/names
#   obs_processed: dict — 经过处理的观测数据
#   prefix: str — 键前缀，"observation" → 输出键如 "observation.state", "observation.images.handeye"
# ── 返回值 observation_frame: dict，例如:
# {
#     "observation.state": np.array([0.5, -0.3, 0.1, 0.2, 0.0, 50.0]),  # shape=(6,)
#     "observation.images.handeye": np.ndarray,  # shape=(3, 360, 640)，CHW float32
#     "observation.images.fixed": np.ndarray,    # shape=(3, 360, 640)，CHW float32
# }
```

#### 2. 推理阶段 (Inference) - 二选一

**路径A: POLICY 策略推理**
```python
# lerobot/src/lerobot/record.py:368
action_values = predict_action(
    observation=observation_frame,       # dict，上面构建的观测帧（含图像和关节状态）
    policy=policy,                       # PreTrainedPolicy 实例，如 ACTPolicy
    device=get_safe_torch_device(policy.config.device),  # torch.device，树莓派上为 CPU
    preprocessor=preprocessor,           # PolicyProcessorPipeline，做输入归一化、tensor 转换
    postprocessor=postprocessor,         # PolicyProcessorPipeline，做输出反归一化
    use_amp=policy.config.use_amp,       # bool，是否用混合精度（CPU 上为 False）
    task=single_task,                    # str，任务描述，如 "Put the bottle into the black basket."
    robot_type=robot.robot_type,         # str，"so101_follower"
)
# ── 返回值 action_values: torch.Tensor ──
#   shape=(num_actions,)，如 (6,)，值域为归一化后的动作值

action_names = dataset.features["action"]["names"]
# ── action_names: list[str]，如 ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

act_processed_policy = {f"{name}": float(action_values[i]) for i, name in enumerate(action_names)}
# ── act_processed_policy: RobotAction (dict[str, float])，如:
# {"shoulder_pan": 0.7, "shoulder_lift": -0.2, "elbow_flex": 0.3, "wrist_flex": 0.1, "wrist_roll": 0.0, "gripper": 60.0}
```

**路径B: TELEOP 示教读取**
```python
# lerobot/src/lerobot/record.py:388
act = teleop.get_action()
# ── 返回值 act: RobotAction (dict[str, float]) ──
#   从 SO101Leader 主臂读取当前关节角度，格式同 obs 的关节部分:
# {"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 0.1, ...}

act_processed_teleop = teleop_action_processor((act, obs))
# ── 输入: (act, obs) 二元组 — teleop 原始动作 + 当前观测
# ── 输出: act_processed_teleop (dict[str, float]) — 默认直接返回 act（恒等变换）
```

#### 3. 动作阶段 (Action)

```python
# lerobot/src/lerobot/record.py:426-448
# Step 1: 选择动作来源
if policy is not None:
    action_values = act_processed_policy          # dict[str, float]，策略输出的动作
    robot_action_to_send = robot_action_processor((act_processed_policy, obs))
else:
    action_values = act_processed_teleop          # dict[str, float]，teleop 输出的动作
    robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
# ── robot_action_to_send: RobotAction (dict[str, float]) — 最终下发给机器人的动作
# ── 默认与 action_values 相同（恒等变换）

# Step 2: 发送给机器人
_sent_action = robot.send_action(robot_action_to_send)
# ── 参数: robot_action_to_send (dict[str, float])，如 {"shoulder_pan.pos": 0.7, ...}
# ── 返回值 _sent_action: dict[str, float] — 实际写入舵机的值（可能经安全限幅后不同）
# ── 底层调用: bus.sync_write("Goal_Position", goal_pos)

# Step 3: 写入数据集
if dataset is not None:
    action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")
    # ── action_frame: dict，如 {"action": np.array([0.7, -0.2, 0.3, 0.1, 0.0, 60.0])}

    frame = {**observation_frame, **action_frame, "task": single_task}
    # ── frame: dict，合并了观测、动作和任务描述，如:
    # {
    #     "observation.state": np.array([...]),
    #     "observation.images.handeye": np.ndarray,
    #     "observation.images.fixed": np.ndarray,
    #     "action": np.array([...]),
    #     "task": "Put the bottle into the black basket.",
    # }

    dataset.add_frame(frame)
    # ── 将 frame 追加到当前 episode 的内存缓冲区，不立即写盘
```

#### 4. 等待阶段 (Wait)

```python
# lerobot/src/lerobot/record.py:465-466
dt_s = time.perf_counter() - start_loop_t
# ── dt_s: float — 本帧已消耗的时间（秒），包含观测+推理+动作+写数据集的总耗时

busy_wait(1 / fps - dt_s)
# ── 参数: 1/30 - dt_s = 0.0333 - dt_s
# ── 如果 dt_s < 33.33ms，忙等待补齐剩余时间以维持 30fps
# ── 如果 dt_s > 33.33ms，参数为负值，busy_wait 内部直接跳过（帧率下降）
```

---

## 第 2 层：robots/utils.py — 工厂函数

**文件：** `lerobot/src/lerobot/robots/utils.py`

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    """
    根据配置创建机器人实例
    
    Args:
        config (RobotConfig): 机器人配置对象（本命令为 SO101FollowerConfig）
            - port: str — 串口设备路径，本命令为 "/dev/ttyACM0"
            - id: str — 机器人标识，本命令为 "R12254705"
            - motors: dict[str, Motor] — 舵机定义
            - cameras: dict[str, CameraConfig] — 相机配置
            - calibration_dir: Path | None — 标定文件目录
            - disable_torque_on_disconnect: bool — 断开时是否释放力矩
            - max_relative_target: float | dict | None — 安全限幅阈值
            - use_degrees: bool — 是否使用角度单位（本命令 False）
            - calibration: dict[str, MotorCalibration] — 标定数据
        
    Returns:
        Robot: 对应的机器人实例，本命令返回 SO101Follower
        
    路由逻辑：根据 config.type 字符串路由：
        - "so100_follower" → SO100Follower
        - "so101_follower" → SO101Follower ← **本命令走这里**
        - "bi_so100_follower" → BiSO100Follower
        - "bi_so101_follower" → BiSO101Follower
        - "koch_follower" → KochFollower
        - "lekiwi" → LeKiwi
    """
```

*（注：由于函数内容已经很简单，直接把参数说明和路由逻辑直接写在函数说明里）*

---

## 第 3 层：SO101Follower — 机器人实现

**文件：** `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`

---

## 第 3 层：SO101Follower — 机器人实现

**文件：** `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`

### 初始化 `__init__()`

```python
# so101_follower.py:45
def __init__(self, config: SO101FollowerConfig):
    """
    初始化 SO101 从臂机器人
    
    Args:
        config (SO101FollowerConfig): 机器人配置
            - port: str — 串口设备路径，如 "/dev/ttyACM0"
            - id: str — 机器人标识，如 "R12254705"
            - disable_torque_on_disconnect: bool — 断开时是否释放力矩，本命令 True
            - max_relative_target: float | dict | None — 安全限幅阈值，本命令 None
            - cameras: dict[str, CameraConfig] — 相机配置，本命令有 handeye 和 fixed
            - use_degrees: bool — 是否使用角度单位，本命令 False
            - calibration_dir: Path | None — 标定文件目录，本命令 None
    
    内部创建的关键对象：
        - self.bus: FeetechMotorsBus — 舵机通信总线
        - self.cameras: dict[str, OpenCVCamera] — 相机实例映射
    """
    super().__init__(config)
    self.config = config

    # 归一化模式: 身体关节用-100~100度,夹爪用0~100
    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    # ── norm_mode_body: MotorNormMode 枚举值，决定归一化范围
    #    RANGE_M100_100: 将舵机原始值 0~4095 映射到 -100.0~100.0
    #    DEGREES: 映射到角度值（本命令 use_degrees=False，不走此分支）

    # 创建Feetech舵机总线
    self.bus = FeetechMotorsBus(
        port=self.config.port,          # str，串口设备路径，如 "/dev/ttyACM0"
        motors={                        # dict[str, Motor]，舵机名称 → Motor 描述对象
            "shoulder_pan":  Motor(1, "sts3215", norm_mode_body),   # ID=1, 型号sts3215, 范围-100~100
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),   # ID=2, 肩部抬起
            "elbow_flex":    Motor(3, "sts3215", norm_mode_body),   # ID=3, 肘部弯曲
            "wrist_flex":    Motor(4, "sts3215", norm_mode_body),   # ID=4, 腕部弯曲
            "wrist_roll":    Motor(5, "sts3215", norm_mode_body),   # ID=5, 腕部旋转
            "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # ID=6, 夹爪范围0~100
        },
        calibration=self.calibration,   # dict[str, MotorCalibration] — 从 JSON 文件加载的标定数据（offset、drive_mode 等）
    )

    # 创建相机实例
    self.cameras = make_cameras_from_configs(config.cameras)
    # ── config.cameras: dict[str, OpenCVCameraConfig]
    #    返回: dict[str, OpenCVCamera]，如 {"handeye": OpenCVCamera实例, "fixed": OpenCVCamera实例}

**`Motor` 数据结构详解：**

```python
@dataclass
class Motor:
    id: int                # 舵机总线地址，范围 0~253（如 1, 2, ..., 6）
    model: str             # 舵机型号字符串，如 "sts3215"（用于查控制表确定寄存器地址）
    norm_mode: MotorNormMode  # 归一化模式枚举：
                              #   RANGE_M100_100 → 归一化到 -100.0~100.0
                              #   RANGE_0_100    → 归一化到 0.0~100.0
                              #   DEGREES        → 归一化到角度值
```

### `connect()` 建立连接

```python
# so101_follower.py:85
def connect(self, calibrate: bool = True) -> None:
    """
    建立机器人连接（打开串口、校准、连接相机、写入配置）
    
    Args:
        calibrate (bool): 是否在连接后自动校准（如果没有已有标定文件）
            - 如果校准失败，会报 CalibrationError
            - 如果校准成功，保存标定数据到 calibration_fpath
    
    Returns:
        None
    """
    if self.is_connected:
        raise DeviceAlreadyConnectedError(...)

    # Step 1: 连接舵机总线(打开串口)
    self.bus.connect()
    # → 内部调用 PortHandler.openPort()，打开 /dev/ttyACM0
    # → 设置波特率为 1000000 (1Mbps)

    # Step 2: 校准(如需要)
    if not self.is_calibrated and calibrate:
        if not self.calibration:
            logger.info("No calibration file found, running calibration")
        self.calibrate()  # 运行交互式校准流程，生成 offset 和 drive_mode

    # Step 3: 连接所有相机
    for cam in self.cameras.values():
        cam.connect()
        # → 启动后台采集线程
        # → 预热 warmup_s 秒（默认 1 秒）

    # Step 4: 写入运行时配置
    self.configure()
    # → 向每个舵机写入 Torque_Enable=1, Operating_Mode 等寄存器
    logger.info(f"{self} connected.")
```

### `get_observation()` 读取状态

```python
# so101_follower.py:249
def get_observation(self) -> dict[str, Any]:
    """
    获取机器人当前状态观测
    
    Returns:
        dict[str, Any]: 观测字典，包含:
            - "shoulder_pan.pos": float — 关节1水平旋转角度（归一化值，范围-100~100）
            - "shoulder_lift.pos": float — 关节2肩部抬起角度
            - "elbow_flex.pos": float — 关节3肘部弯曲角度  
            - "wrist_flex.pos": float — 关节4腕部弯曲角度
            - "wrist_roll.pos": float — 关节5腕部旋转角度
            - "gripper.pos": float — 夹爪开度（范围0~100）
            - "handeye": np.ndarray — 手眼相机最新帧（H,W,3, BGR uint8）
            - "fixed": np.ndarray — 固定相机最新帧（H,W,3, BGR uint8）
    
    Raises:
        DeviceNotConnectedError: 如果机器人未连接
    
    底层执行流程:
        1. bus.sync_read("Present_Position") → 通过串口发送 Sync Read 指令包
           同步读取 6 个舵机的 Present_Position 寄存器（地址 56，2字节）
        2. cam.async_read() → 从后台线程缓存获取最近相机帧
           每个相机都有独立的后台线程持续采集帧（防止阻塞主循环）
    
    实例返回值（示例）:
        {
            "shoulder_pan.pos": 0.5,    # 当前水平旋转角度
            "shoulder_lift.pos": -0.3,  # 当前肩部抬起角度  
            "elbow_flex.pos": 0.1,      # 当前肘部弯曲角度
            "wrist_flex.pos": 0.2,      # 当前腕部弯曲角度
            "wrist_roll.pos": 0.0,      # 当前腕部旋转角度
            "gripper.pos": 50.0,        # 当前夹爪开度
            "handeye": np.ndarray,      # shape=(360, 640, 3)
            "fixed": np.ndarray,        # shape=(360, 640, 3) 
        }
    """
    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # Step 1: 读取机械臂关节位置
    # 通过电机总线同步读取各舵机的 Present_Position 寄存器
    obs_dict = self.bus.sync_read("Present_Position")
    # 返回值: {"shoulder_pan": 0.5, ...} (已归一化的物理值)
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

    # Step 2: 读取相机图像
    # 逐个相机获取最近帧(异步读,内部由后台线程持续采集)
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()

    return obs_dict

### `send_action()` 发送目标位置

```python
# so101_follower.py:272
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """
    命令机械臂移动到目标关节配置
    
    Args:
        action: 动作字典，键如"{motor_name}.pos"，值如目标位置（归一化值）
            - "shoulder_pan.pos": float - 目标水平旋转角度（范围-100~100）
            - "shoulder_lift.pos": float - 目标肩部抬起角度
            - "elbow_flex.pos": float - 目标肘部弯曲角度
            - "wrist_flex.pos": float - 目标腕部弯曲角度
            - "wrist_roll.pos": float - 目标腕部旋转角度
            - "gripper.pos": float - 目标夹爪开度（范围0~100）
    
    Returns:
        dict[str, Any]: 实际发送给舵机的动作字典（可能经过安全限幅处理）
            - 格式与输入相同，但值可能被修改（如超出安全范围被截断）
    
    Raises:
        DeviceNotConnectedError: 如果机器人未连接
    
    底层执行流程:
        1. 提取目标位置（去掉 ".pos" 后缀）
        2. 安全限幅（如果设置了 max_relative_target）
        3. bus.sync_write("Goal_Position", goal_pos) → 串口发送 Sync Write 包
    
    示例:
        action: {"shoulder_pan.pos": 0.7, "gripper.pos": 75.0}
        return: {"shoulder_pan.pos": 0.7, "gripper.pos": 75.0}（如果无限幅）
    """
    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # Step 1: 从action提取目标位置
    # action: {"shoulder_pan.pos": 0.7, ...}
    # goal_pos: {"shoulder_pan": 0.7, ...}
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

    # Step 2: 安全限幅(可选)
    # 若设置了max_relative_target,限制单帧最大位移（防止机器人突然猛动）
    if self.config.max_relative_target is not None:
        present_pos = self.bus.sync_read("Present_Position")
        goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
        goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

    # Step 3: 发送目标位置到机械臂
    self.bus.sync_write("Goal_Position", goal_pos)

    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

---

## 第 4 层：FeetechMotorsBus — 舵机总线

**文件：** `lerobot/src/lerobot/motors/feetech/feetech.py`

### 初始化

```python
# feetech.py:116
def __init__(
    self, 
    port: str,                                    # 串口设备文件路径
    motors: dict[str, Motor],                    # 舵机名称 → Motor 描述的映射
    calibration: dict[str, MotorCalibration] = None,  # 从 JSON 加载的标定数据
    protocol_version: int = 0                     # 通信协议版本号，0=Feetech SCS 协议
):
    """
    初始化 Feetech 舵机总线，创建串口通信对象
    
    Args:
        port (str): 串口设备路径，如 "/dev/ttyACM0"（本命令）
        motors (dict[str, Motor]): 舵机定义，如 {"shoulder_pan": Motor(1, "sts3215", norm_mode)}
        calibration (dict[str, MotorCalibration] | None): 标定数据，从 JSON 加载（本命令）
            包含 homing_offset, drive_mode, start_pos, end_pos 等字段
        protocol_version (int): 协议版本，0=SCS协议（Feetech 舵机专用）（本命令 0）
    """
    super().__init__(port, motors, calibration)
    self.protocol_version = protocol_version
    self._assert_same_protocol()  # 确保所有舵机使用同一协议版本

    import scservo_sdk as scs

    # Step 1: 创建串口处理器
    self.port_handler = scs.PortHandler(self.port)
    # ── PortHandler 封装了 POSIX open()/read()/write()/close() 系统调用
    # ── 创建时不会立即打开串口，需后续调用 openPort()

    # HACK: 修补官方SDK的超时计算bug
    self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
        self.port_handler, scs.PortHandler
    )

    # Step 2: 创建协议处理器
    self.packet_handler = scs.PacketHandler(protocol_version)
    # ── protocol_version=0 → 使用 SCS 协议（Feetech 舵机专用）
    # ── 负责: 添加帧头 FF FF、计算校验和、解析响应包

    # Step 3: 创建同步读/写对象
    self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
    self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
    # ── 初始地址和长度为 0，每次使用前重新设定
```

**`MotorCalibration` 数据结构：**

```python
@dataclass
class MotorCalibration:
    homing_offset: int      # 归零偏移量（原始编码器值），校准时确定
    drive_mode: int         # 驱动模式（0=正转, 1=反转），用于修正安装方向
    start_pos: int          # 校准时记录的起始位置（原始编码器值）
    end_pos: int            # 校准时记录的结束位置（原始编码器值）
```

**内部创建的关键对象：**

```python
super().__init__(port, motors, calibration)
self.protocol_version = protocol_version  # 0 = SCS 协议
self._assert_same_protocol()             # 确保所有舵机使用同一协议版本

import scservo_sdk as scs

# Step 1: 创建串口处理器 — 管理物理串口的打开/关闭/读写
self.port_handler = scs.PortHandler(self.port)
# ── PortHandler 封装了 POSIX open()/read()/write()/close() 系统调用
# ── 创建时不会立即打开串口，需后续调用 openPort()

# HACK: 修补官方SDK的超时计算bug
self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
    self.port_handler, scs.PortHandler
)

# Step 2: 创建协议处理器 — 负责按照协议格式组包/解包/校验
self.packet_handler = scs.PacketHandler(protocol_version)
# ── protocol_version=0 → 使用 SCS 协议（Feetech 舵机专用）
# ── 负责: 添加帧头 FF FF、计算校验和、解析响应包

# Step 3: 创建同步读/写对象 — 批量操作多个舵机
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
# ── 参数: (port_handler, packet_handler, 初始起始地址, 初始数据长度)
# ── 初始地址和长度为 0，每次使用前通过 start_address/data_length 重新设定

self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
# ── 同上，每次使用前重新设定地址和长度
```

### `_split_into_byte_chunks()` — 整数转字节

```python
# feetech.py:69
def _split_into_byte_chunks(value: int, length: int) -> list[int]:
```

**参数：**

| 参数 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `value` | `int` | 要转换的整数值 | `2048`（Goal_Position 的原始编码器值） |
| `length` | `int` | 目标字节数（1、2 或 4） | `2`（Goal_Position 是 2 字节寄存器） |

**返回值：** `list[int]` — 小端序字节列表

```python
# 示例: _split_into_byte_chunks(2048, 2)
# 2048 = 0x0800
# 低字节 = 0x00，高字节 = 0x08
# 返回: [0x00, 0x08]

if length == 1:
    return [value]                                         # 单字节直接返回
elif length == 2:
    return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)] # 拆成低字节+高字节
elif length == 4:
    return [...]                                           # 拆成4字节
```

---

## 第 5 层：motors_bus.py — 舵机总线抽象层

**文件：** `lerobot/src/lerobot/motors/motors_bus.py`

### `MotorsBus` 核心属性

```python
class MotorsBus(abc.ABC):
    # ── 由 __init__ 传入 ──
    port: str                    # str，串口设备路径，如 "/dev/ttyACM0"
    motors: dict[str, Motor]    # 舵机名称 → Motor 描述对象
                                 # 如 {"shoulder_pan": Motor(1, "sts3215", RANGE_M100_100)}
    calibration: dict[str, MotorCalibration]  # 舵机名称 → 标定数据（offset 等）

    # ── 由子类（FeetechMotorsBus）创建 ──
    port_handler: PortHandler    # scservo_sdk 串口句柄，封装 open/read/write/close
    packet_handler: PacketHandler  # scservo_sdk 协议处理器，负责组包解包
    sync_reader: GroupSyncRead   # scservo_sdk 同步读对象，批量读多个舵机
    sync_writer: GroupSyncWrite  # scservo_sdk 同步写对象，批量写多个舵机
```

---

### `sync_read()` — 同步读取多个舵机

```python
# motors_bus.py:1053
def sync_read(
    self,
    data_name: str,                         # 寄存器名称字符串
    motors: str | list[str] | None = None,  # 要读的舵机
    *,
    normalize: bool = True,                 # 是否归一化到物理单位
    num_retry: int = 0,                     # 失败重试次数
) -> dict[str, float]:
```

**参数详解：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `data_name` | `str` | 控制表中的寄存器名称，用于查表获取地址和长度 | `"Present_Position"` |
| `motors` | `str \| list[str] \| None` | 指定要读的舵机名：`str`=单个, `list`=多个, `None`=全部6个 | `None`（读全部） |
| `normalize` | `bool` | 是否将原始编码器值(0~4095)归一化为物理单位(-100~100) | `True` |
| `num_retry` | `int` | 通信失败时的重试次数 | `0`（不重试） |

**返回值：** `dict[str, float]` — 舵机名称到物理值的映射

```python
# 示例返回值:
{"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 0.1,
 "wrist_flex": 0.2, "wrist_roll": 0.0, "gripper": 50.0}
```

**内部执行流程：**

```python
# 1. 查控制表: "Present_Position" → (addr=56, length=2)
#    通过 get_address(self.model_ctrl_table, model, data_name) 查找
#    model_ctrl_table 是 STS_SMS_SERIES_CONTROL_TABLE（定义在 tables.py）

# 2. 解析电机名 → ID列表
#    motors=None → 使用全部 → motor_ids = [1, 2, 3, 4, 5, 6]

# 3. 调用底层 _sync_read
raw_values, comm = self._sync_read(addr=56, length=2, motor_ids=[1,2,3,4,5,6])
# raw_values: {1: 2048, 2: 1800, ...}  — 原始编码器整数值（0~4095）

# 4. _decode_sign() — 处理有符号寄存器（Present_Position 是无符号的，此步不修改）

# 5. _normalize() — 原始值 → 物理值
#    利用标定数据中的 start_pos/end_pos/homing_offset 做线性映射:
#    normalized = (raw - start) / (end - start) * 200 - 100
#    如 raw=2048, start=0, end=4095 → normalized ≈ 0.0

# 6. 将 ID 映射回舵机名称
#    {1: 0.5} → {"shoulder_pan": 0.5}
```

---

### `_sync_read()` — 底层同步读

```python
# motors_bus.py:1102
def _sync_read(
    self,
    addr: int,              # 寄存器起始地址
    length: int,            # 每个舵机读取的字节数
    motor_ids: list[int],   # 参与同步读的舵机 ID 列表
    *,
    num_retry: int = 0,     # 失败重试次数
    raise_on_error: bool = True,  # 通信失败时是否抛异常
) -> tuple[dict[int, int], int]:
```

**参数详解：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `addr` | `int` | 寄存器起始地址（查控制表得到） | `56`（Present_Position） |
| `length` | `int` | 每个舵机返回的数据字节数 | `2`（位置是2字节） |
| `motor_ids` | `list[int]` | 参与读操作的舵机总线 ID 列表 | `[1, 2, 3, 4, 5, 6]` |
| `num_retry` | `int` | 通信失败重试次数 | `0` |
| `raise_on_error` | `bool` | 通信失败时是否抛异常（False 则只记录警告） | `True` |

**返回值：** `tuple[dict[int, int], int]`

| 返回值分量 | 类型 | 含义 | 示例 |
|------------|------|------|------|
| `values` | `dict[int, int]` | 舵机ID → 原始编码器整数值 | `{1: 2048, 2: 1800, 3: 2100, 4: 1950, 5: 2050, 6: 1500}` |
| `comm` | `int` | 通信状态码，0=成功 | `0`（成功） |

**内部执行流程：**

```python
# Step 1: 配置同步读对象
self._setup_sync_reader(motor_ids=[1,2,3,4,5,6], addr=56, length=2)

# Step 2: 发送 Sync Read 指令包，阻塞等待所有舵机响应
# sync_reader.txRxPacket() 内部:
#   (a) 组装指令包: FF FF FE 0B 82 38 00 02 00 01 02 03 04 05 06 [CHK]
#   (b) PortHandler.writePort() 发送指令包 → write() 系统调用
#   (c) 循环等待各舵机返回 Status 包 → PortHandler.readPort() → read() 系统调用
#   (d) 检查校验和，解析数据
for n_try in range(1 + num_retry):
    comm = self.sync_reader.txRxPacket()
    if self._is_comm_success(comm):
        break

# Step 3: 从 sync_reader 的内部缓冲区提取各舵机数据
values = {id_: self.sync_reader.getData(id_, addr=56, length=2) for id_ in motor_ids}
# getData() 从已接收的 Status 包中提取指定地址的 2 字节数据，拼成整数
return values, comm
```

---

### `_setup_sync_reader()` — 配置同步读

```python
# motors_bus.py:1131
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
```

**参数：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `motor_ids` | `list[int]` | 要读取的舵机 ID 列表 | `[1, 2, 3, 4, 5, 6]` |
| `addr` | `int` | 要读取的寄存器起始地址 | `56`（Present_Position） |
| `length` | `int` | 每个舵机返回的数据字节数 | `2` |

**返回值：** `None`（修改 `self.sync_reader` 的内部状态）

```python
self.sync_reader.clearParam()          # 清空上一次添加的舵机 ID 参数列表
self.sync_reader.start_address = addr   # 设置寄存器起始地址（如 56）
self.sync_reader.data_length = length   # 设置数据长度（如 2 字节）
for id_ in motor_ids:
    self.sync_reader.addParam(id_)     # 逐个添加要读取的舵机 ID
# addParam() 内部为每个 ID 分配一个参数存储槽
```

---

### `sync_write()` — 同步写入多个舵机

```python
# motors_bus.py:1153
def sync_write(
    self,
    data_name: str,                    # 寄存器名称
    values: float | dict[str, float],  # 要写入的值
    *,
    normalize: bool = True,            # 是否将物理值反归一化为原始值
    num_retry: int = 0,                # 失败重试次数
) -> None:
```

**参数详解：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `data_name` | `str` | 控制表中的寄存器名称 | `"Goal_Position"` |
| `values` | `float \| dict[str, float]` | 目标值。`float`=所有舵机写同一值；`dict`=每个舵机写不同值 | `{"shoulder_pan": 0.7, "shoulder_lift": -0.2, ...}` |
| `normalize` | `bool` | 是否先将物理值(-100~100)反归一化为原始编码器值(0~4095) | `True` |
| `num_retry` | `int` | 发送失败重试次数 | `0` |

**返回值：** `None`（sync_write 是无应答模式，不等待舵机确认）

**内部执行流程：**

```python
# Step 1: 将舵机名称转换为 ID，值不变
ids_values = self._get_ids_values_dict(values)
# 输入: {"shoulder_pan": 0.7, ...}
# 输出: {1: 0.7, 2: -0.2, ...}  — 舵机ID → 物理值

# Step 2: 查控制表获取寄存器地址和长度
model = next(iter(models))  # 取第一个舵机的型号，如 "sts3215"
addr, length = get_address(self.model_ctrl_table, model, data_name)
# "Goal_Position" → (addr=42, length=2)

# Step 3: 反归一化 — 物理值 → 原始编码器值
if normalize and data_name in self.normalized_data:
    ids_values = self._unnormalize(ids_values)
# 输入: {1: 0.7, ...}  — 物理值（-100~100）
# 输出: {1: 2143, ...}  — 原始编码器值（0~4095）
# 公式: raw = (value + 100) / 200 * (end_pos - start_pos) + start_pos

# Step 4: 符号位编码 — 处理可能为负的寄存器值
ids_values = self._encode_sign(data_name, ids_values)
# Present_Position/Goal_Position 是无符号的，此步不修改

# Step 5: 调用底层同步写
self._sync_write(addr=42, length=2, ids_values={1: 2143, ...})
```

---

### `_sync_write()` — 底层同步写

```python
# motors_bus.py:1195
def _sync_write(self, addr, length, ids_values, *, num_retry=0) -> int:
```

**参数：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `addr` | `int` | 目标寄存器起始地址 | `42`（Goal_Position） |
| `length` | `int` | 每个舵机写入的数据字节数 | `2` |
| `ids_values` | `dict[int, int]` | 舵机ID → 原始编码器整数值 | `{1: 2143, 2: 1800, ...}` |
| `num_retry` | `int` | 发送失败重试次数 | `0` |

**返回值：** `int` — 通信状态码（0=成功）

**内部执行流程：**

```python
# Step 1: 配置同步写对象
self._setup_sync_writer(ids_values={1: 2143, ...}, addr=42, length=2)
# 内部执行:
#   sync_writer.clearParam()                    # 清空上一次参数
#   sync_writer.start_address = 42              # Goal_Position 寄存器地址
#   sync_writer.data_length = 2                 # 写 2 字节
#   对每个舵机:
#     data = _serialize_data(2143, 2)           # [0x2F, 0x08]（小端序）
#     sync_writer.addParam(id_=1, data=data)    # 将 {ID, data} 加入发送列表

# Step 2: 发送 Sync Write 广播包（一次性发给所有舵机，不等响应）
for n_try in range(1 + num_retry):
    comm = self.sync_writer.txPacket()  # → PortHandler.writePort() → write() 系统调用
    if self._is_comm_success(comm):
        break
# txPacket() 内部:
#   (a) 将所有 (id, data) 对拼接成一个 Sync Write 指令包
#   (b) 计算校验和
#   (c) 通过 PortHandler.writePort() 发送
#   (d) Sync Write 不需要接收响应包

return comm  # 0=发送成功
```

---

## 第 6 层：scservo_sdk — 底层串口通信

**包位置：** `venv/lib/python3.13/site-packages/scservo_sdk/`

### 核心对象

#### `PortHandler(port: str)` — 串口管理

```python
# port_handler.py
port_handler = scs.PortHandler("/dev/ttyACM0")
```

**构造参数：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `port` | `str` | 串口设备文件路径 | `"/dev/ttyACM0"` |

**关键方法：**

```python
port_handler.openPort()
# ── 参数: 无
# ── 返回值: bool — True=打开成功
# ── 作用: 调用 POSIX open() 打开串口设备，设置 O_RDWR | O_NOCTTY | O_NONBLOCK
# ── 然后调用 tcsetattr() 设置 8N1（8数据位、无校验、1停止位）

port_handler.closePort()
# ── 参数: 无
# ── 返回值: 无
# ── 作用: 调用 close() 关闭串口文件描述符

port_handler.setBaudRate(baudrate)
# ── 参数:
#   baudrate: int — 波特率，如 1000000 (1Mbps)
# ── 返回值: bool — True=设置成功
# ── 作用: 调用 cfsetospeed()/cfsetispeed() 设置串口波特率
# ── 同时计算 tx_time_per_byte = (1000.0 / baudrate) * 10.0
#   1Mbps 时: tx_time_per_byte = 10 微秒/字节
#   一帧 Sync Write 约 20 字节: 传输耗时 ~200 微秒

port_handler.writePort(packet)
# ── 参数:
#   packet: list[int] — 要发送的字节列表（完整的协议包）
#   如 [0xFF, 0xFF, 0xFE, 0x13, 0x83, 0x2A, 0x00, 0x02, 0x00, ...]
# ── 返回值: int — 实际写入的字节数
# ── 作用: 调用 POSIX write() 系统调用将字节写入串口

port_handler.readPort(length)
# ── 参数:
#   length: int — 要读取的字节数
# ── 返回值: list[int] — 实际读到的字节列表（可能少于 length）
# ── 作用: 调用 POSIX read() 系统调用从串口读取数据

port_handler.setPacketTimeout(packet_length)
# ── 参数:
#   packet_length: int — 预期包的字节长度
# ── 作用: 根据包长度和波特率计算超时时间
#   packet_timeout = packet_length * tx_time_per_byte + 4.0 (毫秒)

port_handler.isPacketTimeout()
# ── 返回值: bool — True=已超时
# ── 作用: 检查从最后一次 setPacketTimeout() 调用后是否已超时
```

#### `PacketHandler(protocol_version)` — 协议处理

```python
# protocol_packet_handler.py
packet_handler = scs.PacketHandler(0)  # 0 = Feetech SCS 协议
```

**构造参数：**

| 参数 | 类型 | 含义 | 本命令实际值 |
|------|------|------|-------------|
| `protocol_version` | `float` | 协议版本号。0=SCS协议（Feetech舵机） | `0` |

**关键方法：**

```python
# 单舵机读 2 字节寄存器（有应答）
result, data, error = packet_handler.read2ByteTxRx(port_handler, id, address)
# ── 参数:
#   port_handler: PortHandler — 串口句柄
#   id: int — 目标舵机的总线地址，如 1
#   address: int — 要读取的寄存器起始地址，如 56 (Present_Position)
# ── 返回值:
#   result: int — 通信状态码，0=成功
#   data: int — 读到的 2 字节数据拼接成的整数（小端序），如 2048
#   error: int — 舵机内部错误码，0=无错误

# 单舵机写 2 字节寄存器（有应答）
result, error = packet_handler.write2ByteTxRx(port_handler, id, address, value)
# ── 参数:
#   port_handler: PortHandler — 串口句柄
#   id: int — 目标舵机地址，如 1
#   address: int — 要写入的寄存器起始地址，如 42 (Goal_Position)
#   value: int — 要写入的 2 字节整数值，如 2048
# ── 返回值:
#   result: int — 通信状态码，0=成功
#   error: int — 舵机错误码
# ── 注意: write2ByteTxRx 会等待舵机返回 Status 包确认（比 Sync Write 慢但可靠）
```

#### `GroupSyncRead` — 批量同步读

```python
# group_sync_read.py
sync_reader = scs.GroupSyncRead(port_handler, packet_handler, ph_addr, data_length)
```

**构造参数：**

| 参数 | 类型 | 含义 | 备注 |
|------|------|------|------|
| `port_handler` | `PortHandler` | 串口句柄 | 由 FeetechMotorsBus 创建 |
| `packet_handler` | `PacketHandler` | 协议处理器 | 由 FeetechMotorsBus 创建 |
| `ph_addr` | `int` | 初始起始地址 | 传入 `0`，使用前会重新设置 |
| `data_length` | `int` | 初始数据长度 | 传入 `0`，使用前会重新设置 |

**关键属性和方法：**

```python
sync_reader.start_address = 56
# ── int，要读取的寄存器起始地址
# ── 56 = Present_Position

sync_reader.data_length = 2
# ── int，每个舵机返回的数据字节数
# ── Present_Position 是 2 字节

sync_reader.clearParam()
# ── 参数: 无
# ── 返回值: bool — True=清空成功
# ── 作用: 清空内部参数列表（上次添加的舵机 ID 和接收缓存）

sync_reader.addParam(id_)
# ── 参数:
#   id_: int — 要加入同步读的舵机 ID，如 1, 2, ..., 6
# ── 返回值: bool — True=添加成功，False=ID 已存在或列表已满
# ── 作用: 为该 ID 分配接收缓冲区

comm = sync_reader.txRxPacket()
# ── 参数: 无
# ── 返回值: int — 通信状态码，0=全部成功
# ── 作用:
#   (1) 组装 Sync Read 指令包:
#       FF FF FE [Len] 82 [AddrL] [AddrH] [DataLenL] [DataLenH] [ID1] [ID2] ... [CHK]
#   (2) 通过 port_handler.writePort() 发送指令包
#   (3) 循环等待每个舵机返回 Status 包:
#       FF FF [ID] [Len] [Err] [Data1] [Data2] ... [CHK]
#   (4) 将每个舵机的数据存入内部缓冲区

value = sync_reader.getData(id_, address, data_length)
# ── 参数:
#   id_: int — 舵机 ID，如 1
#   address: int — 寄存器地址，如 56
#   data_length: int — 数据字节数，如 2
# ── 返回值: int — 从该舵机的 Status 包中提取的数据值
# ── 如 getData(1, 56, 2) → 返回舵机1在地址56处读到的2字节值（如2048）
# ── 注意: 必须在 txRxPacket() 成功后调用
```

#### `GroupSyncWrite` — 批量同步写

```python
# group_sync_write.py
sync_writer = scs.GroupSyncWrite(port_handler, packet_handler, ph_addr, data_length)
```

**构造参数：**（同 GroupSyncRead）

| 参数 | 类型 | 含义 | 备注 |
|------|------|------|------|
| `port_handler` | `PortHandler` | 串口句柄 | 由 FeetechMotorsBus 创建 |
| `packet_handler` | `PacketHandler` | 协议处理器 | 由 FeetechMotorsBus 创建 |
| `ph_addr` | `int` | 初始起始地址 | 传入 `0`，使用前重新设置 |
| `data_length` | `int` | 初始数据长度 | 传入 `0`，使用前重新设置 |

**关键属性和方法：**

```python
sync_writer.start_address = 42
# ── int，要写入的寄存器起始地址
# ── 42 = Goal_Position

sync_writer.data_length = 2
# ── int，每个舵机写入的数据字节数
# ── Goal_Position 是 2 字节

sync_writer.clearParam()
# ── 参数: 无
# ── 返回值: bool
# ── 作用: 清空内部参数列表（上次添加的舵机 ID 和数据）

sync_writer.addParam(id_, data)
# ── 参数:
#   id_: int — 目标舵机 ID，如 1
#   data: list[int] — 要写入的字节数据，长度必须等于 data_length
#     如 [0x00, 0x08] 表示 Goal_Position=2048（小端序）
# ── 返回值: bool — True=添加成功
# ── 作用: 将 {ID, data} 对加入发送列表

comm = sync_writer.txPacket()
# ── 参数: 无
# ── 返回值: int — 通信状态码，0=发送成功
# ── 作用:
#   (1) 组装 Sync Write 指令包（广播包，ID=0xFE）:
#       FF FF FE [Len] 83 [AddrL] [AddrH] [DataLenL] [DataLenH]
#       [ID1] [D1L] [D1H] [ID2] [D2L] [D2H] ... [CHK]
#   (2) 通过 port_handler.writePort() 发送
#   (3) **不等待舵机响应**（这是 Sync Write 的核心特性）
#       因此延迟极低（只算发送时间），但无法确认舵机是否收到
```

---

## 关键寄存器表（STS3215）

**文件：** `lerobot/src/lerobot/motors/feetech/tables.py`

| 寄存器名 | 地址 | 字节数 | 方向 | 含义 |
|---------|------|-------|------|------|
| `ID` | 5 | 1 | R/W | 舵机总线地址 |
| `Baud_Rate` | 6 | 1 | R/W | 通信波特率 |
| `Min_Position_Limit` | 9 | 2 | R/W | 最小位置限制 |
| `Max_Position_Limit` | 11 | 2 | R/W | 最大位置限制 |
| `Operating_Mode` | 33 | 1 | R/W | 控制模式(位置/速度等) |
| `Torque_Enable` | 40 | 1 | R/W | 1=使能力矩,0=释放 |
| **`Goal_Position`** | **42** | **2** | **W** | **目标位置** |
| `Goal_Velocity` | 46 | 2 | R/W | 目标速度 |
| **`Present_Position`** | **56** | **2** | **R** | **当前位置** |
| `Present_Velocity` | 58 | 2 | R | 当前速度 |
| `Present_Load` | 60 | 2 | R | 当前负载 |
| `Present_Temperature` | 63 | 1 | R | 当前温度 |

---

## 数据流完整示意

### 读取流程（`get_observation`）

```
物理世界: 舵机编码器值 0 ~ 4095 (共4096档)
    ↓ 串口接收 (read()系统调用)
GroupSyncRead.getData() → 原始整数(如2048)
    ↓ _decode_sign() (对有符号寄存器)
无符号整数
    ↓ _normalize()
物理单位 (如 shoulder_pan: 0.0 表示中间位置,范围-100~100)
    ↓
obs["shoulder_pan.pos"] = 0.0
```

### 写入流程（`send_action`）

```
action["shoulder_pan.pos"] = 50.0  (目标位置,-100~100)
    ↓ _unnormalize()
原始整数(如3072)
    ↓ _encode_sign()
无符号整数
    ↓ _serialize_data(3072, length=2)
字节列表 [0x00, 0x0C] (小端序)
    ↓ GroupSyncWrite.addParam(id=1, data=[0x00, 0x0C])
    ↓ GroupSyncWrite.txPacket()
    ↓ PortHandler.writePort() (→ write()系统调用)
串口 → 舵机 → 驱动电机运动到目标位置
```

---

## 协议包格式

### Sync Read 指令包（主机→舵机）

```
Byte:  0    1    2    3    4    5    6    7    8    ...   N
      [FF] [FF] [FE] [Len] [82] [AddrL] [AddrH] [DL] [DH] [ID1] [ID2] ... [CHK]
       ↑    ↑         ↑         ↑      ↑      ↑      ↑
     帧头  帧头   广播ID  长度  同步读  起始地址  数据长度  舵机ID列表  校验和
           指令码
```

- `FF FF`: 帧头(固定)
- `FE`: 广播ID(表示发给所有舵机)
- `82`: INST_SYNC_READ指令码
- 示例(读Present_Position=56,2字节,ID=1-6):
  `FF FF FE 0B 82 38 00 02 00 01 02 03 04 05 06 [CHK]`

### Status 响应包（舵机→主机）

```
Byte:  0    1    2    3    4    5    6    7    8
      [FF] [FF] [ID] [Len] [Err] [DataL] [DataH] [CHK]
       ↑    ↑         ↑      ↑      ↑
     帧头  帧头    舵机ID  长度  错误码  数据(2字节) 校验和
```

### Sync Write 指令包（主机→舵机）

```
Byte:  0    1    2    3    4    5    6    7    8    9    10   ...
      [FF] [FF] [FE] [Len] [83] [AddrL] [AddrH] [DL] [DH] [ID1] [D1L] [D1H] ...
       ↑    ↑         ↑         ↑      ↑      ↑      ↑
     帧头  帧头   广播ID  长度  同步写  起始地址  数据长度
                                                          舵机1数据
```

- `83`: INST_SYNC_WRITE指令码
- 示例(写Goal_Position=42,2字节):
  `FF FF FE 13 83 2A 00 02 00 01 [pos1L] [pos1H] 02 [pos2L] [pos2H] ... [CHK]`

---

## 文件路径速查表

| 层次 | 文件路径 | 核心类/函数 |
|------|---------|-----------|
| 入口 | `lerobot/src/lerobot/record.py` | `record()`, `record_loop()` |
| 工厂 | `lerobot/src/lerobot/robots/utils.py` | `make_robot_from_config()` |
| 机器人 | `lerobot/src/lerobot/robots/so101_follower/so101_follower.py` | `SO101Follower` |
| 示教手 | `lerobot/src/lerobot/teleoperators/so101_leader/so101_leader.py` | `SO101Leader` |
| 总线抽象 | `lerobot/src/lerobot/motors/motors_bus.py` | `MotorsBus`, `sync_read()`, `sync_write()` |
| Feetech | `lerobot/src/lerobot/motors/feetech/feetech.py` | `FeetechMotorsBus` |
| 控制表 | `lerobot/src/lerobot/motors/feetech/tables.py` | `STS_SMS_SERIES_CONTROL_TABLE` |
| SDK | `scservo_sdk` (第三方) | `PortHandler`, `PacketHandler`, `GroupSyncRead`, `GroupSyncWrite` |
| 相机 | `lerobot/src/lerobot/cameras/opencv/camera_opencv.py` | `OpenCVCamera`, `async_read()` |
| 策略推理 | `lerobot/src/lerobot/utils/control_utils.py` | `predict_action()` |

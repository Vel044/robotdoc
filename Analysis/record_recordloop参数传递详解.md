# `record()` 和 `record_loop()` 函数参数传递详解

本文档追踪从命令行 `python -m lerobot.record` 到 `record()` 再到 `record_loop()` 的完整参数传递链，详细说明每个参数的类型、数据结构和实际值。

---

## 1. 命令行输入

```bash
python -m lerobot.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12254705 \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=R07254705 \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}" \
    --dataset.single_task="Put the bottle into the black basket." \
    --policy.path=/home/vel/so101-bottle/last/pretrained_model \
    --dataset.repo_id=${HF_USER}/eval_so101_bottle --dataset.push_to_hub=false
```

### 入口机制

- `python -m lerobot.record` 调用 `lerobot/src/lerobot/record.py` 的 `main()` 函数（L658-L659）
- `main()` 直接调用 `record()`
- `record()` 被 `@parser.wrap()` 装饰器包裹，使用 **draccus** 库将命令行参数解析为 `RecordConfig` 对象
- draccus 通过 `--` 前缀 + 点号嵌套（如 `--robot.type`）映射到嵌套 dataclass 字段

---

## 2. 核心类型别名定义

`record_loop()` 的类型注解使用了以下类型别名（定义在 `processor/core.py:L39-L42`）：

```python
# 基本类型别名
PolicyAction: TypeAlias = torch.Tensor         # 策略输出，是一个 torch 张量
RobotAction: TypeAlias = dict[str, Any]         # 机器人动作，键值对，如 {"shoulder_pan": 0.5, ...}
EnvAction: TypeAlias = np.ndarray               # 环境动作，numpy 数组
RobotObservation: TypeAlias = dict[str, Any]     # 机器人观测，键值对，如 {"shoulder_pan": 0.3, "handeye": <PIL.Image>, ...}

# Pipeline 类型别名（定义在 processor/pipeline.py:L1435-L1436）
RobotProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]
PolicyProcessorPipeline: TypeAlias = DataProcessorPipeline[TInput, TOutput]
# 两者本质上是 DataProcessorPipeline 的泛型别名，只是语义上的区分
```

### `DataProcessorPipeline` 结构

```python
class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):
    name: str = "DataProcessorPipeline"
    steps: list[ProcessorStep]          # 处理步骤列表
    to_transition: Callable             # 输入 → 内部转移格式转换函数
    to_output: Callable                 # 内部转移格式 → 输出转换函数
```

默认情况下，所有三个 processor 都只包含 `IdentityProcessorStep()`（恒等变换，不修改数据）。

---

## 3. `RecordConfig` 完整配置对象

### 3.1 顶层结构

```python
@dataclass
class RecordConfig:
    robot: RobotConfig                                    # 机器人配置
    dataset: DatasetRecordConfig                          # 数据集配置
    teleop: TeleoperatorConfig | None = None              # 遥控器配置（可选）
    policy: PreTrainedConfig | None = None                # 策略配置（可选）
    display_data: bool = False                            # 是否显示数据可视化
    play_sounds: bool = True                              # 是否播放语音提示
    resume: bool = False                                  # 是否恢复录制
```
**源文件**: `lerobot/src/lerobot/record.py:L199-L229`

### 3.2 `robot: SO101FollowerConfig`

继承链: `SO101FollowerConfig` → `RobotConfig` → `draccus.ChoiceRegistry`

```python
@dataclass
class SO101FollowerConfig(RobotConfig):
    port: str                                      # "/dev/ttyACM0" — 串口设备路径
    disable_torque_on_disconnect: bool = True       # True — 断开连接时禁用扭矩
    max_relative_target: float | dict | None = None # None — 安全限幅（未启用）
    cameras: dict[str, CameraConfig] = {...}        # 相机配置字典
    use_degrees: bool = False                       # False — 使用弧度单位
    # 继承自 RobotConfig:
    id: str | None = None                           # "R12254705"
    calibration_dir: Path | None = None             # None（默认 ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/）
```
**源文件**: `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py`

#### `cameras` 字典详细结构

```python
cameras = {
    "handeye": OpenCVCameraConfig(
        type="opencv",              # 注册名（通过 @CameraConfig.register_subclass("opencv")）
        index_or_path=0,            # 对应 /dev/video0
        width=640,                  # 帧宽度
        height=360,                 # 帧高度
        fps=30,                     # 帧率
        color_mode=ColorMode.RGB,   # 颜色模式（默认 RGB）
        rotation=Cv2Rotation.NO_ROTATION,  # 图像旋转（默认无旋转）
        warmup_s=1,                 # 连接后预热秒数
    ),
    "fixed": OpenCVCameraConfig(
        type="opencv",
        index_or_path=2,            # 对应 /dev/video2
        width=640,
        height=360,
        fps=30,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        warmup_s=1,
    ),
}
```
**源文件**: `lerobot/src/lerobot/cameras/opencv/configuration_opencv.py`

### 3.3 `teleop: SO101LeaderConfig`

继承链: `SO101LeaderConfig` → `TeleoperatorConfig` → `draccus.ChoiceRegistry`

```python
@dataclass
class SO101LeaderConfig(TeleoperatorConfig):
    port: str                    # "/dev/ttyACM1" — 遥控臂串口
    use_degrees: bool = False    # False — 使用弧度单位
    # 继承自 TeleoperatorConfig:
    id: str | None = None        # "R07254705"
    calibration_dir: Path | None = None
```
**源文件**: `lerobot/src/lerobot/teleoperators/so101_leader/config_so101_leader.py`

### 3.4 `policy: PreTrainedConfig`

```python
@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    n_obs_steps: int = 1                                # 观测步数
    input_features: dict[str, PolicyFeature] = {...}     # 从预训练模型 config.json 加载
    output_features: dict[str, PolicyFeature] = {...}    # 从预训练模型 config.json 加载
    device: str | None = None                            # 推理设备（自动推断，树莓派为 "cpu"）
    use_amp: bool = False                                # 混合精度（CPU 上自动关闭）
    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool | None = None
    tags: list[str] | None = None
    license: str | None = None
    pretrained_path: str = "/home/vel/so101-bottle/last/pretrained_model"  # __post_init__ 设置
```
**加载方式**: `RecordConfig.__post_init__()` 中通过 `PreTrainedConfig.from_pretrained(policy_path)` 从本地目录加载 JSON 配置
**源文件**: `lerobot/src/lerobot/configs/policies.py`

### 3.5 `dataset: DatasetRecordConfig`

```python
@dataclass
class DatasetRecordConfig:
    repo_id: str = "${HF_USER}/eval_so101_bottle"         # 数据集仓库 ID
    single_task: str = "Put the bottle into the black basket."  # 任务描述
    root: str | Path | None = None                        # 数据集根目录（默认 ~/.cache/huggingface/lerobot）
    fps: int = 30                                         # 目标帧率
    episode_time_s: int | float = 60                      # 每个 episode 录制时长（秒）
    reset_time_s: int | float = 60                        # 重置环境等待时长（秒）
    num_episodes: int = 50                                # 总 episode 数
    video: bool = True                                    # 是否编码为视频
    push_to_hub: bool = False                             # 是否上传到 Hub（本命令为 False）
    private: bool = False                                 # 是否私有仓库
    tags: list[str] | None = None                         # Hub 标签
    num_image_writer_processes: int = 0                   # 图片写子进程数（0=只用线程）
    num_image_writer_threads_per_camera: int = 4          # 每相机图片写线程数
    video_encoding_batch_size: int = 1                    # 视频编码批大小
    rename_map: dict[str, str] = {}                       # 观测/动作重命名映射
```
**源文件**: `lerobot/src/lerobot/record.py:L156-L197`

---

## 4. `record()` 函数

### 函数签名

```python
@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
```
**位置**: `lerobot/src/lerobot/record.py:L499-L500`

### 入参

唯一的参数是 `cfg: RecordConfig`（上面第 3 节描述的完整配置对象）。

### 函数内部派生的组件

`record()` 函数从 `cfg` 派生出以下组件，最终传给 `record_loop()`：

#### 4.1 `robot: SO101Follower`

```python
robot = make_robot_from_config(cfg.robot)  # L506
```
- 工厂函数根据 `cfg.robot.type == "so101_follower"` 导入并实例化 `SO101Follower(cfg.robot)`
- **类型**: `Robot` 的子类 `SO101Follower`
- **关键属性**:
  - `robot.name` → `"so101_follower"`（机器人类型名称）
  - `robot.robot_type` → `"so101_follower"`
  - `robot.id` → `"R12254705"`
  - `robot.cameras` → `{"handeye": OpenCVCamera, "fixed": OpenCVCamera}`（connect() 后可用）
  - `robot.action_features` → `{"shoulder_pan": float, "shoulder_lift": float, ...}`（6 个关节的 float）
  - `robot.observation_features` → `{"shoulder_pan": float, ..., "handeye": (360, 640, 3), "fixed": (360, 640, 3)}`

#### 4.2 `teleop: SO101Leader`

```python
teleop = make_teleoperator_from_config(cfg.teleop)  # L507
```
- 工厂函数根据 `cfg.teleop.type == "so101_leader"` 导入并实例化 `SO101Leader(cfg.teleop)`
- **类型**: `Teleoperator` 的子类 `SO101Leader`

#### 4.3 三个 Processor Pipeline

```python
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()  # L509
```

| 变量 | 泛型签名 | 输入类型 | 输出类型 | 内部步骤 | 转换函数 |
|------|----------|----------|----------|----------|----------|
| `teleop_action_processor` | `RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]` | `(dict, dict)` | `dict` | `[IdentityProcessorStep]` | `robot_action_observation_to_transition` → `transition_to_robot_action` |
| `robot_action_processor` | `RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]` | `(dict, dict)` | `dict` | `[IdentityProcessorStep]` | `robot_action_observation_to_transition` → `transition_to_robot_action` |
| `robot_observation_processor` | `RobotProcessorPipeline[RobotObservation, RobotObservation]` | `dict` | `dict` | `[IdentityProcessorStep]` | `observation_to_transition` → `transition_to_observation` |

> **注意**: 默认 processor 都是恒等变换。LeRobot v0.3.4 的处理逻辑中，默认情况下这三个 pipeline 不修改数据，直接透传。

#### 4.4 `dataset: LeRobotDataset`

```python
dataset = LeRobotDataset.create(          # L542-L552
    cfg.dataset.repo_id,                  # "${HF_USER}/eval_so101_bottle"
    cfg.dataset.fps,                      # 30
    root=cfg.dataset.root,                # None
    robot_type=robot.name,                # "so101_follower"
    features=dataset_features,            # 合并后的数据集特征定义（见下）
    use_videos=cfg.dataset.video,         # True
    image_writer_processes=0,
    image_writer_threads=8,               # 4 threads/camera × 2 cameras
    batch_encoding_size=1,
)
```

**`dataset_features` 的构建过程** (L511-L524):

```python
dataset_features = combine_feature_dicts(
    # 动作特征（来自 robot.action_features）
    aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=True,
    ),
    # 观测特征（来自 robot.observation_features）
    aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=True,
    ),
)
```

最终 `dataset_features` 结构示例：
```python
{
    "action": {
        "dtype": "float32",
        "shape": [6],
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [6],
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
    },
    "observation.images.handeye": {
        "dtype": "video",
        "shape": [3, 360, 640],    # CHW 格式
        "names": ["channel", "height", "width"],
    },
    "observation.images.fixed": {
        "dtype": "video",
        "shape": [3, 360, 640],
        "names": ["channel", "height", "width"],
    },
}
```

#### 4.5 `policy: PreTrainedPolicy`

```python
policy = make_policy(cfg.policy, ds_meta=dataset.meta)  # L555
```
- 从本地路径 `/home/vel/so101-bottle/last/pretrained_model` 加载模型权重
- **类型**: `PreTrainedPolicy`（具体实现可能是 `ACTPolicy` 等）

#### 4.6 `preprocessor` 和 `postprocessor`

```python
preprocessor, postprocessor = make_pre_post_processors(     # L559-L567
    policy_cfg=cfg.policy,
    pretrained_path=cfg.policy.pretrained_path,
    dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
    preprocessor_overrides={
        "device_processor": {"device": cfg.policy.device},
        "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
    },
)
```
- **preprocessor**: `PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]` — 输入预处理（归一化、设备转移等）
- **postprocessor**: `PolicyProcessorPipeline[PolicyAction, PolicyAction]` — 输出后处理（反归一化等）

#### 4.7 `events: dict`

```python
listener, events = init_keyboard_listener()  # L573
```

`events` 字典结构（`control_utils.py:L208-L211`）：

```python
events = {
    "exit_early": False,          # 按 → 键提前结束当前 episode
    "rerecord_episode": False,    # 按 ← 键重录当前 episode
    "stop_recording": False,      # 按 Esc 键停止全部录制
}
```

---

## 5. `record_loop()` 函数参数详解

### 函数签名

```python
@safe_stop_image_writer
def record_loop(
    robot: Robot,                                                    # 执行动作的机器人实例
    events: dict,                                                    # 键盘事件控制字典
    fps: int,                                                        # 目标帧率
    teleop_action_processor: RobotProcessorPipeline[                 # teleop 动作后处理器
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[                  # 机器人动作后处理器
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[             # 观测后处理器
        RobotObservation, RobotObservation
    ],
    dataset: LeRobotDataset | None = None,                           # 数据集实例（None=不记录）
    teleop: Teleoperator | list[Teleoperator] | None = None,         # 遥控器（单个/多个/无）
    policy: PreTrainedPolicy | None = None,                           # 策略模型
    preprocessor: PolicyProcessorPipeline[dict, dict] | None = None,  # 策略预处理器
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,  # 策略后处理器
    control_time_s: int | None = None,                                # 控制循环时长（秒）
    single_task: str | None = None,                                   # 任务描述字符串
    display_data: bool = False,                                       # 是否可视化
):
```
**位置**: `lerobot/src/lerobot/record.py:L263-L284`

### 参数说明

| 参数 | 类型 | 录制阶段实际值 | 重置阶段实际值 | 说明 |
|------|------|---------------|---------------|------|
| `robot` | `SO101Follower` | SO101Follower 实例 | 同左 | 执行动作的从臂 |
| `events` | `dict[str, bool]` | `{"exit_early": False, "rerecord_episode": False, "stop_recording": False}` | 同左 | 键盘事件控制 |
| `fps` | `int` | `30` | `30` | 目标帧率 |
| `teleop_action_processor` | `RobotProcessorPipeline` | 含 IdentityProcessorStep | 同左 | 处理 teleop 原始动作 |
| `robot_action_processor` | `RobotProcessorPipeline` | 含 IdentityProcessorStep | 同左 | 处理下发给机器人的动作 |
| `robot_observation_processor` | `RobotProcessorPipeline` | 含 IdentityProcessorStep | 同左 | 处理观测数据 |
| `dataset` | `LeRobotDataset \| None` | LeRobotDataset 实例 | `None` | 录制时保存，重置时不保存 |
| `teleop` | `SO101Leader` | SO101Leader 实例 | 同左 | 示教主臂 |
| `policy` | `PreTrainedPolicy \| None` | PreTrainedPolicy 实例 | `None` | 策略推理（重置时不用） |
| `preprocessor` | `PolicyProcessorPipeline \| None` | 实例 | `None` | 策略输入预处理 |
| `postprocessor` | `PolicyProcessorPipeline \| None` | 实例 | `None` | 策略输出后处理 |
| `control_time_s` | `int` | `60`（episode_time_s） | `60`（reset_time_s） | 控制循环时长 |
| `single_task` | `str` | `"Put the bottle into the black basket."` | 同左 | 任务描述 |
| `display_data` | `bool` | `False` | `False` | 可视化开关 |

---

## 6. `record()` → `record_loop()` 两次调用

### 6.1 录制阶段调用 (L593-L608)

```python
record_loop(
    robot=robot,
    events=events,
    fps=cfg.dataset.fps,                               # 30
    teleop_action_processor=teleop_action_processor,
    robot_action_processor=robot_action_processor,
    robot_observation_processor=robot_observation_processor,
    teleop=teleop,
    policy=policy,                                      # 有策略
    preprocessor=preprocessor,                          # 有预处理器
    postprocessor=postprocessor,                        # 有后处理器
    dataset=dataset,                                    # 有数据集，记录数据
    control_time_s=cfg.dataset.episode_time_s,          # 60
    single_task=cfg.dataset.single_task,
    display_data=cfg.display_data,
)
```

### 6.2 重置阶段调用 (L619-L630)

```python
record_loop(
    robot=robot,
    events=events,
    fps=cfg.dataset.fps,                               # 30
    teleop_action_processor=teleop_action_processor,
    robot_action_processor=robot_action_processor,
    robot_observation_processor=robot_observation_processor,
    teleop=teleop,
    # 以下参数不传（使用默认值 None）:
    # policy=None,
    # preprocessor=None,
    # postprocessor=None,
    # dataset=None,                                    # 不保存数据
    control_time_s=cfg.dataset.reset_time_s,            # 60
    single_task=cfg.dataset.single_task,
    display_data=cfg.display_data,
)
```

### 6.3 两次调用的关键差异

| 方面 | 录制阶段 | 重置阶段 |
|------|----------|----------|
| 目的 | 录制 episode 数据，policy 推理 + teleop 记录 | 手动将机器人复位，不记录数据 |
| `policy` | PreTrainedPolicy 实例 | `None`（省略） |
| `dataset` | LeRobotDataset 实例，保存每帧 | `None`（省略） |
| `control_time_s` | `episode_time_s = 60` | `reset_time_s = 60` |

> **注意**: 此命令同时传了 `teleop` 和 `policy`。在 `record_loop()` 内部（L365-L413），当 `policy is not None` 时走策略推理路径，不会同时使用 teleop 获取动作。teleop 在重置阶段才有实际作用。

---

## 7. `record_loop()` 内部数据流

```
record_loop() 每帧执行流程:

[Robot] ──get_observation()──→ obs: RobotObservation (dict)
                                    │
                    robot_observation_processor(obs)
                                    │
                                    ▼
                            obs_processed: RobotObservation (dict)
                                    │
                    build_dataset_frame(features, obs_processed, prefix="observation")
                                    │
                                    ▼
                    observation_frame: dict  (如 {"observation.state": [...], "observation.images.handeye": [...]})
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
     [Policy 路径]                                    [Teleop 路径]
     predict_action(...)                             teleop.get_action()
            │                                               │
            ▼                                               ▼
     action_values: dict                            act: RobotAction (dict)
            │                                               │
            │                                teleop_action_processor((act, obs))
            │                                               │
            │                                               ▼
            │                                act_processed_teleop: RobotAction (dict)
            │
            ▼
     act_processed_policy: RobotAction (dict)
            │
            └───────────────────────┬───────────────────────┘
                                    │
                    robot_action_processor((action, obs))
                                    │
                                    ▼
                    robot_action_to_send: RobotAction (dict)
                                    │
                    robot.send_action(robot_action_to_send)
                                    │
                                    ▼
                            [物理机器人执行动作]
                                    │
                    dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
                                    │
                                    ▼
                    busy_wait(1/fps - elapsed)  ← 帧率控制
```

---

## 8. 命令行参数 → RecordConfig 字段 → record_loop() 参数 映射表

| 命令行参数 | RecordConfig 字段 | record_loop() 参数 | 实际值 |
|------------|-------------------|-------------------|--------|
| `--robot.type=so101_follower` | `cfg.robot` (SO101FollowerConfig) | `robot` (SO101Follower) | 实例化后的机器人 |
| `--robot.port=/dev/ttyACM0` | `cfg.robot.port` | → robot.config.port | `"/dev/ttyACM0"` |
| `--robot.id=R12254705` | `cfg.robot.id` | → robot.id | `"R12254705"` |
| `--robot.disable_torque_on_disconnect` | `cfg.robot.disable_torque_on_disconnect` | → robot.config 属性 | `True` |
| `--robot.cameras={...}` | `cfg.robot.cameras` | → robot.cameras | `{"handeye": ..., "fixed": ...}` |
| `--teleop.type=so101_leader` | `cfg.teleop` (SO101LeaderConfig) | `teleop` (SO101Leader) | 实例化后的遥控臂 |
| `--teleop.port=/dev/ttyACM1` | `cfg.teleop.port` | → teleop.config.port | `"/dev/ttyACM1"` |
| `--teleop.id=R07254705` | `cfg.teleop.id` | → teleop.id | `"R07254705"` |
| `--policy.path=...` | `cfg.policy` (PreTrainedConfig) | `policy` (PreTrainedPolicy) | 加载后的策略模型 |
| `--dataset.repo_id=...` | `cfg.dataset.repo_id` | → dataset.repo_id | `"${HF_USER}/eval_so101_bottle"` |
| `--dataset.single_task=...` | `cfg.dataset.single_task` | `single_task` | `"Put the bottle into the black basket."` |
| `--dataset.push_to_hub=false` | `cfg.dataset.push_to_hub` | （record() 内使用，不传 record_loop） | `False` |
| `--dataset.fps` (默认) | `cfg.dataset.fps` | `fps` | `30` |
| `--dataset.episode_time_s` (默认) | `cfg.dataset.episode_time_s` | `control_time_s`（录制） | `60` |
| `--dataset.reset_time_s` (默认) | `cfg.dataset.reset_time_s` | `control_time_s`（重置） | `60` |
| `--dataset.num_episodes` (默认) | `cfg.dataset.num_episodes` | （record() 外层循环使用） | `50` |
| — | — | `events` | `{"exit_early": False, "rerecord_episode": False, "stop_recording": False}` |
| — | — | `teleop_action_processor` | `RobotProcessorPipeline`（恒等变换） |
| — | — | `robot_action_processor` | `RobotProcessorPipeline`（恒等变换） |
| — | — | `robot_observation_processor` | `RobotProcessorPipeline`（恒等变换） |
| — | — | `preprocessor` | `PolicyProcessorPipeline`（从模型加载） |
| — | — | `postprocessor` | `PolicyProcessorPipeline`（从模型加载） |

---

## 参考资料

| 内容 | 源文件路径 |
|------|-----------|
| 主入口 + record() + record_loop() | `lerobot/src/lerobot/record.py` |
| RecordConfig / DatasetRecordConfig | `lerobot/src/lerobot/record.py:L156-L229` |
| SO101FollowerConfig | `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py` |
| SO101LeaderConfig | `lerobot/src/lerobot/teleoperators/so101_leader/config_so101_leader.py` |
| RobotConfig 基类 | `lerobot/src/lerobot/robots/config.py` |
| TeleoperatorConfig 基类 | `lerobot/src/lerobot/teleoperators/config.py` |
| CameraConfig / OpenCVCameraConfig | `lerobot/src/lerobot/cameras/opencv/configuration_opencv.py` |
| PreTrainedConfig | `lerobot/src/lerobot/configs/policies.py` |
| 类型别名 (RobotAction 等) | `lerobot/src/lerobot/processor/core.py:L39-L42` |
| Pipeline 类型别名 | `lerobot/src/lerobot/processor/pipeline.py:L1435-L1436` |
| make_default_processors | `lerobot/src/lerobot/processor/factory.py:L58-L62` |
| make_robot_from_config | `lerobot/src/lerobot/robots/utils.py` |
| make_teleoperator_from_config | `lerobot/src/lerobot/teleoperators/utils.py` |
| init_keyboard_listener | `lerobot/src/lerobot/utils/control_utils.py:L192-L211` |

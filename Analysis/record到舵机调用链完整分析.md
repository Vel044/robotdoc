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

### 入口函数 `record()`

```python
# record.py:503
@parser.wrap()
def record(cfg: RecordConfig) -> None:
```

**参数 `cfg: RecordConfig` 的主要字段：**

| 字段 | 类型 | 含义 |
|------|------|------|
| `cfg.robot` | `RobotConfig` | 机器人硬件配置（串口、舵机型号等） |
| `cfg.dataset` | `DatasetRecordConfig` | 数据集配置（repo_id、fps、episode数量等） |
| `cfg.teleop` | `TeleoperatorConfig` | 示教设备配置（示教手类型、串口等） |
| `cfg.policy` | `PreTrainedConfig` | 策略配置（模型路径、设备等） |
| `cfg.display_data` | `bool` | 是否开启rerun可视化 |
| `cfg.play_sounds` | `bool` | 是否使用语音播报事件 |
| `cfg.resume` | `bool` | 是否从已有数据集继续录制 |

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
# lerobot/src/lerobot/record.py:262
@safe_stop_image_writer  # 装饰器: 确保图像写入器正确停止
def record_loop(
    robot: Robot,                        # SO101Follower实例
    events: dict,                        # 事件字典: exit_early, rerecord_episode, stop_recording
    fps: int,                            # 目标帧率(Hz),如30
    teleop_action_processor,              # 示教动作处理器
    robot_action_processor,              # 机器人动作处理器
    robot_observation_processor,          # 观测处理器
    dataset: LeRobotDataset | None,       # 数据集(为None时不记录)
    teleop: Teleoperator | None,           # 示教设备
    policy: PreTrainedPolicy | None,      # 策略(可与teleop同时存在)
    preprocessor, postprocessor,         # 策略前后处理器
    control_time_s: int | None,           # 控制时长(秒)
    single_task: str | None,              # 任务描述
    display_data: bool = False,           # 是否可视化
) -> None:
```

**循环内每帧执行流程（每帧约33ms @ 30fps）：**

#### 1. 观测阶段 (Observation)

```python
# lerobot/src/lerobot/record.py:341
# Step 1: 读取硬件观测
obs_hw_start_t = time.perf_counter()
obs = robot.get_observation()  # → SO101Follower.get_observation()
# 返回: {
#     "shoulder_pan.pos": 0.5,    # 关节角度(归一化值)
#     "shoulder_lift.pos": -0.3,
#     "elbow_flex.pos": 0.1,
#     "wrist_flex.pos": 0.2,
#     "wrist_roll.pos": 0.0,
#     "gripper.pos": 50.0,        # 夹爪开度(0-100)
#     "handeye": array(H,W,3),   # 相机图像
#     "fixed": array(H,W,3),
# }
obs_hw_end_t = time.perf_counter()

# Step 2: 观测后处理(可能包含归一化等)
obs_processed = robot_observation_processor(obs)

# Step 3: 打包成observation_frame(用于策略推理)
observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix="observation")
```

#### 2. 推理阶段 (Inference) - 二选一

**路径A: POLICY 策略推理**
```python
# lerobot/src/lerobot/record.py:372
action_values = predict_action(
    observation=observation_frame,   # 观测帧(包含图像和关节)
    policy=policy,                   # 预训练策略(如ACT)
    device=get_safe_torch_device(policy.config.device),  # CPU/CUDA
    preprocessor=preprocessor,       # 输入预处理
    postprocessor=postprocessor,    # 输出后处理
    use_amp=policy.config.use_amp,  # 混合精度
    task=single_task,               # 任务描述
    robot_type=robot.robot_type,    # "so101_follower"
)
# action_values: torch.Tensor [num_actions]

# 转换为字典格式
action_names = dataset.features["action"]["names"]
act_processed_policy = {
    f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
}
# act_processed_policy: {"shoulder_pan.pos": 0.7, ...}
```

**路径B: TELEOP 示教读取**
```python
# lerobot/src/lerobot/record.py:392
act = teleop.get_action()  # → SO101Leader.get_action()
# act: {"shoulder_pan.pos": 0.5, ...} (示教手当前角度)

act_processed_teleop = teleop_action_processor((act, obs))
# teleop_action_processor可能做: 坐标系变换、限幅等
```

#### 3. 动作阶段 (Action)

```python
# lerobot/src/lerobot/record.py:430-437
# Step 1: 动作后处理
if policy is not None:
    action_values = act_processed_policy
    robot_action_to_send = robot_action_processor((act_processed_policy, obs))
else:
    action_values = act_processed_teleop
    robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
# robot_action_to_send: {"shoulder_pan.pos": 0.7, ...}

# Step 2: 发送给机器人
_sent_action = robot.send_action(robot_action_to_send)
# → SO101Follower.send_action()
# 底层: bus.sync_write("Goal_Position", goal_pos)

# Step 3: 写入数据集
if dataset is not None:
    action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")
    frame = {**observation_frame, **action_frame, "task": single_task}
    dataset.add_frame(frame)  # 写入episode buffer
```

#### 4. 等待阶段 (Wait)

```python
# lerobot/src/lerobot/record.py:468
dt_s = time.perf_counter() - start_loop_t  # 本帧已耗时
busy_wait(1 / fps - dt_s)                  # 等待达到目标帧间隔
```

---

## 第 2 层：robots/utils.py — 工厂函数

**文件：** `lerobot/src/lerobot/robots/utils.py`

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
```

根据 `config.type` 字符串路由到对应的机器人类：

| `config.type` | 对应类 | 说明 |
|--------------|--------|------|
| `"so100_follower"` | `SO100Follower` | 单臂SO100 |
| `"so101_follower"` | `SO101Follower` | 单臂SO101 |
| `"bi_so100_follower"` | `BiSO100Follower` | 双臂SO100 |
| `"bi_so101_follower"` | `BiSO101Follower` | 双臂SO101 |
| `"koch_follower"` | `KochFollower` | Koch单臂 |
| `"lekiwi"` | `LeKiwi` | 移动抓取机器人 |

---

## 第 3 层：SO101Follower — 机器人实现

**文件：** `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`

### 初始化 `__init__()`

```python
# so101_follower.py:45
def __init__(self, config: SO101FollowerConfig):
    super().__init__(config)
    self.config = config

    # 归一化模式: 身体关节用-100~100度,夹爪用0~100
    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

    # 创建Feetech舵机总线
    self.bus = FeetechMotorsBus(
        port=self.config.port,          # 如 "/dev/ttyACM0"
        motors={
            # 舵机名称: Motor(ID, 型号, 归一化模式)
            "shoulder_pan":  Motor(1, "sts3215", norm_mode_body),   # 关节1: 水平旋转
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),   # 关节2: 肩部抬起
            "elbow_flex":    Motor(3, "sts3215", norm_mode_body),   # 关节3: 肘部弯曲
            "wrist_flex":    Motor(4, "sts3215", norm_mode_body),   # 关节4: 腕部弯曲
            "wrist_roll":    Motor(5, "sts3215", norm_mode_body),   # 关节5: 腕部旋转
            "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # 关节6: 夹爪
        },
        calibration=self.calibration,   # 标定数据
    )

    # 创建相机实例(如handeye, fixed)
    self.cameras = make_cameras_from_configs(config.cameras)
```

### `connect()` 建立连接

```python
# so101_follower.py:85
def connect(self, calibrate: bool = True) -> None:
    if self.is_connected:
        raise DeviceAlreadyConnectedError(...)

    # Step 1: 连接舵机总线(打开串口)
    self.bus.connect()

    # Step 2: 校准(如需要)
    if not self.is_calibrated and calibrate:
        if not self.calibration:
            logger.info("No calibration file found, running calibration")
        self.calibrate()  # 运行交互式校准流程

    # Step 3: 连接所有相机
    for cam in self.cameras.values():
        cam.connect()

    # Step 4: 写入运行时配置
    self.configure()
    logger.info(f"{self} connected.")
```

### `get_observation()` 读取状态

```python
# so101_follower.py:249
def get_observation(self) -> dict[str, Any]:
    """
    获取机器人当前状态观测

    Returns:
        dict[str, Any]: 包含以下键的观测字典:
            - "{motor_name}.pos": float, 各关节当前位置(归一化值)
            - "{camera_key}": np.ndarray, 各相机最近一帧图像(H,W,3)

    底层执行流程:
        1. bus.sync_read("Present_Position") -> 通过串口发送Sync Read指令包
        2. cam.async_read() -> 从后台线程缓存获取最近相机帧
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
    # 最终返回: {"shoulder_pan.pos": 0.5, "handeye": array, "fixed": array, ...}
```

### `send_action()` 发送目标位置

```python
# so101_follower.py:272
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """
    命令机械臂移动到目标关节配置

    Args:
        action: 动作字典,键如"{motor_name}.pos",值如目标位置(归一化值)

    Returns:
        dict[str, Any]: 实际发送给舵机的动作(可能经过安全限幅处理)

    底层执行流程:
        1. 提取目标位置
        2. 安全限幅(可选)
        3. bus.sync_write("Goal_Position", goal_pos) -> 串口发送Sync Write包
    """
    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # Step 1: 从action提取目标位置
    # action: {"shoulder_pan.pos": 0.7, ...}
    # goal_pos: {"shoulder_pan": 0.7, ...}
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

    # Step 2: 安全限幅(可选)
    # 若设置了max_relative_target,限制单帧最大位移
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
def __init__(self, port, motors, calibration=None, protocol_version=0):
    super().__init__(port, motors, calibration)
    self.protocol_version = protocol_version
    self._assert_same_protocol()

    import scservo_sdk as scs

    # Step 1: 创建串口处理器
    self.port_handler = scs.PortHandler(self.port)
    # HACK: 修补官方SDK的超时计算bug
    self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
        self.port_handler, scs.PortHandler
    )

    # Step 2: 创建协议处理器(负责组包/解包)
    self.packet_handler = scs.PacketHandler(protocol_version)

    # Step 3: 创建同步读/写对象
    self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
    self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
```

### `_serialize_data()` — 整数转字节

```python
# feetech.py:69
def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    """将整数拆分为小端序字节列表

    用于将多字节寄存器值(如Goal_Position=2048)转换为字节列表,
    以便通过串口发送给舵机。

    Example:
        >>> _split_into_byte_chunks(2048, 2)
        [0x00, 0x08]  # 低字节在前,高字节在后
    """
    if length == 1:
        return [value]
    elif length == 2:
        return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    elif length == 4:
        return [...]
```

---

## 第 5 层：motors_bus.py — 舵机总线抽象层

**文件：** `lerobot/src/lerobot/motors/motors_bus.py`

### `MotorsBus` 核心属性

```python
class MotorsBus(abc.ABC):
    port: str                    # 串口路径,如 "/dev/ttyACM0"
    motors: dict[str, Motor]    # 舵机名 → Motor(id, model, norm_mode)
    calibration: dict[str, MotorCalibration]  # 标定数据

    # 由子类填充:
    port_handler: PortHandler    # 串口句柄(scservo_sdk)
    packet_handler: PacketHandler  # 协议处理(scservo_sdk)
    sync_reader: GroupSyncRead   # 同步读对象(scservo_sdk)
    sync_writer: GroupSyncWrite  # 同步写对象(scservo_sdk)
```

---

### `sync_read()` — 同步读取多个舵机

```python
# motors_bus.py:1053
def sync_read(
    self,
    data_name: str,                         # 寄存器名,如 "Present_Position"
    motors: str | list[str] | None = None,  # 要读的舵机,None=全部
    *,
    normalize: bool = True,                 # 是否归一化到物理单位
    num_retry: int = 0,                      # 失败重试次数
) -> dict[str, Value]:
    """从多个舵机同步读取同一寄存器

    执行流程:
        1. 查控制表: data_name → (addr, length)
        2. 解析电机名 → ID列表
        3. 调用_sync_read(addr, length, motor_ids)
        4. _decode_sign() 处理符号位
        5. _normalize() 原始字节 → 物理单位
        6. 返回 {舵机名: 值}
    """
```

### `_sync_read()` — 底层同步读

```python
# motors_bus.py:1102
def _sync_read(
    self,
    addr: int,              # 寄存器起始地址(如56=Present_Position)
    length: int,            # 读取字节数(如2)
    motor_ids: list[int],  # 舵机ID列表([1,2,3,4,5,6])
    *,
    num_retry: int = 0,
    raise_on_error: bool = True,
) -> tuple[dict[int, int], int]:

    # Step 1: 配置同步读
    self._setup_sync_reader(motor_ids, addr, length)

    # Step 2: 发送Sync Read指令包,等待所有舵机响应
    for n_try in range(1 + num_retry):
        comm = self.sync_reader.txRxPacket()
        if self._is_comm_success(comm):
            break

    # Step 3: 提取各舵机数据
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
    return values, comm
```

### `_setup_sync_reader()` — 配置同步读

```python
# motors_bus.py:1131
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
    # 配置同步读对象
    self.sync_reader.clearParam()          # 清空上一次的参数
    self.sync_reader.start_address = addr   # 寄存器起始地址(如56)
    self.sync_reader.data_length = length   # 数据长度(如2字节)
    for id_ in motor_ids:
        self.sync_reader.addParam(id_)     # 添加舵机ID
```

---

### `sync_write()` — 同步写入多个舵机

```python
# motors_bus.py:1153
def sync_write(
    self,
    data_name: str,                    # 寄存器名,如 "Goal_Position"
    values: Value | dict[str, Value],  # 单值或{motor: value}
    *,
    normalize: bool = True,           # 是否反归一化
    num_retry: int = 0,
) -> None:
    """向多个舵机同步写入同一寄存器(无应答模式)

    与write()不同,sync_write()不等待舵机返回状态包,
    因此可能丢失数据包,但速度更快,适用于遥操作场景。
    """
    ids_values = self._get_ids_values_dict(values)
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)

    # 物理单位 → 原始字节值
    if normalize and data_name in self.normalized_data:
        ids_values = self._unnormalize(ids_values)

    ids_values = self._encode_sign(data_name, ids_values)

    self._sync_write(addr, length, ids_values, ...)
```

### `_sync_write()` — 底层同步写

```python
# motors_bus.py:1195
def _sync_write(self, addr, length, ids_values, ...) -> int:
    # Step 1: 配置同步写对象
    self._setup_sync_writer(ids_values, addr, length)
    # → sync_writer.clearParam()
    # → sync_writer.start_address = 42 (Goal_Position)
    # → sync_writer.data_length = 2
    # → 对每个舵机: data = _serialize_data(value, 2)
    # → sync_writer.addParam(id_, data)

    # Step 2: 发送Sync Write广播包
    for n_try in range(1 + num_retry):
        comm = self.sync_writer.txPacket()  # → PortHandler.writePort()
        if self._is_comm_success(comm):
            break
```

---

## 第 6 层：scservo_sdk — 底层串口通信

**包位置：** `venv/lib/python3.13/site-packages/scservo_sdk/`

### 核心对象

#### `PortHandler(port: str)` — 串口管理

```python
# port_handler.py
port_handler = scs.PortHandler("/dev/ttyACM0")

port_handler.openPort()           # 打开串口(内部调用open()系统调用)
port_handler.closePort()          # 关闭串口
port_handler.setBaudRate(1000000) # 设置波特率(STS3215默认1Mbps)
port_handler.writePort(packet)    # 写字节到串口(→ write()系统调用)
port_handler.readPort(length)     # 从串口读N字节(→ read()系统调用)
port_handler.setPacketTimeout(t)  # 设置读超时
port_handler.isPacketTimeout()    # 检查是否超时
```

**波特率计算:**
```python
tx_time_per_byte = (1000.0 / baudrate) * 10.0
# 1Mbps: tx_time_per_byte = 10微秒/字节
# 1帧Sync Write(约20字节): ~200微秒
```

#### `PacketHandler(protocol_version)` — 协议处理

```python
# protocol_packet_handler.py
packet_handler = scs.PacketHandler(0)  # 0 = Feetech SCS协议

# 单舵机读(有应答)
result, data, error = packet_handler.read2ByteTxRx(port_handler, id=1, address=56)

# 单舵机写(有应答)
result, error = packet_handler.write2ByteTxRx(port_handler, id=1, address=42, value=2048)
```

#### `GroupSyncRead` — 批量同步读

```python
# group_sync_read.py
sync_reader = scs.GroupSyncRead(port_handler, packet_handler, 0, 0)

# 使用前配置
sync_reader.clearParam()
sync_reader.start_address = 56   # Present_Position寄存器地址
sync_reader.data_length = 2     # 读2字节

sync_reader.addParam(1)          # 添加舵机ID=1
sync_reader.addParam(2)
...
sync_reader.addParam(6)

# 发送Sync Read指令包,阻塞等待所有舵机响应
comm = sync_reader.txRxPacket()

# 提取各舵机数据
pos1 = sync_reader.getData(id=1, address=56, data_length=2)
```

#### `GroupSyncWrite` — 批量同步写

```python
# group_sync_write.py
sync_writer = scs.GroupSyncWrite(port_handler, packet_handler, 0, 0)

# 使用前配置
sync_writer.clearParam()
sync_writer.start_address = 42   # Goal_Position寄存器地址
sync_writer.data_length = 2      # 写2字节

# 为每个舵机添加目标值
sync_writer.addParam(id=1, data=[0x00, 0x08])  # 2048
sync_writer.addParam(id=2, data=[0x10, 0x07])  # 1808
...

# 发送Sync Write指令包(广播,无需等待响应)
comm = sync_writer.txPacket()
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

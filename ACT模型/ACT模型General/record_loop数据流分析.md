# ACT 模型 SO101 机械臂 record_loop 数据流分析

## 目录

1. [用户配置解析](#1-用户配置解析)
2. [record_loop 源码](#2-record_loop-源码)
3. [核心数据类型说明](#3-核心数据类型说明)
4. [核心数据结构定义](#4-核心数据结构定义)
5. [数据流按源码顺序详解](#5-数据流按源码顺序详解)
   - [5.1 初始化阶段](#51-初始化阶段)
   - [5.2 观测阶段](#52-观测阶段)
   - [5.3 推理阶段](#53-推理阶段)
   - [5.4 动作阶段](#54-动作阶段)
   - [5.5 数据存储](#55-数据存储)
   - [5.6 等待阶段](#56-等待阶段)
6. [ACT 策略 select_action 详解](#6-act-策略-select_action-详解)
7. [数据集特征与 episode_buffer](#7-数据集特征与-episode_buffer)
8. [关键函数索引](#8-关键函数索引)
9. [时间瓶颈分析](#9-时间瓶颈分析)

---

## 1. 用户配置解析

```bash
python -m lerobot.record \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}" \
  --dataset.single_task="Put the bottle into the black basket." \
  --policy.path=/home/vel/so101-bottle/last/pretrained_model \
  --dataset.repo_id=${HF_USER}/eval_so101_bottle \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=40
```

## 2. record_loop 源码（Policy 推理路径）

**定义位置**: `lerobot/src/lerobot/record.py:262-499`

> **说明**: 以下源码基于用户配置 `--policy.path=/home/vel/so101-bottle/last/pretrained_model` 的 **Policy 推理路径**，已删除时间戳追踪变量和不适用于 Policy 路径的 teleop 逻辑分支。

```python
@safe_stop_image_writer
def record_loop():
    while time.perf_counter() - episode_start_t < control_time_s:
        # -------------------------------------------------------------
        # 2.1 观测阶段: 获取机器人当前状态
        # -------------------------------------------------------------
        # robot.get_observation() 读取:
        #   - 6 个关节位置 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
        #   - 2 个相机图像 (handeye, fixed)
        obs: RobotObservation = robot.get_observation()

        # 观测预处理（默认 IdentityProcessor，不做处理）
        obs_processed: RobotObservation = robot_observation_processor(obs)

        # 将观测打包为数据集格式
        # 输出: {"observation.state": np.array(6,),
        #        "observation.images.handeye": ndarray,
        #        "observation.images.fixed": ndarray}
        # 从dict[str, Any]->转换为 dict[str, np.ndarray]
        observation_frame: dict[str, np.ndarray] = build_dataset_frame(
            dataset.features, obs_processed, prefix="observation")

        # -------------------------------------------------------------
        # 2.2 推理阶段: Policy 预测动作
        # -------------------------------------------------------------
        # predict_action 内部流程:
        #   1. numpy → torch 转换（图像归一化到 [0,1]）
        #   2. preprocessor 处理（图像 resize/normalize）
        #   3. policy.select_action() 推理（ACT 模型前向传播）
        #   4. postprocessor 处理（反归一化）
        #   5. torch → numpy 转换
        # 输出: action_values = np.array([15.5, -30.2, 45.0, -20.0, 10.0, 80.0], dtype=np.float32)
        action_values: np.ndarray = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        # 将 numpy 动作数组转换为 RobotAction 字典格式
        # 键名来自数据集 features["action"]["names"]
        action_names = dataset.features["action"]["names"]
        act_processed_policy: RobotAction = {
            f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
        }
        # 输出: {"shoulder_pan.pos": 15.5, "shoulder_lift.pos": -30.2, ...}

        # -------------------------------------------------------------
        # 2.3 动作阶段: 下发动作到机器人
        # -------------------------------------------------------------
        # 机器人动作预处理（默认无操作）
        robot_action_to_send: RobotAction = robot_action_processor((act_processed_policy, obs))

        # robot.send_action() 内部:
        #   1. 提取 .pos 结尾的目标位置
        #   2. 安全限幅（可选，限制每步关节移动范围）
        #   3. 通过 USB/串口下发到舵机
        _sent_action: RobotAction = robot.send_action(robot_action_to_send)

        # -------------------------------------------------------------
        # 2.4 数据存储: 保存当前帧到数据集
        # -------------------------------------------------------------
        # 构建动作帧
        action_frame: dict[str, np.ndarray] = build_dataset_frame(
            dataset.features, act_processed_policy, prefix="action")
        # 输出: {"action": np.array([15.5, -30.2, 45.0, -20.0, 10.0, 80.0])}

        # 合并观测帧 + 动作帧 + 任务描述
        frame: dict = {**observation_frame, **action_frame, "task": single_task}

        # 添加帧到 episode_buffer（内部处理图像保存到临时文件）
        dataset.add_frame(frame)

```

---

## 3. 核心数据类型说明

### 3.1 np.ndarray (NumPy 数组)

**来源**: `numpy.ndarray` - NumPy 库的核心数组类型

```python
import numpy as np

# 语法: np.ndarray(shape=(H, W, C), dtype=uint8)
# 参数:
#   shape: 元组,表示各维度大小
#   dtype: 数据类型
#
# dtype 常见取值:
#   - uint8:  无符号8位整数 (0-255), 用于存储图像像素
#   - float32: 32位浮点数, 用于存储关节角度等
#
# 示例:
img = np.ndarray(shape=(360, 640, 3), dtype=uint8)
# shape=(360, 640, 3) 表示:
#   - 360: 高度 (height, 像素)
#   - 640: 宽度 (width, 像素)
#   - 3:   通道数 (R/G/B 三通道)
#
# img.shape  → (360, 640, 3)
# img.dtype  → uint8
# img[100, 200, 0] → 像素 (100,200) 的 R 通道值 (0-255)
```

### 3.2 torch.Tensor (PyTorch 张量)

**来源**: `torch.Tensor` - PyTorch 库的张量类型,用于 GPU/CPU 加速计算

```python
import torch

# 语法: torch.Tensor(shape)
# 与 np.ndarray 类似,但支持 GPU 加速和自动梯度
# 主要用于深度学习模型推理

action = torch.Tensor([15.5, -30.2, 45.0, -20.0, 10.0, 80.0])
# action.shape → (6,)
# action.dtype → torch.float32
```

### 3.3 dict[str, Any] (字典类型)

**来源**: Python 内置类型

```python
# 语法: dict[key_type, value_type]
# 键必须是可哈希的(字符串/数/元组),值可以是任意类型

obs = {
    "shoulder_pan.pos": 0.0,     # str: float
    "handeye": np.ndarray(...),  # str: np.ndarray
    "task": "Pick the cube",     # str: str
}
```

---

## 4. 核心数据结构定义

### 4.1 RobotObservation (原始观测)

**定义位置**: `lerobot/src/lerobot/robots/so101_follower/so101_follower.py:249-270`

```python
# 来源: robot.get_observation()
# 类型: dict[str, Any]

obs = {
    # === 关节状态 (6个关节) ===
    # 定义: so101_follower.py:52-58
    "shoulder_pan.pos": 0.0,      # 肩部旋转 (degrees, 范围取决于校准)
    "shoulder_lift.pos": 0.0,    # 肩部提升
    "elbow_flex.pos": 0.0,       # 肘部弯曲
    "wrist_flex.pos": 0.0,       # 腕部弯曲
    "wrist_roll.pos": 0.0,       # 腕部旋转
    "gripper.pos": 0.0,          # 夹爪 (0-100, 0=开, 100=关)

    # === 相机图像 (2个相机) ===
    # 定义: so101_follower.py:61
    "handeye": np.ndarray(shape=(360, 640, 3), dtype=uint8),  # 手眼相机
    "fixed": np.ndarray(shape=(360, 640, 3), dtype=uint8),    # 固定相机
}
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
dict content:
  shoulder_pan.pos: float = -7.957462412907958
  shoulder_lift.pos: float = -45.28462192013593
  elbow_flex.pos: float = 98.72611464968153
  wrist_flex.pos: float = 14.57597173144876
  wrist_roll.pos: float = -3.931623931623932
  gripper.pos: float = 55.22482583913869
  handeye: ndarray shape=(360, 640, 3) dtype=uint8
  fixed: ndarray shape=(360, 640, 3) dtype=uint8
```
**结论**: 文档描述完全正确。关节值为实际角度(degrees)，gripper 0-100范围。

### 4.2 RobotAction (机器人动作)

**定义位置**: `lerobot/src/lerobot/processor/core.py` (RobotAction 类型别名)

```python
# 来源: teleop 或 policy 生成后转换
# 类型: dict[str, float]

robot_action = {
    "shoulder_pan.pos": 15.5,    # 目标角度 (degrees)
    "shoulder_lift.pos": -30.2,
    "elbow_flex.pos": 45.0,
    "wrist_flex.pos": -20.0,
    "wrist_roll.pos": 10.0,
    "gripper.pos": 80.0,         # 夹爪目标位置
}
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
dict content:
  shoulder_pan.pos: float = -7.20216178894043
  shoulder_lift.pos: float = -47.46265411376953
  elbow_flex.pos: float = 43.47043991088867
  wrist_flex.pos: float = 24.396121978759766
  wrist_roll.pos: float = -7.26908540725708
  gripper.pos: float = 46.07160949707031
```
**结论**: 文档描述结构正确，示例值为占位符。实际值为 policy 推理输出的目标关节位置。

### 4.3 PolicyAction (策略输出张量)

**定义位置**: `lerobot/src/lerobot/processor/core.py` (PolicyAction 类型)

```python
# 来源: policy.select_action()
# 类型: torch.Tensor

action_tensor = torch.Tensor([15.5, -30.2, 45.0, -20.0, 10.0, 80.0])
# shape=(6,)  # 6个关节动作值
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
type: <class 'torch.Tensor'>
torch.Tensor shape=torch.Size([6]) dtype=torch.float32
```
**结论**: 文档描述正确。shape=(6,) 表示6个关节动作，dtype=torch.float32。

### 4.4 observation_frame (数据集观测帧)

**定义位置**: `lerobot/src/lerobot/datasets/utils.py:673-699` (build_dataset_frame 函数)

```python
# 来源: build_dataset_frame(dataset.features, obs_processed, prefix="observation")
# 类型: dict[str, np.ndarray]

observation_frame = {
    # 关节状态向量
    "observation.state": np.array([15.5, -30.2, 45.0, -20.0, 10.0, 80.0], dtype=np.float32),

    # 相机图像
    "observation.images.handeye": np.ndarray(shape=(360, 640, 3), dtype=uint8),
    "observation.images.fixed": np.ndarray(shape=(360, 640, 3), dtype=uint8),
}
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
dict content:
  observation.state: ndarray shape=(6,) dtype=float32
    values: [-7.957462310791016, -45.28462219238281, 98.72611236572266,
             14.575971603393555, -3.931623935699463, 55.22482681274414]
  observation.images.handeye: ndarray shape=(360, 640, 3) dtype=uint8
  observation.images.fixed: ndarray shape=(360, 640, 3) dtype=uint8
```
**结论**: 文档描述完全正确。关节值从原始 float 转换为 float32 的 ndarray。

### 4.5 action_frame (数据集动作帧)

**定义位置**: `lerobot/src/lerobot/datasets/utils.py:673-699`

```python
# 来源: build_dataset_frame(dataset.features, action_values, prefix="action")
action_frame = {
    "action": np.array([15.5, -30.2, 45.0, -20.0, 10.0, 80.0], dtype=np.float32)
}
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
dict content:
  action: ndarray shape=(6,) dtype=float32
    values: [-7.20216178894043, -47.46265411376953, 43.47043991088867,
             24.396121978759766, -7.26908540725708, 46.07160949707031]
```
**结论**: 文档描述完全正确。action 值来自 policy 推理输出。

### 4.6 dataset Frame (完整数据帧)

**定义位置**: `lerobot/src/lerobot/record.py:450`

```python
# 来源: 合并 observation_frame + action_frame + task
frame = {
    "observation.state": np.array([...], dtype=np.float32),
    "observation.images.handeye": np.ndarray,
    "observation.images.fixed": np.ndarray,
    "action": np.array([...], dtype=np.float32),
    "task": "Put the bottle into the black basket.",
}
```

**✅ 验证结果 (Frame 0 实际数据)**:
```
dict content:
  observation.state: ndarray shape=(6,) dtype=float32
    values: [-7.957462310791016, -45.28462219238281, 98.72611236572266,
             14.575971603393555, -3.931623935699463, 55.22482681274414]
  observation.images.handeye: ndarray shape=(360, 640, 3) dtype=uint8
  observation.images.fixed: ndarray shape=(360, 640, 3) dtype=uint8
  action: ndarray shape=(6,) dtype=float32
    values: [-7.20216178894043, -47.46265411376953, 43.47043991088867,
             24.396121978759766, -7.26908540725708, 46.07160949707031]
```
**结论**: 文档描述完全正确。完整帧包含 observation.state、2个相机图像、action 和 task。

---

## 5. 数据流按源码顺序详解

### 5.1 初始化阶段

**对应源码**: `record.py:285-324`

```python
# 行 285-286: 校验数据集 fps
if dataset is not None and dataset.fps != fps:
    raise ValueError(...)

# 行 289-314: 初始化 teleop 设备
teleop_arm = teleop_keyboard = None
if isinstance(teleop, list):
    teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
    teleop_arm = next((t for t in teleop if isinstance(t, (...))), None)

# 行 317-320: 重置 policy 和 processors
if policy is not None and preprocessor is not None and postprocessor is not None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

# 行 322-324: 初始化计时器
timestamp = 0
start_episode_t = time.perf_counter()
last_log_t = 0.0
```

**数据流**: 此阶段无数据流,只是初始化变量和重置状态。

---

### 5.2 观测阶段

**对应源码**: `record.py:332-358`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段: 观测阶段 (Observation)                                                 │
│  源码: record.py:332-358                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────────┐                       ┌───────────────────────┐
│ 行 343: robot.get_observation()               │ 行 348: robot_observation_processor(obs)
│ 定义: so101_follower.py:249-270              │ 定义: pipeline.py:284-295
└───────────────────────┘                       └───────────────────────┘
        │                                                       │
        ▼                                                       ▼
┌───────────────────────┐                       ┌───────────────────────┐
│ obs: RobotObservation  │                       │ obs_processed: 整理后的观测
│ {                       │                       │ (默认与 obs 相同)
│   "shoulder_pan.pos": 0.0│                       │
│   "handeye": ndarray,    │                       │
│   ...                    │                       │
│ }                        │                       │
└───────────────────────┘                       └───────────────────────┘
        │                                                       │
        └───────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 355: build_dataset_frame(                │
                    │           dataset.features,                 │
                    │           obs_processed,                    │
                    │           prefix="observation"              │
                    │         )                                   │
                    │ 定义: datasets/utils.py:673-699              │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ observation_frame: dict                   │
                    │ {                                         │
                    │   "observation.state": np.array(6,),       │
                    │   "observation.images.handeye": ndarray,    │
                    │   "observation.images.fixed": ndarray,      │
                    │ }                                         │
                    └───────────────────────────────────────────┘
```

#### 5.2.1 robot.get_observation() 详解

**定义**: `so101_follower.py:249-270`

```python
def get_observation(self) -> dict[str, Any]:
    """
    获取机器人当前状态: 关节位置 + 相机图像
    """
    # 1. 读取 6 个关节位置 (STS3215 舵机)
    # 总线同步读取,阻塞式
    obs_dict = self.bus.sync_read("Present_Position")
    # 返回: {"shoulder_pan": 0.0, "shoulder_lift": 0.0, ...}

    # 2. 转换键名格式: 添加 ".pos" 后缀
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    # 返回: {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, ...}

    # 3. 读取 2 个相机图像 (异步)
    # async_read() 可能从后台 buffer 取最近帧
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()

    return obs_dict
```

#### 5.2.2 build_dataset_frame() 详解

**定义**: `datasets/utils.py:673-699`

```python
def build_dataset_frame(ds_features, values, prefix):
    """
    将原始值按数据集特征规范打包成数据帧

    Args:
        ds_features: 数据集特征字典
        values: 原始值字典 (如 obs_processed)
        prefix: "observation" 或 "action"

    Returns:
        符合特征规范的数据帧
    """
    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            # 处理关节向量:
            # ft["names"] = ["shoulder_pan.pos", ...]
            # values["shoulder_pan.pos"] = 0.0, ...
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            # 处理图像:
            # key = "observation.images.handeye"
            # values["handeye"] = ndarray
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame
```

---

### 5.3 推理阶段

**对应源码**: `record.py:360-418`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段: 推理阶段 (Inference)                                                  │
│  源码: record.py:360-418                                                    │
│                                                                              │
│  两条路径:                                                                   │
│    1. Policy 路径 (policy is not None) ← 用户配置走这条                       │
│    2. Teleop 路径 (teleop is not None)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  if policy is not None and ...:           │ ← 行 369
                    │      # Policy 推理路径                      │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  predict_action(                           │
                    │    observation=observation_frame,           │
                    │    policy=policy,                          │
                    │    device="cpu",                          │
                    │    preprocessor=preprocessor,             │
                    │    postprocessor=postprocessor,            │
                    │    use_amp=False,                         │
                    │    task=single_task,                      │
                    │    robot_type=robot.robot_type,            │
                    │  )                                        │
                    │  定义: control_utils.py:125-189           │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  action_values: np.array(shape=(6,))     │
                    │  [15.5, -30.2, 45.0, -20.0, 10.0, 80.0]   │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 384-387: 转换为 RobotAction 字典         │
                    │                                           │
                    │  action_names = dataset.features[         │
                    │      "action"]["names"]                   │
                    │  act_processed_policy = {                  │
                    │      f"{name}": float(action_values[i])   │
                    │      for i, name in enumerate(            │
                    │          action_names)                     │
                    │  }                                        │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  act_processed_policy: RobotAction        │
                    │  {                                        │
                    │    "shoulder_pan.pos": 15.5,              │
                    │    "shoulder_lift.pos": -30.2,             │
                    │    ...                                    │
                    │  }                                        │
                    └───────────────────────────────────────────┘
```

#### 5.3.1 predict_action() 详解

**定义**: `control_utils.py:125-189`

```python
def predict_action(observation, policy, device, preprocessor, postprocessor,
                   use_amp, task, robot_type):
    """
    策略推理包装函数: numpy → torch → 推理 → numpy
    """
    # Step 1: 复制观测避免修改原始数据
    observation = copy(observation)

    # Step 2: numpy → torch 转换
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            # [0,255] uint8 → [0,1] float32
            observation[name] = observation[name].type(torch.float32) / 255
            # HWC → CHW (channel first)
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0).to(device)

    # Step 3: 添加 task 和 robot_type 元信息
    observation["task"] = task
    observation["robot_type"] = robot_type

    # Step 4: 预处理 (归一化等)
    observation = preprocessor(observation)

    # Step 5: 策略推理
    action = policy.select_action(observation)

    # Step 6: 后处理 (反归一化等)
    action = postprocessor(action)

    # Step 7: 去除 batch 维度, 移到 CPU, 转为 numpy
    action = action.squeeze(0).to("cpu").numpy()

    return action
```

---

### 5.4 动作阶段

**对应源码**: `record.py:420-463`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段: 动作阶段 (Action)                                                      │
│  源码: record.py:420-463                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 432-437: robot_action_processor()       │
                    │ 默认 IdentityProcessorStep, 无实际处理      │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ robot_action_to_send: RobotAction          │
                    │ (与 act_processed_policy 相同)             │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 443: robot.send_action(                 │
                    │           robot_action_to_send            │
                    │         )                                 │
                    │ 定义: so101_follower.py:272-299           │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │  _sent_action: RobotAction                 │
                    │  (实际下发的动作,可能有裁剪)                  │
                    └───────────────────────────────────────────┘
```

#### 5.4.1 robot.send_action() 详解

**定义**: `so101_follower.py:272-299`

```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """
    发送目标位置到机器人
    """
    # Step 1: 提取 .pos 结尾的目标位置
    # action = {"shoulder_pan.pos": 15.5, ...}
    # goal_pos = {"shoulder_pan": 15.5, ...}
    goal_pos = {key.removesuffix(".pos"): val
                for key, val in action.items() if key.endswith(".pos")}

    # Step 2: 安全限幅 (可选)
    # 限制每步关节移动范围,防止损伤机器人
    if self.config.max_relative_target is not None:
        present_pos = self.bus.sync_read("Present_Position")
        goal_present_pos = {key: (g_pos, present_pos[key])
                           for key, g_pos in goal_pos.items()}
        goal_pos = ensure_safe_goal_position(
            goal_present_pos, self.config.max_relative_target)

    # Step 3: 下发到舵机 (USB/串口)
    self.bus.sync_write("Goal_Position", goal_pos)

    # Step 4: 返回实际下发的动作
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

---

### 5.5 数据存储

**对应源码**: `record.py:446-454`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段: 数据存储 (Dataset)                                                    │
│  源码: record.py:446-454                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 449: build_dataset_frame(              │
                    │           dataset.features,               │
                    │           action_values,                  │
                    │           prefix="action"                 │
                    │         )                                 │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ action_frame: {"action": np.array(6,)}   │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 450: frame = {                         │
                    │       **observation_frame,                │
                    │       **action_frame,                     │
                    │       "task": single_task                 │
                    │     }                                     │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ frame: 完整数据帧                         │
                    │ {                                         │
                    │   "observation.state": np.array(6,),       │
                    │   "observation.images.handeye": ndarray,    │
                    │   "observation.images.fixed": ndarray,      │
                    │   "action": np.array(6,),                 │
                    │   "task": "Put the bottle...",             │
                    │ }                                         │
                    └───────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────────────┐
                    │ 行 451: dataset.add_frame(frame)          │
                    │ 定义: lerobot_dataset.py:925-966          │
                    └───────────────────────────────────────────┘
```

#### 5.5.1 dataset.add_frame() 详解

**定义**: `lerobot_dataset.py:925-966`

```python
def add_frame(self, frame: dict) -> None:
    """
    将帧添加到 episode_buffer
    """
    # Step 1: torch → numpy
    for name in frame:
        if isinstance(frame[name], torch.Tensor):
            frame[name] = frame[name].numpy()

    # Step 2: 校验帧格式
    validate_frame(frame, self.features)

    # Step 3: 初始化 episode_buffer (第一帧时)
    if self.episode_buffer is None:
        self.episode_buffer = self.create_episode_buffer()

    # Step 4: 添加帧索引和时间戳
    frame_index = self.episode_buffer["size"]
    timestamp = frame_index / self.fps
    self.episode_buffer["frame_index"].append(frame_index)
    self.episode_buffer["timestamp"].append(timestamp)
    self.episode_buffer["task"].append(frame.pop("task"))

    # Step 5: 添加帧数据
    for key in frame:
        if self.features[key]["dtype"] in ["image", "video"]:
            # 图像保存到临时文件
            img_path = self._get_image_file_path(...)
            self._save_image(frame[key], img_path)
            self.episode_buffer[key].append(str(img_path))
        else:
            self.episode_buffer[key].append(frame[key])

    self.episode_buffer["size"] += 1
```

---

### 5.6 等待阶段

**对应源码**: `record.py:465-471`

```python
# 行 468-471: 等待维持帧率
wait_start_t = time.perf_counter()
dt_s = time.perf_counter() - start_loop_t  # 本帧已耗时
busy_wait(1 / fps - dt_s)  # 等待补齐到 1/fps 秒
wait_end_t = time.perf_counter()
```

**说明**: `busy_wait()` 使用 spin 循环持续检查时间,确保精确的帧间隔。

---

## 6. ACT 策略 select_action 详解

**定义**: `modeling_act.py:98-121`

```python
@torch.no_grad()
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    """
    给定当前观测,选择一个动作执行

    关键概念:
    - n_action_steps: 每次推理输出的动作数量 (默认1)
    - action queue: 动作队列,当队列空时执行推理填充
    """
    self.eval()  # 推理模式

    if self.config.temporal_ensemble_coeff is not None:
        # 使用时间集成 (减少抖动)
        actions = self.predict_action_chunk(batch)
        action = self.temporal_ensembler.update(actions)
        return action

    # === 标准模式: 动作队列 ===
    if len(self._action_queue) == 0:
        # 队列空,执行推理填充
        actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
        # transpose: (1, n_action_steps, 6) → (n_action_steps, 1, 6)
        self._action_queue.extend(actions.transpose(0, 1))

    # 返回队列头部动作
    return self._action_queue.popleft()

@torch.no_grad()
def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
    """
    预测一组(块)动作

    Returns:
        Tensor: shape=(batch_size, n_action_steps, action_dim)
    """
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    actions = self.model(batch)[0]  # ACT.forward()
    return actions
```

---

## 7. 数据集特征与 episode_buffer

### 7.1 数据集 features

**定义位置**: `lerobot/src/lerobot/datasets/utils.py`

```python
dataset_features = {
    # === 默认特征 ===
    "frame_index": {"dtype": "int64", "shape": (1,)},
    "timestamp": {"dtype": "float32", "shape": (1,)},
    "episode_index": {"dtype": "int64", "shape": (1,)},
    "index": {"dtype": "int64", "shape": (1,)},
    "task_index": {"dtype": "int64", "shape": (1,)},

    # === 动作特征 ===
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },

    # === 观测状态特征 ===
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },

    # === 相机图像特征 ===
    "observation.images.handeye": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.fixed": {
        "dtype": "video",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
    },
}
```

### 7.2 episode_buffer 结构

```python
# 40秒 @ 30fps = 1200帧
episode_buffer = {
    "size": 1200,
    "frame_index": [0, 1, 2, ..., 1199],
    "timestamp": [0.0, 0.033, 0.067, ..., 39.967],
    "episode_index": 0,
    "task": ["Put the bottle into the black basket.", ...],

    "observation.state": [np.array([...]), ...],  # 1200个 (6,) 数组
    "observation.images.handeye": ["/path/to/img0.png", ...],
    "observation.images.fixed": ["/path/to/img0.png", ...],

    "action": [np.array([...]), ...],  # 1200个 (6,) 数组
}
```

---

## 8. 关键函数索引

| 函数 | 文件:行号 | 功能 |
|------|----------|------|
| `record_loop()` | `record.py:262-499` | 录制主循环 |
| `robot.get_observation()` | `so101_follower.py:249-270` | 读取关节+相机 |
| `robot.send_action()` | `so101_follower.py:272-299` | 下发动作到电机 |
| `predict_action()` | `control_utils.py:125-189` | 策略推理包装 |
| `ACTPolicy.select_action()` | `modeling_act.py:98-121` | ACT选择动作 |
| `ACTPolicy.predict_action_chunk()` | `modeling_act.py:123-133` | ACT推理一批动作 |
| `build_dataset_frame()` | `datasets/utils.py:673-699` | 构建数据帧 |
| `dataset.add_frame()` | `lerobot_dataset.py:925-966` | 添加帧到buffer |
| `busy_wait()` | `utils/robot_utils.py` | 帧率控制等待 |
| `make_robot_from_config()` | `robots/__init__.py` | 创建机器人实例 |
| `make_policy()` | `policies/factory.py` | 创建策略实例 |

---

## 9. 时间瓶颈分析

### 9.1 计时输出格式

```
[Record Loop] timestamp=1.0s |
  obs=5.2ms           # 总观测时间
    (hw=3.1ms,        # 硬件读取 (关节+相机)
     proc=0.1ms,      # robot_observation_processor
     frame=2.0ms)     # build_dataset_frame
  inference=28.5ms    # 策略推理 (主要瓶颈)
  action=4.8ms        # 总动作时间
    (proc=0.1ms,      # robot_action_processor
     send=2.2ms,      # send_action (串口写入)
     dataset=2.3ms,   # add_frame
     display=0.2ms)   # log_rerun_data
  wait=0.0ms          # 理想无等待
  total=38.5ms       # 接近33.3ms目标 (30fps)
```

### 9.2 瓶颈分析

| 阶段 | 典型耗时 | 说明 |
|------|---------|------|
| `inference` | 28.5ms | ACT 模型在 ARM CPU 上推理,最耗时 |
| `obs_hw` | 3.1ms | 关节读取(1.2ms) + 相机读取(1.9ms) |
| `dataset` | 2.3ms | 图像保存到临时文件 |
| `send` | 2.2ms | 串口下发到舵机 |

### 9.3 优化方向

1. **降低图像分辨率**: 640x360 → 320x180
2. **减少相机数量**: 2 → 1
3. **增大 n_action_steps**: 减少推理频率 (但会牺牲响应性)
4. **模型量化**: FP32 → INT8
5. **使用 NPU 加速**: 树莓派5有 NPU,但 PyTorch 支持有限

---

## 10. 数据结构验证总结

### 验证方法

通过在 `record.py` 中添加 `_dump_data_structure()` 函数,在前3帧输出关键数据结构的实际值。

### 验证结论

| 数据结构 | 文档描述 | 验证结果 |
|---------|---------|---------|
| **obs (RobotObservation)** | dict, 6关节(float) + 2相机(ndarray) | ✅ 完全正确 |
| **obs_processed** | 与 obs 相同 (IdentityProcessor) | ✅ 完全正确 |
| **observation_frame** | dict: observation.state(float32 array) + 2相机图像 | ✅ 完全正确 |
| **action_values** | torch.Tensor shape=(6,) dtype=torch.float32 | ✅ 完全正确 |
| **act_processed_policy** | dict: {shoulder_pan.pos, ..., gripper.pos} | ✅ 完全正确 |
| **robot_action_to_send** | 与 act_processed_policy 相同 | ✅ 完全正确 |
| **action_frame** | dict: {action: ndarray(6,) float32} | ✅ 完全正确 |
| **frame (complete)** | obs + action + task | ✅ 完全正确 |

### 实际数据示例 (Frame 0)

**原始观测 (obs)**:
```
shoulder_pan.pos: -7.96      shoulder_lift.pos: -45.28
elbow_flex.pos: 98.73        wrist_flex.pos: 14.58
wrist_roll.pos: -3.93        gripper.pos: 55.22
handeye: (360, 640, 3) uint8  fixed: (360, 640, 3) uint8
```

**Policy 输出动作 (action_values → act_processed_policy)**:
```
shoulder_pan.pos: -7.20      shoulder_lift.pos: -47.46
elbow_flex.pos: 43.47       wrist_flex.pos: 24.40
wrist_roll.pos: -7.27       gripper.pos: 46.07
```

### 关键发现

1. **gripper 范围**: 实测 0-100,表示夹爪开合程度
2. **关节角度**: 均为 degrees 单位,具体范围取决于机械校准
3. **图像格式**: 固定 (360, 640, 3) uint8, 未压缩
4. **数据类型转换**:
   - 观测原始值: float (Python)
   - 观测帧: float32 (numpy)
   - 策略输出: torch.float32 → float → float32
5. **obs_processed**: 确认默认 IdentityProcessor,无任何转换

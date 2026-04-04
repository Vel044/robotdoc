# SO101 Follower 读取 ACM0 数据完整流程

**文档位置**: `workspace/docs/SO101_Follower_读取ACM0数据完整流程.md`

---

## 📋 概述

当 `record.py` 调用 `robot.get_observation()` 时，会触发一系列调用链，最终通过串口 `/dev/ttyACM0` 读取所有舵机的当前位置。

---

## 🔗 完整调用链

```
record.py
    └── robot.get_observation()
            └── SO101Follower.get_observation()
                    └── self.bus.sync_read("Present_Position")
                            └── FeetechMotorsBus.sync_read()
                                    └── _sync_read(addr, length, ids)
                                            └── scservo_sdk.GroupSyncRead.txRxPacket()
                                                    └── 串口通信 (/dev/ttyACM0)
```

---

## 📊 详细流程分解

### 1️⃣ record.py 调用入口

**源码位置**: `lerobot/src/lerobot/record.py` (第 331 行)

```python
# record.py (第 331 行)
obs = robot.get_observation()
```

这是数据采集的主循环，每次循环都会调用 `get_observation()` 获取当前观测。

---

### 2️⃣ SO101Follower.get_observation()

**源码位置**: `lerobot/src/lerobot/robots/so101_follower/so101_follower.py` (第 186-205 行)

```python
# so101_follower.py (第 186-205 行)
def get_observation(self) -> dict[str, Any]:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # 【核心】读取机械臂位置
    start = time.perf_counter()
    obs_dict = self.bus.sync_read("Present_Position")  # ← 这里！
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    dt_ms = (time.perf_counter() - start) * 1e3
    logger.debug(f"{self} read state: {dt_ms:.1f}ms")

    # 读取摄像头图像
    for cam_key, cam in self.cameras.items():
        start = time.perf_counter()
        obs_dict[cam_key] = cam.async_read()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

    return obs_dict
```

**返回数据格式**:
```python
{
    "shoulder_pan.pos": 2048,    # 肩部旋转位置
    "shoulder_lift.pos": 1500,   # 肩部抬升位置
    "elbow_flex.pos": 1800,      # 肘部弯曲位置
    "wrist_flex.pos": 1200,      # 手腕弯曲位置
    "wrist_roll.pos": 1000,      # 手腕旋转位置
    "gripper.pos": 500,          # 夹爪位置
    "camera_name": image_array,  # 摄像头图像
}
```

---

### 3️⃣ FeetechMotorsBus.sync_read()

**源码位置**: `lerobot/src/lerobot/motors/motors_bus.py` (第 1053-1091 行)

```python
# motors_bus.py (第 1053-1091 行)
def sync_read(
    self,
    data_name: str,           # "Present_Position"
    motors: str | list[str] | None = None,
    *,
    normalize: bool = True,
    num_retry: int = 0,
) -> dict[str, Value]:
    """读取多个舵机的同一个寄存器"""
    
    # 1. 获取要读取的舵机列表
    names = self._get_motors_list(motors)  # ["shoulder_pan", "shoulder_lift", ...]
    ids = [self.motors[motor].id for motor in names]  # [1, 2, 3, 4, 5, 6]
    models = [self.motors[motor].model for motor in names]  # ["sts3215", ...]
    
    # 2. 查找寄存器地址和长度
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)
    # addr = 56, length = 2 (Present_Position 地址是 56，长度是 2 字节)
    
    # 3. 执行同步读取
    ids_values, _ = self._sync_read(
        addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=err_msg
    )
    
    # 4. 数据解码和归一化
    ids_values = self._decode_sign(data_name, ids_values)
    if normalize and data_name in self.normalized_data:
        ids_values = self._normalize(ids_values)
    
    # 5. 返回 {舵机名: 值} 字典
    return {self._id_to_name(id_): value for id_, value in ids_values.items()}
```

---

### 4️⃣ _sync_read() - 底层实现

**源码位置**: `lerobot/src/lerobot/motors/motors_bus.py` (第 1093-1111 行)

```python
# motors_bus.py (第 1093-1111 行)
def _sync_read(
    self,
    addr: int,          # 56 (Present_Position 地址)
    length: int,        # 2 (字节数)
    motor_ids: list[int],  # [1, 2, 3, 4, 5, 6]
    *,
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> tuple[dict[int, int], int]:
    
    # 1. 设置同步读取器
    self._setup_sync_reader(motor_ids, addr, length)
    
    # 2. 发送并接收数据包（重试机制）
    for n_try in range(1 + num_retry):
        comm = self.sync_reader.txRxPacket()  # ← 核心！发送请求并接收响应
        if self._is_comm_success(comm):
            break
    
    # 3. 从响应中提取每个舵机的数据
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
    return values, comm
```

---

### 5️⃣ _setup_sync_reader() - 配置同步读取器

**源码位置**: `lerobot/src/lerobot/motors/motors_bus.py` (第 1143-1148 行)

```python
# motors_bus.py (第 1143-1148 行)
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
    # 清空之前的参数
    self.sync_reader.clearParam()
    
    # 设置读取地址和长度
    self.sync_reader.start_address = addr   # 56
    self.sync_reader.data_length = length   # 2
    
    # 添加每个舵机ID到读取列表
    for id_ in motor_ids:
        self.sync_reader.addParam(id_)  # 添加 ID 1, 2, 3, 4, 5, 6
```

---

### 6️⃣ scservo_sdk.GroupSyncRead.txRxPacket()

**源码位置**: `workspace/scservo_sdk_src/group_sync_read.py` (第 66-70 行)

**这是最底层的串口通信！**

```python
# group_sync_read.py (第 66-70 行)
def txRxPacket(self):
    result = self.txPacket()
    if result != COMM_SUCCESS:
        return result
    return self.rxPacket()
```

**在 FeetechMotorsBus.__init__() 中初始化**:

**源码位置**: `lerobot/src/lerobot/motors/feetech/feetech.py` (第 130-135 行)

```python
# feetech.py (第 130-135 行)
import scservo_sdk as scs
self.port_handler = scs.PortHandler(self.port)  # self.port = "/dev/ttyACM0"
self.packet_handler = scs.PacketHandler(protocol_version)
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
```

**txPacket() 发送请求** (group_sync_read.py 第 44-50 行):

```python
# group_sync_read.py (第 44-50 行)
def txPacket(self):
    if len(self.data_dict.keys()) == 0:
        return COMM_NOT_AVAILABLE

    if self.is_param_changed is True or not self.param:
        self.makeParam()

    return self.ph.syncReadTx(self.port, self.start_address, self.data_length, 
                              self.param, len(self.data_dict.keys()) * 1)
```

**rxPacket() 接收响应** (group_sync_read.py 第 52-64 行):

```python
# group_sync_read.py (第 52-64 行)
def rxPacket(self):
    self.last_result = False
    result = COMM_RX_FAIL

    if len(self.data_dict.keys()) == 0:
        return COMM_NOT_AVAILABLE

    for scs_id in self.data_dict:
        self.data_dict[scs_id], result, _ = self.ph.readRx(
            self.port, scs_id, self.data_length)
        if result != COMM_SUCCESS:
            return result

    if result == COMM_SUCCESS:
        self.last_result = True

    return result
```

**完整通信流程**:

```
1. txPacket() - 发送请求包
   ┌─────────────────────────────────────────────────────────────┐
   │  构建数据包:                                                  │
   │  [0xFF, 0xFF]              - 帧头                            │
   │  [0xFE]                    - 广播ID (254)                    │
   │  [length]                  - 数据长度                         │
   │  [INST_SYNC_READ]          - 指令码 (130)                    │
   │  [start_addr]              - 起始地址 (56)                   │
   │  [data_length]             - 数据长度 (2)                    │
   │  [checksum]                - 校验和                          │
   └─────────────────────────────────────────────────────────────┘
   
2. 通过串口发送: port_handler.writePort(packet)
   → /dev/ttyACM0

3. rxPacket() - 接收响应包
   ┌─────────────────────────────────────────────────────────────┐
   │  每个舵机返回:                                                │
   │  [0xFF, 0xFF]              - 帧头                            │
   │  [servo_id]                - 舵机ID (1-6)                    │
   │  [length]                  - 数据长度                         │
   │  [error]                   - 错误码                          │
   │  [Present_Position_L]      - 位置低字节                      │
   │  [Present_Position_H]      - 位置高字节                      │
   │  [checksum]                - 校验和                          │
   └─────────────────────────────────────────────────────────────┘
   
4. 通过串口接收: port_handler.readPort()
   ← /dev/ttyACM0

5. 解析响应，提取数据
   Present_Position = Present_Position_H << 8 | Present_Position_L
```

---

## 📝 Present_Position 寄存器详情

**控制表定义** (`tables.py`):

**源码位置**: `lerobot/src/lerobot/motors/feetech/tables.py` (第 56 行)

```python
# tables.py (第 56 行)
STS_SMS_SERIES_CONTROL_TABLE = {
    ...
    "Present_Position": (56, 2),  # 地址 56, 长度 2 字节
    ...
}
```

**寄存器属性**:
| 属性 | 值 |
|------|-----|
| 地址 | 56 (0x38) |
| 长度 | 2 字节 |
| 类型 | 只读 |
| 范围 | 0 ~ 4095 (12-bit) |
| 单位 | 脉冲数 (需归一化转换为角度) |

---

## 🔄 数据归一化流程

### 原始值 → 归一化值

**源码位置**: `lerobot/src/lerobot/motors/motors_bus.py` (第 974-1000 行)

```python
# motors_bus.py (第 974-1000 行)
def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
    """将原始脉冲值转换为归一化值"""
    normalized = {}
    for id_, value in ids_values.items():
        motor_name = self._id_to_name(id_)
        calibration = self.calibration[motor_name]
        
        # 归一化到 [-100, 100] 或 [0, 100]
        range_min = calibration.range_min
        range_max = calibration.range_max
        
        # 线性映射
        normalized_value = (value - range_min) / (range_max - range_min) * 200 - 100
        normalized[id_] = normalized_value
    
    return normalized
```

**归一化模式** (motors_bus.py 第 48-51 行):
```python
# motors_bus.py (第 48-51 行)
class MotorNormMode(str, Enum):
    RANGE_M100_100 = "range_m100_100"  # 映射到 [-100, 100]
    RANGE_0_100 = "range_0_100"        # 映射到 [0, 100]
    DEGREES = "degrees"                # 映射到角度值
```

---

## ⏱️ 时序分析

### 单次读取时间

```
┌────────────────────────────────────────────────────────────┐
│ get_observation() 总耗时: ~5-15ms                          │
├────────────────────────────────────────────────────────────┤
│ ├── sync_read("Present_Position"): ~3-8ms                  │
│ │   ├── 构建请求包: ~0.1ms                                  │
│ │   ├── 串口发送: ~0.5ms                                    │
│ │   ├── 舵机响应: ~2-5ms (取决于舵机数量)                    │
│ │   ├── 串口接收: ~0.5ms                                    │
│ │   └── 数据解析: ~0.1ms                                    │
│ └── camera.async_read(): ~5-10ms (每个摄像头)              │
└────────────────────────────────────────────────────────────┘
```

### 通信协议时序

```
主机                                    舵机
  │                                       │
  │──── SYNC_READ 请求 ─────────────────→│
  │     [FF FF FE 08 82 38 02 XX]        │
  │                                       │
  │←──── 响应 (舵机1) ───────────────────│
  │     [FF FF 01 04 00 LL HH XX]        │
  │                                       │
  │←──── 响应 (舵机2) ───────────────────│
  │     [FF FF 02 04 00 LL HH XX]        │
  │                                       │
  │         ... (舵机3-6)                 │
  │                                       │
```

---

## 🔧 SO101 Follower 舵机配置

**源码位置**: `lerobot/src/lerobot/robots/so101_follower/so101_follower.py` (第 37-52 行)

```python
# so101_follower.py (第 37-52 行)
self.bus = FeetechMotorsBus(
    port=self.config.port,  # "/dev/ttyACM0"
    motors={
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
    calibration=self.calibration,
)
```

---

## 📦 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         record.py 主循环                            │
│                              │                                      │
│                              ▼                                      │
│              robot.get_observation()                                │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│              SO101Follower.get_observation()                        │
│     (so101_follower.py 第 186-205 行)                               │
│                              │                                      │
│                              ▼                                      │
│         bus.sync_read("Present_Position")                          │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│              FeetechMotorsBus.sync_read()                           │
│     (motors_bus.py 第 1053-1091 行)                                 │
│                              │                                      │
│    1. 获取舵机列表: [1,2,3,4,5,6]                                    │
│    2. 查找地址: Present_Position = (56, 2)                          │
│                              │                                      │
│                              ▼                                      │
│                    _sync_read(56, 2, [1,2,3,4,5,6])                 │
│     (motors_bus.py 第 1093-1111 行)                                 │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                       _sync_read()                                  │
│                              │                                      │
│    _setup_sync_reader([1,2,3,4,5,6], 56, 2)                        │
│     (motors_bus.py 第 1143-1148 行)                                 │
│                              │                                      │
│                              ▼                                      │
│              sync_reader.txRxPacket()                               │
│     (group_sync_read.py 第 66-70 行)                                │
│                              │                                      │
│    ┌─────────────────────────┴─────────────────────────┐            │
│    │ txPacket()  →  发送 SYNC_READ 请求到 /dev/ttyACM0 │            │
│    │ (group_sync_read.py 第 44-50 行)                  │            │
│    │ rxPacket()  ←  接收所有舵机的响应                  │            │
│    │ (group_sync_read.py 第 52-64 行)                  │            │
│    └─────────────────────────┬─────────────────────────┘            │
│                              │                                      │
│                              ▼                                      │
│    getData(id, addr, length) 提取每个舵机的位置值                    │
│     (group_sync_read.py 第 72-82 行)                                │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│                    数据后处理                                        │
│                              │                                      │
│    _decode_sign()     符号解码                                       │
│    _normalize()       归一化到 [-100, 100] 或 [0, 100]              │
│     (motors_bus.py 第 974-1000 行)                                  │
│                              │                                      │
│                              ▼                                      │
│    返回: {                                                          │
│        "shoulder_pan.pos": 45.2,                                    │
│        "shoulder_lift.pos": -30.5,                                  │
│        "elbow_flex.pos": 60.0,                                      │
│        "wrist_flex.pos": -15.3,                                     │
│        "wrist_roll.pos": 0.0,                                       │
│        "gripper.pos": 80.0,                                         │
│    }                                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 总结

**读取 ACM0 数据的完整过程**:

1. **record.py** (第 331 行) 调用 `robot.get_observation()`
2. **SO101Follower** (第 186-205 行) 调用 `bus.sync_read("Present_Position")`
3. **FeetechMotorsBus** (第 1053-1091 行) 查找 Present_Position 地址 (56, 2)
4. **_sync_read()** (第 1093-1111 行) 配置 GroupSyncRead，添加所有舵机 ID
5. **scservo_sdk** (group_sync_read.py 第 66-70 行) 通过串口 `/dev/ttyACM0` 发送 SYNC_READ 指令
6. **舵机** 响应当前位置数据
7. **scservo_sdk** (第 72-82 行) 解析响应，提取位置值
8. **FeetechMotorsBus** (第 974-1000 行) 解码、归一化数据
9. **返回** `{舵机名: 位置值}` 字典

**关键参数**:
- 串口: `/dev/ttyACM0`
- 波特率: `1000000`
- Present_Position 地址: `56 (0x38)`
- 数据长度: `2 字节`
- 舵机数量: `6 个`
- 读取周期: `~5-15ms`

---

## 📁 文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| `record.py` | `lerobot/src/lerobot/record.py` | 数据采集主程序 |
| `so101_follower.py` | `lerobot/src/lerobot/robots/so101_follower/so101_follower.py` | SO101 Follower 实现 |
| `feetech.py` | `lerobot/src/lerobot/motors/feetech/feetech.py` | Feetech 电机总线 |
| `motors_bus.py` | `lerobot/src/lerobot/motors/motors_bus.py` | 电机总线基类 |
| `tables.py` | `lerobot/src/lerobot/motors/feetech/tables.py` | 参数表定义 |
| `group_sync_read.py` | `workspace/scservo_sdk_src/group_sync_read.py` | 同步读取 SDK |
| `port_handler.py` | `workspace/scservo_sdk_src/port_handler.py` | 串口处理 SDK |
| `protocol_packet_handler.py` | `workspace/scservo_sdk_src/protocol_packet_handler.py` | 协议处理 SDK |

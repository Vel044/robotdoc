# SO-101 舵机读写完整链路

> 以实际命令为基准：
> `--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12254705`
> 6个 STS-3215 舵机，ID=1~6，波特率 1Mbps，SRAM 寄存器内存映射访问

---

## 一、总体架构

```
         record.py   推理循环 record_loop()
              │ robot.get_observation() / send_action()
              ↓
┌─ Robot 抽象类 ──────────────┐      ┌─ MotorsBus 抽象类 ─────────────────────────┐
│  ┌─ so101_follower.py ───┐  │      │  motors_bus.py  归一化/反归一化 · 符号编码   │
│  │  观测/动作键名          │  │      │                                            │
│  │  标定 · 安全限幅        │  │─────▶│  ┌─ feetech/feetech.py ────────────────┐  │
│  └───────────────────────┘  │self.bus  │  Feetech协议 · 字节序 · Sign-Magnitude │  │
└─────────────────────────────┘      │  └────────────────────────────────────────┘  │
                                     └───────────────────────────────────────────────┘
                                                          │ import scservo_sdk
                                                          ↓
                                                    scservo_sdk
                                             GroupSyncRead / GroupSyncWrite
                                                          │ /dev/ttyACM0
                                                          ↓
                                                   STS-3215 × 6
                                               MCU 内存映射寄存器（EPROM + SRAM）
```

### Call Stack A — `robot.get_observation()` 读取当前位置

```
record.py: record_loop()
│
│  robot.get_observation()
│  ─────────────────────────────────────────────────────────────────────────
▼
robots/so101_follower/so101_follower.py
  SO101Follower.get_observation()
  入参: 无
  出参: dict[str, Any]  {"shoulder_pan.pos": 24.7, ..., "gripper.pos": 60.0,
                         "cam_0": np.ndarray(H,W,3), ...}
  │  调用 self.bus.sync_read("Present_Position")
  ▼
motors/motors_bus.py
  MotorsBus.sync_read(data_name="Present_Position", motors=None, normalize=True)
  入参: data_name(str) 寄存器名; motors(None)=全部6个电机; normalize(bool)=归一化开关
  出参: dict[str, float]  {"shoulder_pan": 24.7, "shoulder_lift": ..., "gripper": 60.0}
  │  查控制表 → addr=56, length=2
  │  调用 self._sync_read(addr=56, length=2, motor_ids=[1,2,3,4,5,6])
  │  → _decode_sign()  / _normalize()  处理原始值
  ▼
motors/motors_bus.py
  MotorsBus._sync_read(addr=56, length=2, motor_ids=[1..6], num_retry=0)
  入参: addr(int) 寄存器起始地址; length(int) 字节数; motor_ids(list[int]) 电机ID列表
  出参: tuple[dict[int,int], int]  ({1:2301,2:1800,...,6:3500}, comm_status)
  │  ┌─ 阶段1: _setup_sync_reader(motor_ids, addr=56, length=2)
  │  │    → sync_reader.clearParam()          ← 进入scs: data_dict.clear()，清空上次ID列表
  │  │    → sync_reader.start_address = 56    ← 仅Python属性赋值，不调用scs代码；txPacket()时才被读取
  │  │    → sync_reader.data_length = 2       ← 同上，纯字段赋值
  │  │    → sync_reader.addParam(id_) ×6      ← 进入scs: data_dict[id_]=[], is_param_changed=True
  │  │
  │  ├─ 阶段2: self.sync_reader.txRxPacket()  ← 进入 scservo_sdk
  │  │    入参: 无（参数已在 setup 阶段写入对象属性）
  │  │    出参: int  comm_status（0=COMM_SUCCESS）
  │  │    内部: txPacket() 发 0x82 广播读包 → rxPacket() 等6个舵机依次回包
  │  │          每包: [0xFF 0xFF ID 0x04 ERR DATA_L DATA_H CS]
  │  │          收到后校验 checksum，存入 data_dict[id_] = [DATA_L, DATA_H]
  │  │
  │  └─ 阶段3: self.sync_reader.getData(id_, 56, 2) ×6  ← 进入 scservo_sdk
  │       入参: scs_id(int), address(int)=56, data_length(int)=2
  │       出参: int  原始tick值，小端拼合 DATA_L|(DATA_H<<8)，如 0xFD|(0x08<<8)=2301
  └─ 返回 {1:2301, 2:1800, 3:2048, 4:2100, 5:2048, 6:3500}
```

---

### Call Stack B — `robot.send_action()` 写入目标位置

```
record.py: record_loop()
│
│  robot.send_action(action={"shoulder_pan.pos":-15.2, ..., "gripper.pos":72.0})
│  ─────────────────────────────────────────────────────────────────────────
▼
robots/so101_follower/so101_follower.py
  SO101Follower.send_action(action: dict[str,Any])
  入参: action(dict)  键="{motor_name}.pos", 值=归一化目标位置(float, -100~100 或 0~100)
  出参: dict[str, Any]  实际写入的目标位置（安全限幅后，键同入参）
  │  去掉 ".pos" 后缀 → goal_pos={"shoulder_pan":-15.2,...}
  │  调用 self.bus.sync_write("Goal_Position", goal_pos)
  ▼
motors/motors_bus.py
  MotorsBus.sync_write(data_name="Goal_Position", values={"shoulder_pan":-15.2,...}, normalize=True)
  入参: data_name(str) 寄存器名; values(dict[str,float]) 电机名→归一化目标值; normalize(bool)
  出参: None（无返回，写操作舵机不回包）
  │  _get_ids_values_dict() → {1:-15.2, 2:32.0, ..., 6:72.0}（名→ID）
  │  查控制表 → addr=42, length=2
  │  _unnormalize() → {1:1891, 2:2713, ..., 6:3436}（归一化值→原始tick）
  │  _encode_sign() → Goal_Position不在编码表，值不变
  │  调用 self._sync_write(addr=42, length=2, ids_values={1:1891,...})
  ▼
motors/motors_bus.py
  MotorsBus._sync_write(addr=42, length=2, ids_values={1:1891,...,6:3436}, num_retry=0)
  入参: addr(int) 寄存器起始地址; length(int) 字节数; ids_values(dict[int,int]) ID→原始tick值
  出参: int  comm_status（0=COMM_SUCCESS）
  │  ┌─ 阶段1: _setup_sync_writer(ids_values, addr=42, length=2)
  │  │    → sync_writer.clearParam()            ← 进入scs: data_dict.clear()
  │  │    → sync_writer.start_address = 42      ← 仅Python属性赋值，txPacket()时才被读取
  │  │    → sync_writer.data_length = 2         ← 同上，纯字段赋值
  │  │    → 对每个 (id_, value):
  │  │        _serialize_data(1891, 2)
  │  │          → SCS_LOBYTE(1891)=0x83, SCS_HIBYTE(1891)=0x07  ← 进入 scservo_sdk
  │  │          → 返回 [0x83, 0x07]（小端序字节列表）
  │  │        sync_writer.addParam(id_=1, data=[0x83,0x07]) ← 进入 scservo_sdk
  │  │          → data_dict[1]=[0x83,0x07]（注册ID+数据）
  │  │        ... ×6个电机
  │  │
  │  └─ 阶段2: self.sync_writer.txPacket()  ← 进入 scservo_sdk
  │       入参: 无（参数已在 setup 阶段写入对象属性）
  │       出参: int  comm_status（0=COMM_SUCCESS）
  │       内部: makeParam() 展开 data_dict → param=[1,0x83,0x07, 2,..., 6,...]
  │             syncWriteTxOnly() 拼包: [0xFF 0xFF 0xFE LEN 0x83 42 0 2 0
  │                                      1 0x83 0x07  2 ...  6 ...  CS]
  │             writePort() 写串口，不等待任何回包
  └─ 6个舵机同时收到广播写包，各自更新 Goal_Position 寄存器(地址42)，驱动电机转动
```

---

feetech.py 在 `__init__` 中通过 `import scservo_sdk as scs` 导入 SDK，并创建了以下关键对象：

```python
# feetech.py:126-137
import scservo_sdk as scs

self.port_handler = scs.PortHandler(self.port)                # 串口管理器（打开/关闭/读/写）
self.packet_handler = scs.PacketHandler(protocol_version)     # 协议处理器（组包/拆包/校验）
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)   # 同步读管理器
self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)  # 同步写管理器
```



## 二、读取当前位置：`get_observation()` → `sync_read("Present_Position")`

### 第1层：so101_follower.py — `get_observation()`

> 文件：`lerobot/src/lerobot/robots/so101_follower/so101_follower.py:408-462`

```python
def get_observation(self) -> dict[str, Any]:
    """获取机器人的当前观测数据（每帧调用一次，约 33ms@30fps）。"""
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # 步骤1: 批量同步读取6个舵机的当前位置
    # 返回原始格式: {"shoulder_pan": -12.5, "shoulder_lift": 30.0, ..., "gripper": 60.0}
    obs_dict = self.bus.sync_read("Present_Position")

    # 步骤2: 键名添加 ".pos" 后缀，与 observation_features schema 对齐
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    dt_ms = (time.perf_counter() - start) * 1e3

    # 步骤3: 非阻塞读取相机最新帧
    for cam_key, cam in self.cameras.items():
        start = time.perf_counter()
        obs_dict[cam_key] = cam.async_read()

    return obs_dict
```

---

### 第2层：motors_bus.py — `sync_read()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1628-1699`

```python
def sync_read(
    self,
    data_name: str,
    motors: str | list[str] | None = None,
    *,
    normalize: bool = True,
    num_retry: int = 0,
) -> dict[str, Value]:
    """同步读取多个电机的同一寄存器（一次广播，所有电机同时响应）。"""
    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # 1) 校验协议支持
    self._assert_protocol_is_compatible("sync_read")

    # 2) 解析电机名称为 ID 和型号
    names = self._get_motors_list(motors)
    ids = [self.motors[motor].id for motor in names]
    models = [self.motors[motor].model for motor in names]

    # 3) 校验地址一致性（混用不同型号时必须地址兼容）
    if self._has_different_ctrl_tables:
        assert_same_address(self.model_ctrl_table, models, data_name)

    # 查控制表，得到寄存器地址和字节长度
    # "Present_Position" → (56, 2)，即从舵机内存第56字节起读取2字节
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)

    # 4) 执行底层同步读取（串口通信主耗时 ~5~15ms）
    ids_values, _ = self._sync_read(
        addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=...
    )
    # 此时 ids_values = {1: 2301, 2: 1800, 3: 2048, 4: 2100, 5: 2048, 6: 3500}
    # （原始 12-bit tick 值，0~4095）

    # 5) Sign-Magnitude 符号位解码（bit15为符号位）
    ids_values = self._decode_sign(data_name, ids_values)
    # 正常运行时位置值为正（0~4095范围内），bit15 通常为0，解码后值不变

    # 6) 归一化（原始 tick → -100~100 或 0~100）
    if normalize and data_name in self.normalized_data:
        ids_values = self._normalize(ids_values)

    # 7) ID → 电机名称
    return {self._id_to_name(id_): value for id_, value in ids_values.items()}
    # {1: 24.7, ...} → {"shoulder_pan": 24.7, "shoulder_lift": ..., "gripper": 60.0}
```

---

### 第3层：motors_bus.py — `_sync_read()`（串口通信三阶段）

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1701-1766`
>
> 注意：`self.sync_reader` 是 `scs.GroupSyncRead` 的实例（在 feetech.py:134 创建），
> 以下三个方法调用全部进入 scservo_sdk：
> - `self.sync_reader.clearParam()` / `addParam()` — 注册要读取的电机 ID 列表
> - `self.sync_reader.txRxPacket()` — 发送广播读指令 + 等待所有电机回包
> - `self.sync_reader.getData()` — 从 SDK 内部缓冲区解析各电机返回的原始值

```python
def _sync_read(
    self,
    addr: int,
    length: int,
    motor_ids: list[int],
    *,
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> tuple[dict[int, int], int]:
    """底层同步读取实现（setup → txrx → unpack 三阶段）。"""

    # === 阶段1: Setup — 配置 scservo_sdk 的 GroupSyncRead 对象 ===
    # 告诉 SDK：要读哪个寄存器（addr=56），读几个字节（length=2），读哪些电机（ID=1~6）
    # 内部做了4件事：
    #   1. clearParam()  — 清空上次的参数（每次 sync_read 都要重新设置）
    #   2. start_address = addr   — 告诉 SDK 目标寄存器起始地址（56 = Present_Position）
    #   3. data_length = length   — 告诉 SDK 每个电机要读几个字节（2字节 = 16位位置值）
    #   4. addParam(id_) × 6     — 注册6个电机 ID，SDK 后续只会等这些电机的回包
    self._setup_sync_reader(motor_ids, addr, length)

    # === 阶段2: TxRx — 发送广播读包 + 等待6个应答（主要耗时 ~5~15ms） ===
    # self.sync_reader.txRxPacket() 进入 scservo_sdk，做了两件事：
    #   1. txPacket() — 把上面 setup 的参数打包成一条 Sync Read 指令包（指令码 0x82），
    #                  通过 port_handler.writePort() 发到串口
    #   2. rxPacket() — 循环等待6个电机依次回包（半双工，一次只能一个回），
    #                  每收到一个就校验 checksum 并存入内部 data_dict
    for n_try in range(1 + num_retry):
        comm = self.sync_reader.txRxPacket()
        if self._is_comm_success(comm):
            break

    if not self._is_comm_success(comm) and raise_on_error:
        raise ConnectionError(...)

    # === 阶段3: Unpack — 从 SDK 内部缓冲区提取各电机的原始值 ===
    # self.sync_reader.getData() 进入 scservo_sdk，
    # 从阶段2存好的 data_dict 中按 (电机ID, 地址, 长度) 取出原始字节，拼成整数返回
    # 小端序：DATA_L | (DATA_H << 8)，例如 0xFD + (0x08 << 8) = 0x08FD = 2301
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
    # → {1: 2301, 2: 1800, 3: 2048, 4: 2100, 5: 2048, 6: 3500}

    return values, comm
```

**串口上发生的事情（Feetech 协议 v0）：**

```
主机发出广播读包（Sync Read Instruction，指令码 0x82）：
┌────────────────────────────────────────────────────────────┐
│ 0xFF 0xFF 0xFE LEN 0x82 ADDR_L ADDR_H LEN_L LEN_H         │
│               ↑广播ID  ↑指令码  ↑地址56     ↑长度2          │
│ ID1 ID2 ID3 ID4 ID5 ID6  CHECKSUM                          │
└────────────────────────────────────────────────────────────┘

每个舵机（ID=1~6）依次回包（半双工，不能同时发）：
┌──────────────────────────────────────────────────────────┐
│ 0xFF 0xFF ID  LEN ERR  DATA_L DATA_H  CHECKSUM           │
│              ↑0x04      ↑ 2字节位置原始值（小端序）         │
└──────────────────────────────────────────────────────────┘
  ID=1: [0xFF 0xFF 0x01 0x04 0x00 0xFD 0x08 CS]  → 0x08FD = 2301
  ID=2: [0xFF 0xFF 0x02 0x04 0x00 0x08 0x07 CS]  → 0x0708 = 1800
```

**小端序**：`DATA_L=0xFD, DATA_H=0x08 → 0x08FD = 2301`

---

### 辅助：motors_bus.py — `_setup_sync_reader()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1768-1782`
>
> `self.sync_reader` 是 `scs.GroupSyncRead` 实例，以下调用全部进入 scservo_sdk：
> - `clearParam()` — 清空之前注册的电机 ID 列表
> - `start_address = addr` — 设置要读取的寄存器起始地址
> - `data_length = length` — 设置每个电机要读取的字节数
> - `addParam(id_)` — 把电机 ID 注册进去，SDK 后续只等这些 ID 的回包

```python
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
    """配置 scservo_sdk 的 GroupSyncRead：告诉 SDK 读哪个寄存器、读几个字节、读哪些电机。"""
    self.sync_reader.clearParam()
    self.sync_reader.start_address = addr     # 56（Present_Position）
    self.sync_reader.data_length = length     # 2（2字节）
    for id_ in motor_ids:
        self.sync_reader.addParam(id_)        # 注册 ID=1,2,3,4,5,6
```

---

### 辅助：scservo_sdk — `GroupSyncRead.addParam()` / `GroupSyncWrite.addParam()`

> 文件：`scservo_sdk/group_sync_read.py:42-54`、`scservo_sdk/group_sync_write.py:48-68`
>
> 两者都是把电机信息注册到 SDK 对象的**内部字典** `data_dict`，供后续 `txPacket()` 取用。

**Sync Read 的 addParam**（只注册 ID，不需要数据——因为我们是"要读"，还没拿到数据）：

```python
# group_sync_read.py:42-54
def addParam(self, scs_id):
    if scs_id in self.data_dict:  # 已存在则拒绝重复
        return False
    self.data_dict[scs_id] = []   # 空列表占位，等 rxPacket() 填充实际数据
    self.is_param_changed = True  # 标记"参数变了，下次发送前要重新 makeParam()"
    return True
```

**Sync Write 的 addParam**（需要注册 ID + 要写入的数据——因为我们是"要写"，数据已知）：

```python
# group_sync_write.py:48-68
def addParam(self, scs_id, data):
    if scs_id in self.data_dict:  # 已存在则拒绝重复
        return False
    if len(data) > self.data_length:  # 数据超长则拒绝
        return False
    self.data_dict[scs_id] = data   # 存入 [byte0, byte1]，例如 [0x83, 0x07]
    self.is_param_changed = True    # 标记"参数变了，下次发送前要重新 makeParam()"
    return True
```

**两者共用的 `makeParam()`**（发送前把字典展开为连续字节列表）：

```python
# group_sync_read.py:29-40
def makeParam(self):
    self.param = []
    for scs_id in self.data_dict:
        self.param.append(scs_id)  # 只有 ID，如 [1, 2, 3, 4, 5, 6]

# group_sync_write.py:27-46
def makeParam(self):
    self.param = []
    for scs_id in self.data_dict:
        self.param.append(scs_id)              # ID
        self.param.extend(self.data_dict[scs_id])  # + 数据，如 [1, 0x83, 0x07, 2, 0x09, 0x0A, ...]
```

> **核心设计**：`addParam` 只往 `self.data_dict` 字典里存数据，`txPacket()` 发送时检查 `is_param_changed` 标记，
> 如果变了就调 `makeParam()` 把字典展开成 `self.param` 连续字节列表，再交给 `ph.syncReadTx()` / `ph.syncWriteTxOnly()` 拼完整包发出去。

---

### 辅助：feetech.py — `_decode_sign()`

> 文件：`lerobot/src/lerobot/motors/feetech/feetech.py:351-359`

```python
def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
    """Sign-Magnitude 符号位解码：查编码表，对需要解码的寄存器进行解码。"""
    for id_ in ids_values:
        model = self._id_to_model(id_)
        encoding_table = self.model_encoding_table.get(model)
        # Present_Position 在 STS_SMS_SERIES_ENCODINGS_TABLE 里，sign_bit=15
        if encoding_table and data_name in encoding_table:
            sign_bit = encoding_table[data_name]
            ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)

    return ids_values
```

---

### 辅助：encoding_utils.py — `decode_sign_magnitude()`

> 文件：`lerobot/src/lerobot/utils/encoding_utils.py:29-36`

```python
def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):
    """将 Feetech 寄存器原始值 → Python int（符号-幅值解码）。"""
    direction_bit = (encoded_value >> sign_bit_index) & 1   # 取 bit15
    magnitude_mask = (1 << sign_bit_index) - 1              # = 0x7FFF，取低15位
    magnitude = encoded_value & magnitude_mask
    return -magnitude if direction_bit else magnitude
```

---

### 辅助：motors_bus.py — `_normalize()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1221-1277`

```python
def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
    """将原始编码值归一化为用户友好的范围。"""
    if not self.calibration:
        raise RuntimeError(f"{self} has no calibration registered.")

    normalized_values = {}
    for id_, val in ids_values.items():
        motor = self._id_to_name(id_)
        min_ = self.calibration[motor].range_min
        max_ = self.calibration[motor].range_max
        drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
        if max_ == min_:
            raise ValueError(...)

        bounded_val = min(max_, max(min_, val))

        if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
            # 身体关节：((raw - min) / (max - min)) * 200 - 100
            # 例: shoulder_pan, raw=2301, min=1024, max=3072
            #     = ((2301-1024)/(3072-1024))*200 - 100 = 24.7
            norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
            normalized_values[id_] = -norm if drive_mode else norm

        elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
            # 夹爪：((raw - min) / (max - min)) * 100
            norm = ((bounded_val - min_) / (max_ - min_)) * 100
            normalized_values[id_] = 100 - norm if drive_mode else norm

        elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
            mid = (min_ + max_) / 2
            max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            normalized_values[id_] = (val - mid) * 360 / max_res

    return normalized_values
```

`min` / `max` 来自标定文件 `~/.lerobot/calibration/robots/R12254705.json`：
```json
{
  "shoulder_pan": {"range_min": 1024, "range_max": 3072, "homing_offset": 2048, ...},
  "gripper":      {"range_min": 2500, "range_max": 3800, ...}
}
```

---

## 三、写入目标位置：`send_action()` → `sync_write("Goal_Position")`

### 第1层：so101_follower.py — `send_action()`

> 文件：`lerobot/src/lerobot/robots/so101_follower/so101_follower.py:464-528`

```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """命令机械臂移动到目标关节配置（每帧调用一次）。"""
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    total_start = time.perf_counter()

    # 步骤1: 去掉键名的 ".pos" 后缀，转为总线期望的电机名格式
    # {"shoulder_pan.pos": -15.2} → {"shoulder_pan": -15.2}
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

    # 步骤2: 安全限幅（可选，本次 max_relative_target=None，跳过）
    read_ms = 0.0
    if self.config.max_relative_target is not None:
        read_start = time.perf_counter()
        present_pos = self.bus.sync_read("Present_Position")
        read_ms = (time.perf_counter() - read_start) * 1e3
        goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
        goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

    # 步骤3: 批量写入目标位置到6个舵机的 Goal_Position 寄存器
    write_start = time.perf_counter()
    self.bus.sync_write("Goal_Position", goal_pos)
    write_ms = (time.perf_counter() - write_start) * 1e3

    # 步骤4: 返回实际写入的目标位置（加回 ".pos" 后缀）
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

---

### 第2层：motors_bus.py — `sync_write()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1784-1834`

```python
def sync_write(
    self,
    data_name: str,
    values: Value | dict[str, Value],
    *,
    normalize: bool = True,
    num_retry: int = 0,
) -> None:
    """向多个电机的同一寄存器同步写入（无响应包，速度快但可能丢包）。"""
    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # 1) 规整化输入为 {电机ID: 值}
    # {"shoulder_pan": -15.2} → {1: -15.2, 2: 32.0, ..., 6: 72.0}
    ids_values = self._get_ids_values_dict(values)
    models = [self._id_to_model(id_) for id_ in ids_values]

    # 2) 校验地址一致性
    if self._has_different_ctrl_tables:
        assert_same_address(self.model_ctrl_table, models, data_name)

    # 查控制表：Goal_Position → (42, 2)
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)

    # 3) 反归一化（-100~100 → 原始 tick 0~4095）
    if normalize and data_name in self.normalized_data:
        ids_values = self._unnormalize(ids_values)
    # 结果：{1: 1891, 2: 2713, 3: 2150, 4: 1946, 5: 2423, 6: 3436}

    # 4) Sign-Magnitude 符号位编码
    # Goal_Position 不在编码表里，直接跳过
    ids_values = self._encode_sign(data_name, ids_values)

    # 5) 底层同步写入
    self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=...)
```

---

### 第3层：motors_bus.py — `_sync_write()`（串口通信两阶段）

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1836-1893`
>
> 注意：`self.sync_writer` 是 `scs.GroupSyncWrite` 的实例（在 feetech.py:135 创建），
> 以下方法调用全部进入 scservo_sdk：
> - `self.sync_writer.clearParam()` / `addParam()` — 注册要写入的电机 ID 和数据
> - `self.sync_writer.txPacket()` — 把所有电机的数据打包成一条 Sync Write 广播包发出去，**不等待任何回包**

```python
def _sync_write(
    self,
    addr: int,
    length: int,
    ids_values: dict[int, int],
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> int:
    """底层同步写入实现（setup → tx 两阶段，不等待应答）。"""

    # === 阶段1: Setup — 配置 scservo_sdk 的 GroupSyncWrite 对象 ===
    # 告诉 SDK：写哪个寄存器（addr=42），写几个字节（length=2），每个电机写什么值
    # 内部做了4件事：
    #   1. clearParam()  — 清空上次的参数
    #   2. start_address = addr   — 告诉 SDK 目标寄存器起始地址（42 = Goal_Position）
    #   3. data_length = length   — 告诉 SDK 每个电机要写几个字节（2字节）
    #   4. addParam(id_, data) × 6 — 把每个电机的 ID + 小端序字节数据注册进去
    #      例如 ID=1, value=1891 → data=[0x83, 0x07]（小端序）
    self._setup_sync_writer(ids_values, addr, length)

    # === 阶段2: Tx — 发送广播写包（主要耗时 ~1~3ms） ===
    # self.sync_writer.txPacket() 进入 scservo_sdk，做了一件事：
    #   把上面 setup 注册的所有 (ID, data) 打包成一条 Sync Write 指令包（指令码 0x83），
    #   通过 port_handler.writePort() 发到串口，**不等待任何回包**，直接返回
    for n_try in range(1 + num_retry):
        comm = self.sync_writer.txPacket()
        if self._is_comm_success(comm):
            break

    if not self._is_comm_success(comm) and raise_on_error:
        raise ConnectionError(...)

    return comm
```

**串口上发生的事情：**

```
主机发出广播写包（Sync Write Instruction，指令码 0x83）：
┌─────────────────────────────────────────────────────────────────────────┐
│ 0xFF 0xFF 0xFE LEN 0x83 ADDR_L ADDR_H DATA_LEN_L DATA_LEN_H            │
│               ↑广播ID  ↑指令码  ↑地址42           ↑每条数据字节数=2       │
│ [ID1 DATA_L DATA_H]  [ID2 DATA_L DATA_H] ... [ID6 DATA_L DATA_H]       │
│  0x01 0x83 0x07       0x02 ...                 0x06 ...                 │
│ CHECKSUM                                                                 │
└─────────────────────────────────────────────────────────────────────────┘

6个舵机同时收到，同时更新自己的 Goal_Position 寄存器（地址42，写2字节）
舵机不回包 → 主机不等待 → 耗时仅为发送时间 ≈ 1~3ms
```

---

### 辅助：motors_bus.py — `_setup_sync_writer()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1895-1908`
>
> `self.sync_writer` 是 `scs.GroupSyncWrite` 实例，以下调用全部进入 scservo_sdk：
> - `clearParam()` — 清空之前注册的电机 ID 和数据
> - `start_address = addr` — 设置要写入的寄存器起始地址
> - `data_length = length` — 设置每个电机要写入的字节数
> - `addParam(id_, data)` — 把电机 ID 和对应的小端序字节数据注册进去

```python
def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
    """配置 scservo_sdk 的 GroupSyncWrite：告诉 SDK 写哪个寄存器、写几个字节、每个电机写什么值。"""
    self.sync_writer.clearParam()
    self.sync_writer.start_address = addr     # 42（Goal_Position）
    self.sync_writer.data_length = length     # 2（2字节）
    for id_, value in ids_values.items():
        data = self._serialize_data(value, length)
        # _serialize_data(1891, 2) → [0x83, 0x07]（小端：低字节先发）
        self.sync_writer.addParam(id_, data)
```

---

### 辅助：motors_bus.py — `_unnormalize()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1279-1317`

```python
def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
    """将归一化值反向转换为电机原始编码值。"""
    if not self.calibration:
        raise RuntimeError(f"{self} has no calibration registered.")

    unnormalized_values = {}
    for id_, val in ids_values.items():
        motor = self._id_to_name(id_)
        min_ = self.calibration[motor].range_min
        max_ = self.calibration[motor].range_max
        drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
        if max_ == min_:
            raise ValueError(...)

        if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
            # 身体关节：norm=-15.2 → raw=1891
            # raw = int(((val+100)/200) * (max-min) + min)
            val = -val if drive_mode else val
            bounded_val = min(100.0, max(-100.0, val))
            unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)

        elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
            # 夹爪：val=72.0 → raw=3436
            val = 100 - val if drive_mode else val
            bounded_val = min(100.0, max(0.0, val))
            unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)

        elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
            mid = (min_ + max_) / 2
            max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
            unnormalized_values[id_] = int((val * max_res / 360) + mid)

    return unnormalized_values
```

**举例计算：**
```
shoulder_pan（RANGE_M100_100）: val=-15.2, min=1024, max=3072
raw = int(((-15.2 + 100) / 200) * (3072 - 1024) + 1024)
    = int((84.8 / 200) * 2048 + 1024)
    = int(867.9 + 1024) = 1891

gripper（RANGE_0_100）: val=72.0, min=2500, max=3800
raw = int((72.0 / 100) * (3800 - 2500) + 2500)
    = int(0.72 * 1300 + 2500) = 3436
```

---

### 辅助：feetech.py — `_encode_sign()`

> 文件：`lerobot/src/lerobot/motors/feetech/feetech.py:341-349`

```python
def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
    """Sign-Magnitude 符号位编码：查编码表，对需要的寄存器进行编码。"""
    for id_ in ids_values:
        model = self._id_to_model(id_)
        encoding_table = self.model_encoding_table.get(model)
        # Goal_Position 不在编码表里，直接跳过不做任何处理
        if encoding_table and data_name in encoding_table:
            sign_bit = encoding_table[data_name]
            ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)

    return ids_values
```

---

### 辅助：feetech.py — `_split_into_byte_chunks()`（小端序序列化）

> 文件：`lerobot/src/lerobot/motors/feetech/feetech.py:69-83`
>
> `import scservo_sdk as scs` 在函数内部导入，
> `scs.SCS_LOBYTE` / `scs.SCS_HIBYTE` 是 SDK 提供的字节拆分工具函数：
> - `SCS_LOBYTE(1891)` = `1891 & 0xFF` = `0x83`（低字节）
> - `SCS_HIBYTE(1891)` = `1891 >> 8` = `0x07`（高字节）

```python
def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    """将整数值按小端序拆分为字节列表（Feetech 使用小端序）。"""
    import scservo_sdk as scs

    if length == 1:
        data = [value]
    elif length == 2:
        # 1891 → [0x83, 0x07]
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
    elif length == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    return data
```

---

### 辅助：encoding_utils.py — `encode_sign_magnitude()`

> 文件：`lerobot/src/lerobot/utils/encoding_utils.py:16-26`

```python
def encode_sign_magnitude(value: int, sign_bit_index: int):
    """将 Python int → Feetech 寄存器原始值（符号-幅值编码）。"""
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(...)

    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude
```

---

## 四、读 vs 写对比

|             | sync_read（get_observation）             | sync_write（send_action）                        |
| ----------- | ---------------------------------------- | ------------------------------------------------ |
| 入口        | `self.bus.sync_read("Present_Position")` | `self.bus.sync_write("Goal_Position", goal_pos)` |
| 寄存器      | Present_Position: 地址56, 2字节          | Goal_Position: 地址42, 2字节                     |
| SDK对象     | `scs.GroupSyncRead` (feetech.py:134)     | `scs.GroupSyncWrite` (feetech.py:135)            |
| SDK关键调用 | `sync_reader.txRxPacket()` — 发+收       | `sync_writer.txPacket()` — 只发不收              |
| 指令码      | `0x82`                                   | `0x83`                                           |
| 归一化      | `_normalize`（tick→-100~100）            | `_unnormalize`（-100~100→tick）                  |
| 符号解码    | `_decode_sign`（bit15符号位解码）        | `_encode_sign`（Goal_Position不编码，跳过）      |
| 舵机响应    | 每个舵机回一个状态包                     | **无响应包**                                     |
| 耗时        | ~5~15ms（等6个应答）                     | ~1~3ms（只发不收）                               |
| 风险        | 无                                       | 丢包时舵机不动（静默失败）                       |

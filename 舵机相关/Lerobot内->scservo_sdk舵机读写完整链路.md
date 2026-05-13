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

来源：lerobot/src/lerobot/motors/feetech/feetech.py:__init__.sync_reader 初始化（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# feetech.py:126-137
import scservo_sdk as scs  # 本行参与当前链路的控制流或数据准备

self.port_handler = scs.PortHandler(self.port)                # 串口管理器（打开/关闭/读/写）
self.packet_handler = scs.PacketHandler(protocol_version)     # 协议处理器（组包/拆包/校验）
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)   # 同步读管理器
self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)  # 同步写管理器
```



## 二、读取当前位置：`get_observation()` → `sync_read("Present_Position")`

### 第1层：so101_follower.py — `get_observation()`

> 文件：`lerobot/src/lerobot/robots/so101_follower/so101_follower.py:408-462`

来源：lerobot/src/lerobot/robots/so101_follower/so101_follower.py:get_observation（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def get_observation(self) -> dict[str, Any]:  # 定义本链路要说明的函数入口
    """获取机器人的当前观测数据（每帧调用一次，约 33ms@30fps）。"""
    if not self.is_connected:  # 检查条件，决定是否进入该分支
        raise DeviceNotConnectedError(f"{self} is not connected.")  # 调用下一层函数继续完成当前链路动作

    # 步骤1: 批量同步读取6个舵机的当前位置
    # 返回原始格式: {"shoulder_pan": -12.5, "shoulder_lift": 30.0, ..., "gripper": 60.0}
    obs_dict = self.bus.sync_read("Present_Position")  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤2: 键名添加 ".pos" 后缀，与 observation_features schema 对齐
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}  # 保存本链路后续步骤需要使用的中间状态或参数
    dt_ms = (time.perf_counter() - start) * 1e3  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤3: 非阻塞读取相机最新帧
    for cam_key, cam in self.cameras.items():  # 遍历本次链路涉及的元素
        start = time.perf_counter()  # 保存本链路后续步骤需要使用的中间状态或参数
        obs_dict[cam_key] = cam.async_read()  # 保存本链路后续步骤需要使用的中间状态或参数

    return obs_dict  # 把本层处理结果返回给调用方
```

---

### 第2层：motors_bus.py — `sync_read()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1628-1699`

来源：lerobot/src/lerobot/motors/motors_bus.py:sync_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def sync_read(  # 定义本链路要说明的函数入口
    self,  # 本行参与当前链路的控制流或数据准备
    data_name: str,  # 本行参与当前链路的控制流或数据准备
    motors: str | list[str] | None = None,  # 保存本链路后续步骤需要使用的中间状态或参数
    *,  # 本行参与当前链路的控制流或数据准备
    normalize: bool = True,  # 保存本链路后续步骤需要使用的中间状态或参数
    num_retry: int = 0,  # 保存本链路后续步骤需要使用的中间状态或参数
) -> dict[str, Value]:  # 本行参与当前链路的控制流或数据准备
    """同步读取多个电机的同一寄存器（一次广播，所有电机同时响应）。"""
    if not self.is_connected:  # 检查条件，决定是否进入该分支
        raise DeviceNotConnectedError(...)  # 调用下一层函数继续完成当前链路动作

    # 1) 校验协议支持
    self._assert_protocol_is_compatible("sync_read")  # 调用下一层函数继续完成当前链路动作

    # 2) 解析电机名称为 ID 和型号
    names = self._get_motors_list(motors)  # 保存本链路后续步骤需要使用的中间状态或参数
    ids = [self.motors[motor].id for motor in names]  # 保存本链路后续步骤需要使用的中间状态或参数
    models = [self.motors[motor].model for motor in names]  # 保存本链路后续步骤需要使用的中间状态或参数

    # 3) 校验地址一致性（混用不同型号时必须地址兼容）
    if self._has_different_ctrl_tables:  # 检查条件，决定是否进入该分支
        assert_same_address(self.model_ctrl_table, models, data_name)  # 调用下一层函数继续完成当前链路动作

    # 查控制表，得到寄存器地址和字节长度
    # "Present_Position" → (56, 2)，即从舵机内存第56字节起读取2字节
    model = next(iter(models))  # 保存本链路后续步骤需要使用的中间状态或参数
    addr, length = get_address(self.model_ctrl_table, model, data_name)  # 保存本链路后续步骤需要使用的中间状态或参数

    # 4) 执行底层同步读取（串口通信主耗时 ~5~15ms）
    ids_values, _ = self._sync_read(  # 保存本链路后续步骤需要使用的中间状态或参数
        addr, length, ids, num_retry=num_retry, raise_on_error=True, err_msg=...  # 保存本链路后续步骤需要使用的中间状态或参数
    )
    # 此时 ids_values = {1: 2301, 2: 1800, 3: 2048, 4: 2100, 5: 2048, 6: 3500}
    # （原始 12-bit tick 值，0~4095）

    # 5) Sign-Magnitude 符号位解码（bit15为符号位）
    ids_values = self._decode_sign(data_name, ids_values)  # 保存本链路后续步骤需要使用的中间状态或参数
    # 正常运行时位置值为正（0~4095范围内），bit15 通常为0，解码后值不变

    # 6) 归一化（原始 tick → -100~100 或 0~100）
    if normalize and data_name in self.normalized_data:  # 检查条件，决定是否进入该分支
        ids_values = self._normalize(ids_values)  # 保存本链路后续步骤需要使用的中间状态或参数

    # 7) ID → 电机名称
    return {self._id_to_name(id_): value for id_, value in ids_values.items()}  # 把本层处理结果返回给调用方
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

来源：lerobot/src/lerobot/motors/motors_bus.py:_sync_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _sync_read(  # 定义本链路要说明的函数入口
    self,  # 本行参与当前链路的控制流或数据准备
    addr: int,  # 本行参与当前链路的控制流或数据准备
    length: int,  # 本行参与当前链路的控制流或数据准备
    motor_ids: list[int],  # 本行参与当前链路的控制流或数据准备
    *,  # 本行参与当前链路的控制流或数据准备
    num_retry: int = 0,  # 保存本链路后续步骤需要使用的中间状态或参数
    raise_on_error: bool = True,  # 保存本链路后续步骤需要使用的中间状态或参数
    err_msg: str = "",  # 保存本链路后续步骤需要使用的中间状态或参数
) -> tuple[dict[int, int], int]:  # 本行参与当前链路的控制流或数据准备
    """底层同步读取实现（setup → txrx → unpack 三阶段）。"""

    # === 阶段1: Setup — 配置 scservo_sdk 的 GroupSyncRead 对象 ===
    # 告诉 SDK：要读哪个寄存器（addr=56），读几个字节（length=2），读哪些电机（ID=1~6）
    # 内部做了4件事：
    #   1. clearParam()  — 清空上次的参数（每次 sync_read 都要重新设置）
    #   2. start_address = addr   — 告诉 SDK 目标寄存器起始地址（56 = Present_Position）
    #   3. data_length = length   — 告诉 SDK 每个电机要读几个字节（2字节 = 16位位置值）
    #   4. addParam(id_) × 6     — 注册6个电机 ID，SDK 后续只会等这些电机的回包
    self._setup_sync_reader(motor_ids, addr, length)  # 调用下一层函数继续完成当前链路动作

    # === 阶段2: TxRx — 发送广播读包 + 等待6个应答（主要耗时 ~5~15ms） ===
    # self.sync_reader.txRxPacket() 进入 scservo_sdk，做了两件事：
    #   1. txPacket() — 把上面 setup 的参数打包成一条 Sync Read 指令包（指令码 0x82），
    #                  通过 port_handler.writePort() 发到串口
    #   2. rxPacket() — 循环等待6个电机依次回包（半双工，一次只能一个回），
    #                  每收到一个就校验 checksum 并存入内部 data_dict
    for n_try in range(1 + num_retry):  # 遍历本次链路涉及的元素
        comm = self.sync_reader.txRxPacket()  # 保存本链路后续步骤需要使用的中间状态或参数
        if self._is_comm_success(comm):  # 检查条件，决定是否进入该分支
            break  # 本行参与当前链路的控制流或数据准备

    if not self._is_comm_success(comm) and raise_on_error:  # 检查条件，决定是否进入该分支
        raise ConnectionError(...)  # 调用下一层函数继续完成当前链路动作

    # === 阶段3: Unpack — 从 SDK 内部缓冲区提取各电机的原始值 ===
    # self.sync_reader.getData() 进入 scservo_sdk，
    # 从阶段2存好的 data_dict 中按 (电机ID, 地址, 长度) 取出原始字节，拼成整数返回
    # 小端序：DATA_L | (DATA_H << 8)，例如 0xFD + (0x08 << 8) = 0x08FD = 2301
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}  # 保存本链路后续步骤需要使用的中间状态或参数
    # → {1: 2301, 2: 1800, 3: 2048, 4: 2100, 5: 2048, 6: 3500}

    return values, comm  # 把本层处理结果返回给调用方
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

来源：lerobot/src/lerobot/motors/motors_bus.py:_setup_sync_reader（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:  # 定义本链路要说明的函数入口
    """配置 scservo_sdk 的 GroupSyncRead：告诉 SDK 读哪个寄存器、读几个字节、读哪些电机。"""
    self.sync_reader.clearParam()  # 清理上一轮注册参数，避免旧 ID 或旧数据残留
    self.sync_reader.start_address = addr     # 56（Present_Position）  # 保存本链路后续步骤需要使用的中间状态或参数
    self.sync_reader.data_length = length     # 2（2字节）
    for id_ in motor_ids:  # 遍历本次链路涉及的元素
        self.sync_reader.addParam(id_)        # 注册 ID=1,2,3,4,5,6
```

---

### 辅助：scservo_sdk — `GroupSyncRead.addParam()` / `GroupSyncWrite.addParam()`

> 文件：`scservo_sdk/group_sync_read.py:42-54`、`scservo_sdk/group_sync_write.py:48-68`
>
> 两者都是把电机信息注册到 SDK 对象的**内部字典** `data_dict`，供后续 `txPacket()` 取用。

**Sync Read 的 addParam**（只注册 ID，不需要数据——因为我们是"要读"，还没拿到数据）：

来源：scservo_sdk/group_sync_read.py:addParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py:42-54
def addParam(self, scs_id):  # 定义本链路要说明的函数入口
    if scs_id in self.data_dict:  # 已存在则拒绝重复
        return False  # 把本层处理结果返回给调用方
    self.data_dict[scs_id] = []   # 空列表占位，等 rxPacket() 填充实际数据
    self.is_param_changed = True  # 标记"参数变了，下次发送前要重新 makeParam()"
    return True  # 把本层处理结果返回给调用方
```

**Sync Write 的 addParam**（需要注册 ID + 要写入的数据——因为我们是"要写"，数据已知）：

来源：scservo_sdk/group_sync_write.py:addParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_write.py:48-68
def addParam(self, scs_id, data):  # 定义本链路要说明的函数入口
    if scs_id in self.data_dict:  # 已存在则拒绝重复
        return False  # 把本层处理结果返回给调用方
    if len(data) > self.data_length:  # 数据超长则拒绝
        return False  # 把本层处理结果返回给调用方
    self.data_dict[scs_id] = data   # 存入 [byte0, byte1]，例如 [0x83, 0x07]
    self.is_param_changed = True    # 标记"参数变了，下次发送前要重新 makeParam()"
    return True  # 把本层处理结果返回给调用方
```

**两者共用的 `makeParam()`**（发送前把字典展开为连续字节列表）：

来源：scservo_sdk/group_sync_read.py:makeParam 与 scservo_sdk/group_sync_write.py:makeParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py:29-40
def makeParam(self):  # 定义本链路要说明的函数入口
    self.param = []  # 保存本链路后续步骤需要使用的中间状态或参数
    for scs_id in self.data_dict:  # 遍历本次链路涉及的元素
        self.param.append(scs_id)  # 只有 ID，如 [1, 2, 3, 4, 5, 6]

# group_sync_write.py:27-46
def makeParam(self):  # 定义本链路要说明的函数入口
    self.param = []  # 保存本链路后续步骤需要使用的中间状态或参数
    for scs_id in self.data_dict:  # 遍历本次链路涉及的元素
        self.param.append(scs_id)              # ID  # 调用下一层函数继续完成当前链路动作
        self.param.extend(self.data_dict[scs_id])  # + 数据，如 [1, 0x83, 0x07, 2, 0x09, 0x0A, ...]
```

> **核心设计**：`addParam` 只往 `self.data_dict` 字典里存数据，`txPacket()` 发送时检查 `is_param_changed` 标记，
> 如果变了就调 `makeParam()` 把字典展开成 `self.param` 连续字节列表，再交给 `ph.syncReadTx()` / `ph.syncWriteTxOnly()` 拼完整包发出去。

---

### 辅助：feetech.py — `_decode_sign()`

> 文件：`lerobot/src/lerobot/motors/feetech/feetech.py:351-359`

来源：lerobot/src/lerobot/motors/feetech/feetech.py:_decode_sign（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:  # 定义本链路要说明的函数入口
    """Sign-Magnitude 符号位解码：查编码表，对需要解码的寄存器进行解码。"""
    for id_ in ids_values:  # 遍历本次链路涉及的元素
        model = self._id_to_model(id_)  # 保存本链路后续步骤需要使用的中间状态或参数
        encoding_table = self.model_encoding_table.get(model)  # 保存本链路后续步骤需要使用的中间状态或参数
        # Present_Position 在 STS_SMS_SERIES_ENCODINGS_TABLE 里，sign_bit=15
        if encoding_table and data_name in encoding_table:  # 检查条件，决定是否进入该分支
            sign_bit = encoding_table[data_name]  # 保存本链路后续步骤需要使用的中间状态或参数
            ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)  # 保存本链路后续步骤需要使用的中间状态或参数

    return ids_values  # 把本层处理结果返回给调用方
```

---

### 辅助：encoding_utils.py — `decode_sign_magnitude()`

> 文件：`lerobot/src/lerobot/utils/encoding_utils.py:29-36`

来源：lerobot/src/lerobot/utils/encoding_utils.py:decode_sign_magnitude（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):  # 定义本链路要说明的函数入口
    """将 Feetech 寄存器原始值 → Python int（符号-幅值解码）。"""
    direction_bit = (encoded_value >> sign_bit_index) & 1   # 取 bit15
    magnitude_mask = (1 << sign_bit_index) - 1              # = 0x7FFF，取低15位
    magnitude = encoded_value & magnitude_mask  # 保存本链路后续步骤需要使用的中间状态或参数
    return -magnitude if direction_bit else magnitude  # 把本层处理结果返回给调用方
```

---

### 辅助：motors_bus.py — `_normalize()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1221-1277`

来源：lerobot/src/lerobot/motors/motors_bus.py:_normalize（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:  # 定义本链路要说明的函数入口
    """将原始编码值归一化为用户友好的范围。"""
    if not self.calibration:  # 检查条件，决定是否进入该分支
        raise RuntimeError(f"{self} has no calibration registered.")  # 调用下一层函数继续完成当前链路动作

    normalized_values = {}  # 保存本链路后续步骤需要使用的中间状态或参数
    for id_, val in ids_values.items():  # 遍历本次链路涉及的元素
        motor = self._id_to_name(id_)  # 保存本链路后续步骤需要使用的中间状态或参数
        min_ = self.calibration[motor].range_min  # 保存本链路后续步骤需要使用的中间状态或参数
        max_ = self.calibration[motor].range_max  # 保存本链路后续步骤需要使用的中间状态或参数
        drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode  # 保存本链路后续步骤需要使用的中间状态或参数
        if max_ == min_:  # 检查条件，决定是否进入该分支
            raise ValueError(...)  # 调用下一层函数继续完成当前链路动作

        bounded_val = min(max_, max(min_, val))  # 保存本链路后续步骤需要使用的中间状态或参数

        if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:  # 检查条件，决定是否进入该分支
            # 身体关节：((raw - min) / (max - min)) * 200 - 100
            # 例: shoulder_pan, raw=2301, min=1024, max=3072
            #     = ((2301-1024)/(3072-1024))*200 - 100 = 24.7
            norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100  # 保存本链路后续步骤需要使用的中间状态或参数
            normalized_values[id_] = -norm if drive_mode else norm  # 保存本链路后续步骤需要使用的中间状态或参数

        elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:  # 检查前一分支未命中后的备选条件
            # 夹爪：((raw - min) / (max - min)) * 100
            norm = ((bounded_val - min_) / (max_ - min_)) * 100  # 保存本链路后续步骤需要使用的中间状态或参数
            normalized_values[id_] = 100 - norm if drive_mode else norm  # 保存本链路后续步骤需要使用的中间状态或参数

        elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:  # 检查前一分支未命中后的备选条件
            mid = (min_ + max_) / 2  # 保存本链路后续步骤需要使用的中间状态或参数
            max_res = self.model_resolution_table[self._id_to_model(id_)] - 1  # 保存本链路后续步骤需要使用的中间状态或参数
            normalized_values[id_] = (val - mid) * 360 / max_res  # 保存本链路后续步骤需要使用的中间状态或参数

    return normalized_values  # 把本层处理结果返回给调用方
```

`min` / `max` 来自标定文件 `~/.lerobot/calibration/robots/R12254705.json`：
示意代码：为说明链路整理，不作为逐字源码；已按当前源码语义核对。
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

来源：lerobot/src/lerobot/robots/so101_follower/so101_follower.py:send_action（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:  # 定义本链路要说明的函数入口
    """命令机械臂移动到目标关节配置（每帧调用一次）。"""
    if not self.is_connected:  # 检查条件，决定是否进入该分支
        raise DeviceNotConnectedError(f"{self} is not connected.")  # 调用下一层函数继续完成当前链路动作

    total_start = time.perf_counter()  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤1: 去掉键名的 ".pos" 后缀，转为总线期望的电机名格式
    # {"shoulder_pan.pos": -15.2} → {"shoulder_pan": -15.2}
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤2: 安全限幅（可选，本次 max_relative_target=None，跳过）
    read_ms = 0.0  # 保存本链路后续步骤需要使用的中间状态或参数
    if self.config.max_relative_target is not None:  # 检查条件，决定是否进入该分支
        read_start = time.perf_counter()  # 保存本链路后续步骤需要使用的中间状态或参数
        present_pos = self.bus.sync_read("Present_Position")  # 保存本链路后续步骤需要使用的中间状态或参数
        read_ms = (time.perf_counter() - read_start) * 1e3  # 保存本链路后续步骤需要使用的中间状态或参数
        goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}  # 保存本链路后续步骤需要使用的中间状态或参数
        goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤3: 批量写入目标位置到6个舵机的 Goal_Position 寄存器
    write_start = time.perf_counter()  # 保存本链路后续步骤需要使用的中间状态或参数
    self.bus.sync_write("Goal_Position", goal_pos)  # 调用下一层函数继续完成当前链路动作
    write_ms = (time.perf_counter() - write_start) * 1e3  # 保存本链路后续步骤需要使用的中间状态或参数

    # 步骤4: 返回实际写入的目标位置（加回 ".pos" 后缀）
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}  # 把本层处理结果返回给调用方
```

---

### 第2层：motors_bus.py — `sync_write()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1784-1834`

来源：lerobot/src/lerobot/motors/motors_bus.py:sync_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def sync_write(  # 定义本链路要说明的函数入口
    self,  # 本行参与当前链路的控制流或数据准备
    data_name: str,  # 本行参与当前链路的控制流或数据准备
    values: Value | dict[str, Value],  # 本行参与当前链路的控制流或数据准备
    *,  # 本行参与当前链路的控制流或数据准备
    normalize: bool = True,  # 保存本链路后续步骤需要使用的中间状态或参数
    num_retry: int = 0,  # 保存本链路后续步骤需要使用的中间状态或参数
) -> None:  # 本行参与当前链路的控制流或数据准备
    """向多个电机的同一寄存器同步写入（无响应包，速度快但可能丢包）。"""
    if not self.is_connected:  # 检查条件，决定是否进入该分支
        raise DeviceNotConnectedError(...)  # 调用下一层函数继续完成当前链路动作

    # 1) 规整化输入为 {电机ID: 值}
    # {"shoulder_pan": -15.2} → {1: -15.2, 2: 32.0, ..., 6: 72.0}
    ids_values = self._get_ids_values_dict(values)  # 保存本链路后续步骤需要使用的中间状态或参数
    models = [self._id_to_model(id_) for id_ in ids_values]  # 保存本链路后续步骤需要使用的中间状态或参数

    # 2) 校验地址一致性
    if self._has_different_ctrl_tables:  # 检查条件，决定是否进入该分支
        assert_same_address(self.model_ctrl_table, models, data_name)  # 调用下一层函数继续完成当前链路动作

    # 查控制表：Goal_Position → (42, 2)
    model = next(iter(models))  # 保存本链路后续步骤需要使用的中间状态或参数
    addr, length = get_address(self.model_ctrl_table, model, data_name)  # 保存本链路后续步骤需要使用的中间状态或参数

    # 3) 反归一化（-100~100 → 原始 tick 0~4095）
    if normalize and data_name in self.normalized_data:  # 检查条件，决定是否进入该分支
        ids_values = self._unnormalize(ids_values)  # 保存本链路后续步骤需要使用的中间状态或参数
    # 结果：{1: 1891, 2: 2713, 3: 2150, 4: 1946, 5: 2423, 6: 3436}

    # 4) Sign-Magnitude 符号位编码
    # Goal_Position 不在编码表里，直接跳过
    ids_values = self._encode_sign(data_name, ids_values)  # 保存本链路后续步骤需要使用的中间状态或参数

    # 5) 底层同步写入
    self._sync_write(addr, length, ids_values, num_retry=num_retry, raise_on_error=True, err_msg=...)  # 保存本链路后续步骤需要使用的中间状态或参数
```

---

### 第3层：motors_bus.py — `_sync_write()`（串口通信两阶段）

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1836-1893`
>
> 注意：`self.sync_writer` 是 `scs.GroupSyncWrite` 的实例（在 feetech.py:135 创建），
> 以下方法调用全部进入 scservo_sdk：
> - `self.sync_writer.clearParam()` / `addParam()` — 注册要写入的电机 ID 和数据
> - `self.sync_writer.txPacket()` — 把所有电机的数据打包成一条 Sync Write 广播包发出去，**不等待任何回包**

来源：lerobot/src/lerobot/motors/motors_bus.py:_sync_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _sync_write(  # 定义本链路要说明的函数入口
    self,  # 本行参与当前链路的控制流或数据准备
    addr: int,  # 本行参与当前链路的控制流或数据准备
    length: int,  # 本行参与当前链路的控制流或数据准备
    ids_values: dict[int, int],  # 本行参与当前链路的控制流或数据准备
    num_retry: int = 0,  # 保存本链路后续步骤需要使用的中间状态或参数
    raise_on_error: bool = True,  # 保存本链路后续步骤需要使用的中间状态或参数
    err_msg: str = "",  # 保存本链路后续步骤需要使用的中间状态或参数
) -> int:  # 本行参与当前链路的控制流或数据准备
    """底层同步写入实现（setup → tx 两阶段，不等待应答）。"""

    # === 阶段1: Setup — 配置 scservo_sdk 的 GroupSyncWrite 对象 ===
    # 告诉 SDK：写哪个寄存器（addr=42），写几个字节（length=2），每个电机写什么值
    # 内部做了4件事：
    #   1. clearParam()  — 清空上次的参数
    #   2. start_address = addr   — 告诉 SDK 目标寄存器起始地址（42 = Goal_Position）
    #   3. data_length = length   — 告诉 SDK 每个电机要写几个字节（2字节）
    #   4. addParam(id_, data) × 6 — 把每个电机的 ID + 小端序字节数据注册进去
    #      例如 ID=1, value=1891 → data=[0x83, 0x07]（小端序）
    self._setup_sync_writer(ids_values, addr, length)  # 调用下一层函数继续完成当前链路动作

    # === 阶段2: Tx — 发送广播写包（主要耗时 ~1~3ms） ===
    # self.sync_writer.txPacket() 进入 scservo_sdk，做了一件事：
    #   把上面 setup 注册的所有 (ID, data) 打包成一条 Sync Write 指令包（指令码 0x83），
    #   通过 port_handler.writePort() 发到串口，**不等待任何回包**，直接返回
    for n_try in range(1 + num_retry):  # 遍历本次链路涉及的元素
        comm = self.sync_writer.txPacket()  # 进入协议发送流程，后续会组帧并写串口
        if self._is_comm_success(comm):  # 检查条件，决定是否进入该分支
            break  # 本行参与当前链路的控制流或数据准备

    if not self._is_comm_success(comm) and raise_on_error:  # 检查条件，决定是否进入该分支
        raise ConnectionError(...)  # 调用下一层函数继续完成当前链路动作

    return comm  # 把本层处理结果返回给调用方
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

来源：lerobot/src/lerobot/motors/motors_bus.py:_setup_sync_writer（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:  # 定义本链路要说明的函数入口
    """配置 scservo_sdk 的 GroupSyncWrite：告诉 SDK 写哪个寄存器、写几个字节、每个电机写什么值。"""
    self.sync_writer.clearParam()  # 清理上一轮注册参数，避免旧 ID 或旧数据残留
    self.sync_writer.start_address = addr     # 42（Goal_Position）  # 保存本链路后续步骤需要使用的中间状态或参数
    self.sync_writer.data_length = length     # 2（2字节）
    for id_, value in ids_values.items():  # 遍历本次链路涉及的元素
        data = self._serialize_data(value, length)  # 保存本链路后续步骤需要使用的中间状态或参数
        # _serialize_data(1891, 2) → [0x83, 0x07]（小端：低字节先发）
        self.sync_writer.addParam(id_, data)  # 把目标电机注册到同步读写参数表
```

---

### 辅助：motors_bus.py — `_unnormalize()`

> 文件：`lerobot/src/lerobot/motors/motors_bus.py:1279-1317`

来源：lerobot/src/lerobot/motors/motors_bus.py:_unnormalize（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:  # 定义本链路要说明的函数入口
    """将归一化值反向转换为电机原始编码值。"""
    if not self.calibration:  # 检查条件，决定是否进入该分支
        raise RuntimeError(f"{self} has no calibration registered.")  # 调用下一层函数继续完成当前链路动作

    unnormalized_values = {}  # 保存本链路后续步骤需要使用的中间状态或参数
    for id_, val in ids_values.items():  # 遍历本次链路涉及的元素
        motor = self._id_to_name(id_)  # 保存本链路后续步骤需要使用的中间状态或参数
        min_ = self.calibration[motor].range_min  # 保存本链路后续步骤需要使用的中间状态或参数
        max_ = self.calibration[motor].range_max  # 保存本链路后续步骤需要使用的中间状态或参数
        drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode  # 保存本链路后续步骤需要使用的中间状态或参数
        if max_ == min_:  # 检查条件，决定是否进入该分支
            raise ValueError(...)  # 调用下一层函数继续完成当前链路动作

        if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:  # 检查条件，决定是否进入该分支
            # 身体关节：norm=-15.2 → raw=1891
            # raw = int(((val+100)/200) * (max-min) + min)
            val = -val if drive_mode else val  # 保存本链路后续步骤需要使用的中间状态或参数
            bounded_val = min(100.0, max(-100.0, val))  # 保存本链路后续步骤需要使用的中间状态或参数
            unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)  # 保存本链路后续步骤需要使用的中间状态或参数

        elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:  # 检查前一分支未命中后的备选条件
            # 夹爪：val=72.0 → raw=3436
            val = 100 - val if drive_mode else val  # 保存本链路后续步骤需要使用的中间状态或参数
            bounded_val = min(100.0, max(0.0, val))  # 保存本链路后续步骤需要使用的中间状态或参数
            unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)  # 保存本链路后续步骤需要使用的中间状态或参数

        elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:  # 检查前一分支未命中后的备选条件
            mid = (min_ + max_) / 2  # 保存本链路后续步骤需要使用的中间状态或参数
            max_res = self.model_resolution_table[self._id_to_model(id_)] - 1  # 保存本链路后续步骤需要使用的中间状态或参数
            unnormalized_values[id_] = int((val * max_res / 360) + mid)  # 保存本链路后续步骤需要使用的中间状态或参数

    return unnormalized_values  # 把本层处理结果返回给调用方
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

来源：lerobot/src/lerobot/motors/feetech/feetech.py:_encode_sign（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:  # 定义本链路要说明的函数入口
    """Sign-Magnitude 符号位编码：查编码表，对需要的寄存器进行编码。"""
    for id_ in ids_values:  # 遍历本次链路涉及的元素
        model = self._id_to_model(id_)  # 保存本链路后续步骤需要使用的中间状态或参数
        encoding_table = self.model_encoding_table.get(model)  # 保存本链路后续步骤需要使用的中间状态或参数
        # Goal_Position 不在编码表里，直接跳过不做任何处理
        if encoding_table and data_name in encoding_table:  # 检查条件，决定是否进入该分支
            sign_bit = encoding_table[data_name]  # 保存本链路后续步骤需要使用的中间状态或参数
            ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)  # 保存本链路后续步骤需要使用的中间状态或参数

    return ids_values  # 把本层处理结果返回给调用方
```

---

### 辅助：feetech.py — `_split_into_byte_chunks()`（小端序序列化）

> 文件：`lerobot/src/lerobot/motors/feetech/feetech.py:69-83`
>
> `import scservo_sdk as scs` 在函数内部导入，
> `scs.SCS_LOBYTE` / `scs.SCS_HIBYTE` 是 SDK 提供的字节拆分工具函数：
> - `SCS_LOBYTE(1891)` = `1891 & 0xFF` = `0x83`（低字节）
> - `SCS_HIBYTE(1891)` = `1891 >> 8` = `0x07`（高字节）

来源：lerobot/src/lerobot/motors/feetech/feetech.py:_split_into_byte_chunks（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _split_into_byte_chunks(value: int, length: int) -> list[int]:  # 定义本链路要说明的函数入口
    """将整数值按小端序拆分为字节列表（Feetech 使用小端序）。"""
    import scservo_sdk as scs  # 本行参与当前链路的控制流或数据准备

    if length == 1:  # 检查条件，决定是否进入该分支
        data = [value]  # 保存本链路后续步骤需要使用的中间状态或参数
    elif length == 2:  # 检查前一分支未命中后的备选条件
        # 1891 → [0x83, 0x07]
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]  # 保存本链路后续步骤需要使用的中间状态或参数
    elif length == 4:  # 检查前一分支未命中后的备选条件
        data = [  # 保存本链路后续步骤需要使用的中间状态或参数
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),  # 调用下一层函数继续完成当前链路动作
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),  # 调用下一层函数继续完成当前链路动作
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),  # 调用下一层函数继续完成当前链路动作
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),  # 调用下一层函数继续完成当前链路动作
        ]
    return data  # 把本层处理结果返回给调用方
```

---

### 辅助：encoding_utils.py — `encode_sign_magnitude()`

> 文件：`lerobot/src/lerobot/utils/encoding_utils.py:16-26`

来源：lerobot/src/lerobot/utils/encoding_utils.py:encode_sign_magnitude（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def encode_sign_magnitude(value: int, sign_bit_index: int):  # 定义本链路要说明的函数入口
    """将 Python int → Feetech 寄存器原始值（符号-幅值编码）。"""
    max_magnitude = (1 << sign_bit_index) - 1  # 保存本链路后续步骤需要使用的中间状态或参数
    magnitude = abs(value)  # 保存本链路后续步骤需要使用的中间状态或参数
    if magnitude > max_magnitude:  # 检查条件，决定是否进入该分支
        raise ValueError(...)  # 调用下一层函数继续完成当前链路动作

    direction_bit = 1 if value < 0 else 0  # 保存本链路后续步骤需要使用的中间状态或参数
    return (direction_bit << sign_bit_index) | magnitude  # 把本层处理结果返回给调用方
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

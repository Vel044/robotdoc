# SO101 Follower 读取 ACM0 数据完整流程（带传入传出详细版）

---

## 调用链总览（含传入传出）

```
record.py:339
  调用: robot.get_observation()
  传入: 无
  传出: obs = {"shoulder_pan.pos": float, ..., "handeye": ndarray, ...}
  │
  └─ SO101Follower.get_observation()          so101_follower.py:249
       传入: 无（self 有 self.bus / self.cameras）
       传出: {"shoulder_pan.pos": float, ..., "handeye": ndarray}
       │
       ├─ self.bus.sync_read("Present_Position")
       │    传入: data_name="Present_Position", motors=None（即全部6个）
       │    传出: {"shoulder_pan": float, "shoulder_lift": float, ...}  ← 归一化后
       │    │
       │    └─ MotorsBus._sync_read(addr=56, length=2, motor_ids=[1,2,3,4,5,6])
       │         传入: addr=56, length=2, motor_ids=[1,2,3,4,5,6]
       │         传出: ({1: int, 2: int, 3: int, 4: int, 5: int, 6: int}, comm_code)
       │         │
       │         ├─ _setup_sync_reader([1,2,3,4,5,6], 56, 2)
       │         │    传入: motor_ids=[1,2,3,4,5,6], addr=56, length=2
       │         │    传出: 无（修改 sync_reader 状态）
       │         │
       │         └─ sync_reader.txRxPacket()
       │              传入: 无（参数已在 setup 阶段写入 sync_reader）
       │              传出: int（COMM_SUCCESS=0 或错误码）
       │              │
       │              ├─ txPacket()
       │              │    → syncReadTx(port, 56, 2, [1,2,3,4,5,6], 6)
       │              │    → txPacket(port, [FF FF FE 07 82 38 02 01 02 03 04 05 06 CHK])
       │              │    → port.writePort(bytes)  ★ write(fd, /dev/ttyACM0)
       │              │
       │              └─ rxPacket()
       │                   for scs_id in [1,2,3,4,5,6]:
       │                     readRx(port, scs_id, 2)
       │                     → rxPacket(port) → port.readPort(N) ★ read(fd, /dev/ttyACM0)
       │                     → data_dict[scs_id] = [byte_low, byte_high]
       │
       └─ cam.async_read()  ← 相机图像，另一条链路
```

---

## 一、`record.py:339` — 调用入口

```python
# lerobot/src/lerobot/record.py:339
obs = robot.get_observation()
# 传入：无
# 传出：obs —— dict[str, Any]
#   例：{"shoulder_pan.pos": 45.2, "shoulder_lift.pos": -30.5, ..., "handeye": np.ndarray}
```

`robot` 是 `SO101Follower` 实例，在 `record.py` 主循环每帧调用一次。

---

## 二、`SO101Follower.get_observation()` — so101_follower.py:249

```python
# lerobot/src/lerobot/robots/so101_follower/so101_follower.py:249
def get_observation(self) -> dict[str, Any]:
    # 传入：无
    # self.bus  = FeetechMotorsBus（管理 6 个 sts3215 舵机，port="/dev/ttyACM0"）
    # self.cameras = {"handeye": OpenCVCamera(...), "fixed": OpenCVCamera(...)}

    if not self.is_connected:
        raise DeviceNotConnectedError(...)

    # ── 读关节位置 ──────────────────────────────────────────────────────
    start = time.perf_counter()

    obs_dict = self.bus.sync_read("Present_Position")
    # 传入 sync_read：data_name="Present_Position", motors=None（默认读全部）
    # 传出 sync_dict：{"shoulder_pan": float, "shoulder_lift": float,
    #                  "elbow_flex": float, "wrist_flex": float,
    #                  "wrist_roll": float, "gripper": float}
    # 值是归一化后的浮点数：
    #   shoulder_pan~wrist_roll → RANGE_M100_100 模式 → 范围 [-100.0, 100.0]
    #   gripper                 → RANGE_0_100 模式   → 范围 [0.0, 100.0]

    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    # 重命名键，加 ".pos" 后缀：
    # {"shoulder_pan.pos": float, "shoulder_lift.pos": float, ...}

    # ── 读相机图像 ──────────────────────────────────────────────────────
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read()
        # cam_key = "handeye" 或 "fixed"
        # 传出：np.ndarray，shape=(360, 640, 3)，dtype=uint8

    return obs_dict
    # 最终传出：
    # {
    #   "shoulder_pan.pos":  float,   # [-100, 100]
    #   "shoulder_lift.pos": float,
    #   "elbow_flex.pos":    float,
    #   "wrist_flex.pos":    float,
    #   "wrist_roll.pos":    float,
    #   "gripper.pos":       float,   # [0, 100]
    #   "handeye":           ndarray, # shape=(360,640,3)
    #   "fixed":             ndarray,
    # }
```

### 电机配置（so101_follower.py:49-60）

```python
# so101_follower.py:45
def __init__(self, config: SO101FollowerConfig):
    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
    # config.use_degrees 默认 False → norm_mode_body = RANGE_M100_100

    self.bus = FeetechMotorsBus(
        port=self.config.port,               # "/dev/ttyACM0"
        motors={
            "shoulder_pan":  Motor(id=1, model="sts3215", norm_mode=RANGE_M100_100),
            "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=RANGE_M100_100),
            "elbow_flex":    Motor(id=3, model="sts3215", norm_mode=RANGE_M100_100),
            "wrist_flex":    Motor(id=4, model="sts3215", norm_mode=RANGE_M100_100),
            "wrist_roll":    Motor(id=5, model="sts3215", norm_mode=RANGE_M100_100),
            "gripper":       Motor(id=6, model="sts3215", norm_mode=RANGE_0_100),
        },
        calibration=self.calibration,        # 从标定文件加载的 range_min/range_max
    )
```

---

## 三、`MotorsBus.sync_read()` — motors_bus.py:1052

```python
# lerobot/src/lerobot/motors/motors_bus.py:1052
def sync_read(
    self,
    data_name: str,                         # "Present_Position"
    motors: str | list[str] | None = None,  # None → 读全部 6 个
    *,
    normalize: bool = True,                 # True → 归一化
    num_retry: int = 0,                     # 失败不重试
) -> dict[str, Value]:
    # 传入：data_name="Present_Position", motors=None

    # ── Step 1：电机名 → ID + 型号 ─────────────────────────────────────
    names  = self._get_motors_list(None)
    # 传出：["shoulder_pan", "shoulder_lift", "elbow_flex",
    #        "wrist_flex", "wrist_roll", "gripper"]

    ids    = [self.motors[m].id    for m in names]
    # 传出：[1, 2, 3, 4, 5, 6]

    models = [self.motors[m].model for m in names]
    # 传出：["sts3215", "sts3215", "sts3215", "sts3215", "sts3215", "sts3215"]

    # ── Step 2：查控制表，确定寄存器地址 + 字节长度 ──────────────────────
    model = next(iter(models))   # "sts3215"
    addr, length = get_address(self.model_ctrl_table, "sts3215", "Present_Position")
    # 查 tables.py → STS_SMS_SERIES_CONTROL_TABLE["Present_Position"] = (56, 2)
    # 传出：addr=56, length=2

    # ── Step 3：执行总线读取 ──────────────────────────────────────────────
    ids_values, _ = self._sync_read(
        addr=56, length=2, motor_ids=[1,2,3,4,5,6],
        num_retry=0, raise_on_error=True
    )
    # 传出：{1: int, 2: int, 3: int, 4: int, 5: int, 6: int}
    # 值为原始寄存器编码（符号幅度格式，未解码）

    # ── Step 4：符号位解码 ────────────────────────────────────────────────
    ids_values = self._decode_sign("Present_Position", ids_values)
    # 查 STS_SMS_SERIES_ENCODINGS_TABLE["Present_Position"] = 15（第15位为符号位）
    # 若 value & (1<<15) != 0：表示负数 → -(value & ~(1<<15))
    # sts3215 正常工作范围 0~4095，第15位通常为0，解码后值不变
    # 传出：{1: int, 2: int, ...}（带符号整数，仍在原始尺度）

    # ── Step 5：归一化 ────────────────────────────────────────────────────
    # normalize=True 且 "Present_Position" 在 normalized_data 中
    ids_values = self._normalize(ids_values)
    # shoulder_pan~wrist_roll：RANGE_M100_100
    #   norm = ((val - range_min) / (range_max - range_min)) * 200 - 100
    #   传出：float in [-100.0, 100.0]
    # gripper：RANGE_0_100
    #   norm = ((val - range_min) / (range_max - range_min)) * 100
    #   传出：float in [0.0, 100.0]
    # 传出：{1: float, 2: float, 3: float, 4: float, 5: float, 6: float}

    # ── Step 6：ID → 电机名 ───────────────────────────────────────────────
    return {self._id_to_name(id_): value for id_, value in ids_values.items()}
    # 传出：{"shoulder_pan": float, "shoulder_lift": float, ...}
```

### Present_Position 寄存器信息（tables.py:83）

```python
# lerobot/src/lerobot/motors/feetech/tables.py:83
"Present_Position": (56, 2),   # 地址=56 (0x38)，占 2 字节，只读
# 原始值范围：0 ~ 4095（sts3215 为 12-bit 分辨率，4096 步/圈）
# 符号位：第 15 位（STS_SMS_SERIES_ENCODINGS_TABLE，正常运动范围不触发）
```

---

## 四、`MotorsBus._sync_read()` — motors_bus.py:1105

```python
# lerobot/src/lerobot/motors/motors_bus.py:1105
def _sync_read(
    self,
    addr: int,              # 56
    length: int,            # 2
    motor_ids: list[int],   # [1, 2, 3, 4, 5, 6]
    *,
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> tuple[dict[int, int], int]:
    # 传入：addr=56, length=2, motor_ids=[1,2,3,4,5,6]

    # ── A. 配置 GroupSyncRead ──────────────────────────────────────────
    self._setup_sync_reader(motor_ids=[1,2,3,4,5,6], addr=56, length=2)
    # 传入：motor_ids=[1,2,3,4,5,6], addr=56, length=2
    # 传出：无（修改 self.sync_reader 状态）
    #   sync_reader.start_address = 56
    #   sync_reader.data_length = 2
    #   sync_reader.data_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}

    # ── B. 执行收发（主要耗时：5~20ms）────────────────────────────────
    comm = self.sync_reader.txRxPacket()
    # 传入：无
    # 传出：comm = COMM_SUCCESS(0) 或错误码

    # ── C. 从回包中提取各电机的值 ──────────────────────────────────────
    values = {id_: self.sync_reader.getData(id_, 56, 2) for id_ in [1,2,3,4,5,6]}
    # getData(scs_id, address=56, data_length=2) →
    #   SCS_MAKEWORD(data_dict[id_][0], data_dict[id_][1])
    #   = (byte_low & 0xFF) | ((byte_high & 0xFF) << 8)
    #   例：[0x00, 0x08] → 0x0800 = 2048
    # 传出：{1: 2048, 2: 1500, 3: 1800, 4: 1200, 5: 1000, 6: 500}

    return values, comm
    # 传出：({1: int, 2: int, 3: int, 4: int, 5: int, 6: int}, 0)
```

### `_setup_sync_reader`（motors_bus.py:1172）

```python
# motors_bus.py:1172
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
    # 传入：motor_ids=[1,2,3,4,5,6], addr=56, length=2
    # 传出：无

    self.sync_reader.clearParam()
    # data_dict 清空 → {}

    self.sync_reader.start_address = 56    # 寄存器地址
    self.sync_reader.data_length   = 2     # 每电机读取字节数

    for id_ in [1, 2, 3, 4, 5, 6]:
        self.sync_reader.addParam(id_)
        # data_dict[id_] = []  ← 预留接收槽
    # 执行后：data_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
```

---

## 五、`GroupSyncRead.txRxPacket()` — group_sync_read.py:114

```python
# scservo_sdk/group_sync_read.py:114
def txRxPacket(self):
    # 传入：无（参数已存在 self.start_address=56, self.data_length=2,
    #           self.data_dict={1:[],2:[],3:[],4:[],5:[],6:[]}）
    # 传出：int（COMM_SUCCESS=0 或错误码）

    result = self.txPacket()   # 发送广播读请求
    # 传出：COMM_SUCCESS=0

    if result != COMM_SUCCESS:
        return result

    return self.rxPacket()     # 接收 6 个电机各自的响应
    # 传出：COMM_SUCCESS=0（全部接收成功）
```

---

## 六、`GroupSyncRead.txPacket()` — group_sync_read.py:74（发送阶段）

```python
# scservo_sdk/group_sync_read.py:74
def txPacket(self):
    # 传入：无
    # 传出：int（COMM_SUCCESS=0 或 COMM_NOT_AVAILABLE=-9）

    if len(self.data_dict.keys()) == 0:
        return COMM_NOT_AVAILABLE

    if self.is_param_changed or not self.param:
        self.makeParam()
        # makeParam(): self.param = [1, 2, 3, 4, 5, 6]（6 个 ID）

    return self.ph.syncReadTx(
        port         = self.port,          # PortHandler("/dev/ttyACM0")
        start_address= 56,                 # self.start_address
        data_length  = 2,                  # self.data_length
        param        = [1, 2, 3, 4, 5, 6], # self.param
        param_length = 6,                  # len(data_dict) * 1 = 6
    )
    # 传出：COMM_SUCCESS=0
```

### `syncReadTx` 组包过程（protocol_packet_handler.py:431）

```python
# scservo_sdk/protocol_packet_handler.py:431
def syncReadTx(self, port, start_address=56, data_length=2, param=[1,2,3,4,5,6], param_length=6):
    # 传入：port, start_address=56, data_length=2, param=[1,2,3,4,5,6], param_length=6
    # 传出：COMM_SUCCESS=0

    txpacket = [0] * (6 + 8)   # param_length + 8 = 14 字节

    txpacket[PKT_ID]          = 0xFE  # 广播 ID
    txpacket[PKT_LENGTH]      = 6 + 4 # = 10（INST+ADDR+LEN+ID×6+CHK）
    txpacket[PKT_INSTRUCTION] = 0x82  # INST_SYNC_READ
    txpacket[PKT_PARAMETER0+0]= 56    # start_address
    txpacket[PKT_PARAMETER0+1]= 2     # data_length（每电机读取字节数）
    txpacket[PKT_PARAMETER0+2: PKT_PARAMETER0+8] = [1, 2, 3, 4, 5, 6]

    # 调用 txPacket 填 Header + Checksum 并写串口
    result = self.txPacket(port, txpacket)
    # → txpacket 最终：[0xFF, 0xFF, 0xFE, 0x0A, 0x82, 0x38, 0x02, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, CHK]
    #   索引:            0     1     2     3     4     5     6     7     8     9     10    11    12    13

    if result == COMM_SUCCESS:
        # 设置接收超时：(6 + data_length) * param_length = (6+2)*6 = 48 ms
        port.setPacketTimeout(48)

    return result   # COMM_SUCCESS=0
```

### 发送数据包内存布局

```
字节索引:  0     1     2     3     4     5     6     7     8     9    10    11    12    13
内容:    [FF]  [FF]  [FE]  [0A]  [82]  [38]  [02]  [01]  [02]  [03]  [04]  [05]  [06]  [CHK]
含义:     H0    H1   广播  LEN  INST  ADDR   SZ   ID1   ID2   ID3   ID4   ID5   ID6   校验

LEN = 0x0A = 10（INST1 + ADDR1 + SZ1 + ID×6 + CHK1 = 10）
ADDR = 0x38 = 56（Present_Position 地址）
SZ   = 0x02（每电机读2字节）

CHK = ~(0xFE + 0x0A + 0x82 + 0x38 + 0x02 + 0x01 + 0x02 + 0x03 + 0x04 + 0x05 + 0x06) & 0xFF
    = ~(0x1DF) & 0xFF = ~0xDF & 0xFF = 0x20
```

### `txPacket` 物理写串口（protocol_packet_handler.py:69）

```python
# scservo_sdk/protocol_packet_handler.py:69
def txPacket(self, port, txpacket):
    # 传入：port=PortHandler("/dev/ttyACM0"), txpacket=[0,0,0xFE,0x0A,...]（14元素）
    # 传出：COMM_SUCCESS=0

    # 1. 并发锁
    if port.is_using: return COMM_PORT_BUSY
    port.is_using = True

    # 2. 填包头
    txpacket[0] = 0xFF
    txpacket[1] = 0xFF

    # 3. 计算校验和：sum(txpacket[2:-1]) 取反截 8 位
    checksum = 0
    for idx in range(2, 14 - 1):   # 索引 2~12
        checksum += txpacket[idx]
    txpacket[13] = ~checksum & 0xFF

    # 4. 写串口
    port.clearPort()                      # ser.flush()
    written = port.writePort(txpacket)    # ★ ser.write(14字节) → write(fd, buf, 14)
    # 传出：written = 14（实际写入字节数）

    if 14 != written: return COMM_TX_FAIL
    return COMM_SUCCESS   # ← 不释放 is_using，rxPacket 负责释放
```

---

## 七、`GroupSyncRead.rxPacket()` — group_sync_read.py:91（接收阶段）

```python
# scservo_sdk/group_sync_read.py:91
def rxPacket(self):
    # 传入：无（data_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}）
    # 传出：COMM_SUCCESS=0

    self.last_result = False
    result = COMM_RX_FAIL

    # 逐个电机接收独立响应包
    for scs_id in [1, 2, 3, 4, 5, 6]:
        self.data_dict[scs_id], result, _ = self.ph.readRx(
            port    = self.port,   # PortHandler
            scs_id  = scs_id,      # 当前期望收到的电机 ID（1~6）
            length  = 2,           # 期望的数据字节数
        )
        # 传出：data_dict[scs_id] = [byte_low, byte_high]
        #        result = COMM_SUCCESS=0
        if result != COMM_SUCCESS:
            return result

    # 执行后：
    # data_dict = {
    #   1: [0x00, 0x08],   # 2048
    #   2: [0xDC, 0x05],   # 1500
    #   3: [0x08, 0x07],   # 1800
    #   4: [0xB0, 0x04],   # 1200
    #   5: [0xE8, 0x03],   # 1000
    #   6: [0xF4, 0x01],   # 500
    # }

    self.last_result = True
    return COMM_SUCCESS   # = 0
```

### `readRx` — 等待并解析单个电机响应包（protocol_packet_handler.py:262）

```python
# scservo_sdk/protocol_packet_handler.py:262
def readRx(self, port, scs_id, length):
    # 传入：port=PortHandler, scs_id=1（期望的电机ID）, length=2
    # 传出：(data=[byte_low, byte_high], result=0, error=0)

    data = []
    while True:
        rxpacket, result = self.rxPacket(port)
        # 传出：rxpacket=[0xFF,0xFF,0x01,0x04,0x00,0x00,0x08,CHK], result=0

        if result != COMM_SUCCESS or rxpacket[PKT_ID] == scs_id:
            break
        # 如果 rxpacket[2] == 1（scs_id），则是我们要的包，退出

    if result == COMM_SUCCESS and rxpacket[PKT_ID] == scs_id:
        error = rxpacket[PKT_ERROR]    # rxpacket[4] = 0x00（无错误）

        # 从 PKT_PARAMETER0（索引5）开始提取 length=2 字节
        data.extend(rxpacket[5: 5 + 2])
        # data = [rxpacket[5], rxpacket[6]] = [0x00, 0x08]

    return data, result, error
    # 传出：([0x00, 0x08], 0, 0)
```

### `rxPacket` — 分块读取并验证一帧（protocol_packet_handler.py:103）

```python
# scservo_sdk/protocol_packet_handler.py:103
def rxPacket(self, port):
    # 传入：port=PortHandler
    # 传出：(rxpacket=[0xFF,0xFF,0x01,0x04,0x00,D0,D1,CHK], result=COMM_SUCCESS)

    rxpacket = []
    wait_length = 6    # 最小包长（FF FF ID LEN ERR CHK）

    while True:
        rxpacket.extend(port.readPort(wait_length - len(rxpacket)))
        # ★ port.readPort(N) → ser.read(N) → read(fd, buf, N) 从 /dev/ttyACM0 读字节
        # 非阻塞：有多少返回多少，可能不足 N 字节

        if len(rxpacket) >= wait_length:
            # 搜索包头 0xFF 0xFF
            for idx in range(0, len(rxpacket)-1):
                if rxpacket[idx] == 0xFF and rxpacket[idx+1] == 0xFF:
                    break

            if idx == 0:  # 包头在起始位置
                # 更新精确包长：rxpacket[3]（LEN字段）+ 4
                # 对于 Present_Position 响应：LEN=4（ERR+D0+D1+CHK），wait_length=8
                if wait_length != rxpacket[PKT_LENGTH] + PKT_LENGTH + 1:
                    wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                    continue    # 继续读直到凑够 8 字节

                if len(rxpacket) < wait_length:
                    if port.isPacketTimeout():   # 超时检查
                        break
                    continue

                # 校验和验证
                checksum = 0
                for i in range(2, wait_length - 1):
                    checksum += rxpacket[i]
                checksum = ~checksum & 0xFF
                result = COMM_SUCCESS if rxpacket[-1] == checksum else COMM_RX_CORRUPT
                break
        else:
            if port.isPacketTimeout():
                break

    port.is_using = False   # ★ 释放并发锁
    return rxpacket, result
```

### 电机响应包内存布局（以电机1，Present_Position=2048为例）

```
字节索引:  0     1     2     3     4     5     6     7
内容:    [FF]  [FF]  [01]  [04]  [00]  [00]  [08]  [F2]
含义:     H0    H1   ID1   LEN   ERR   D_LOW D_HI  CHK

LEN = 0x04 = 4（ERR1 + D_LOW1 + D_HIGH1 + CHK1）
ERR = 0x00（无错误）
D_LOW  = 0x00（位置低字节）
D_HIGH = 0x08（位置高字节）

wait_length = LEN + PKT_LENGTH + 1 = 4 + 3 + 1 = 8

CHK = ~(0x01 + 0x04 + 0x00 + 0x00 + 0x08) & 0xFF
    = ~0x0D & 0xFF = 0xF2
```

---

## 八、`getData()` — 字节 → 整数（group_sync_read.py:146）

```python
# scservo_sdk/group_sync_read.py:146
def getData(self, scs_id, address, data_length):
    # 传入：scs_id=1, address=56, data_length=2
    # data_dict[1] = [0x00, 0x08]（接收阶段填入）
    # 传出：int（原始位置值）

    if not self.isAvailable(scs_id, address, data_length):
        return 0

    # data_length=2，取 data_dict[1][address - start_address] = data_dict[1][0]
    if data_length == 2:
        return SCS_MAKEWORD(
            self.data_dict[1][0],   # a = 0x00（低字节，索引 56-56=0）
            self.data_dict[1][1],   # b = 0x08（高字节，索引 56-56+1=1）
        )
        # SCS_MAKEWORD(0x00, 0x08) = (0x00 & 0xFF) | ((0x08 & 0xFF) << 8)
        #                          = 0x0000 | 0x0800 = 2048

    # 传出：2048
```

### `SCS_MAKEWORD` 字节序（scservo_def.py）

```python
# scservo_sdk/scservo_def.py
SCS_END = 0   # Protocol 0 = 小端模式

def SCS_MAKEWORD(a, b):
    # 传入：a=低字节, b=高字节
    # Protocol 0（小端）：a 在低位，b 在高位
    if SCS_END == 0:
        return (a & 0xFF) | ((b & 0xFF) << 8)
    # 传出：16-bit 整数

# 例：a=0x00, b=0x08 → 0x0800 = 2048
# 例：a=0xDC, b=0x05 → 0x05DC = 1500
```

---

## 九、`_decode_sign()` — 符号位解码（feetech.py:351）

```python
# lerobot/src/lerobot/motors/feetech/feetech.py:351
def _decode_sign(self, data_name, ids_values):
    # 传入：data_name="Present_Position", ids_values={1:2048, 2:1500, ...}
    # 传出：{1: int, 2: int, ...}（带符号，通常与输入相同）

    for id_ in ids_values:
        model = self._id_to_model(id_)   # "sts3215"
        encoding_table = self.model_encoding_table.get(model)
        # encoding_table = STS_SMS_SERIES_ENCODINGS_TABLE
        # = {"Homing_Offset":11, "Goal_Velocity":15,
        #    "Present_Velocity":15, "Present_Position":15}

        if encoding_table and "Present_Position" in encoding_table:
            sign_bit = 15   # 第 15 位（最高位，uint16 符号位）

            value = ids_values[id_]    # 例：2048 = 0x0800
            if value & (1 << 15):      # 0x0800 & 0x8000 = 0 → 正数
                ids_values[id_] = -(value & ~(1 << 15))
            # else: 值不变，正数直接返回

    return ids_values
    # 传出：{1: 2048, 2: 1500, 3: 1800, 4: 1200, 5: 1000, 6: 500}
    # sts3215 位置范围 0~4095，第15位永远为0，解码后与输入相同
```

---

## 十、`_normalize()` — 原始值 → 用户浮点值（motors_bus.py:777）

```python
# lerobot/src/lerobot/motors/motors_bus.py:777
def _normalize(self, ids_values):
    # 传入：ids_values={1:2048, 2:1500, 3:1800, 4:1200, 5:1000, 6:500}
    # 传出：{1: float, 2: float, ..., 6: float}

    for id_, val in ids_values.items():
        motor = self._id_to_name(id_)                    # 例："shoulder_pan"
        min_  = self.calibration[motor].range_min        # 例：100
        max_  = self.calibration[motor].range_max        # 例：3900
        drive_mode = self.calibration[motor].drive_mode  # bool，是否方向反转

        bounded_val = min(max_, max(min_, val))          # 限幅到 [min_, max_]

        if norm_mode is RANGE_M100_100:   # shoulder_pan ~ wrist_roll
            norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
            # 例：val=2048, min=100, max=3900
            # bounded = 2048
            # norm = ((2048-100)/(3900-100)) * 200 - 100
            #      = (1948/3800) * 200 - 100
            #      = 0.5126 * 200 - 100 = 2.53
            normalized_values[id_] = -norm if drive_mode else norm
            # drive_mode=False → 2.53

        elif norm_mode is RANGE_0_100:    # gripper
            norm = ((bounded_val - min_) / (max_ - min_)) * 100
            normalized_values[id_] = 100 - norm if drive_mode else norm

    return normalized_values
    # 传出：{1: 2.53, 2: -30.5, 3: 60.0, 4: -15.3, 5: 0.0, 6: 80.0}
```

---

## 十一、数据变换全链路汇总

```
/dev/ttyACM0 响应原始字节
    [FF FF 01 04 00  00 08  F2]
                     ↓
    port.readPort() → bytes → list[int] = [0x00, 0x08]
                     ↓
    getData() → SCS_MAKEWORD(0x00, 0x08) = 2048   （原始 uint16 编码值）
                     ↓
    _decode_sign()   → 2048   （第15位=0，正数，值不变）
                     ↓
    _normalize() RANGE_M100_100, min=100, max=3900
                 norm = ((2048-100)/(3900-100))*200 - 100 = 2.53
                     ↓
    sync_read 返回  {"shoulder_pan": 2.53}
                     ↓
    get_observation 加后缀  {"shoulder_pan.pos": 2.53}
                     ↓
    record.py  obs["shoulder_pan.pos"] = 2.53
```

---

## 十二、关键文件与行号索引

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 调用入口 | [record.py](lerobot/src/lerobot/record.py) | 339 | `obs = robot.get_observation()` |
| 观测方法 | [so101_follower.py](lerobot/src/lerobot/robots/so101_follower/so101_follower.py) | 249 | `get_observation()` |
| 电机配置 | [so101_follower.py](lerobot/src/lerobot/robots/so101_follower/so101_follower.py) | 49 | 6电机 ID 1~6，sts3215 |
| 同步读公开接口 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | 1052 | `sync_read()` |
| 同步读内部实现 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | 1105 | `_sync_read()` |
| 配置读取器 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | 1172 | `_setup_sync_reader()` |
| 控制表 | [tables.py](lerobot/src/lerobot/motors/feetech/tables.py) | 83 | `Present_Position=(56,2)` |
| 符号位表 | [tables.py](lerobot/src/lerobot/motors/feetech/tables.py) | 211 | `Present_Position=15` |
| SDK 收发 | [group_sync_read.py](scservo_sdk/group_sync_read.py) | 114 | `txRxPacket()` |
| 发包 | [protocol_packet_handler.py](scservo_sdk/protocol_packet_handler.py) | 431 | `syncReadTx()` |
| 写串口 | [protocol_packet_handler.py](scservo_sdk/protocol_packet_handler.py) | 96 | `port.writePort()` → `write(fd)` |
| 收包循环 | [protocol_packet_handler.py](scservo_sdk/protocol_packet_handler.py) | 103 | `rxPacket()` |
| 读串口 | [port_handler.py](scservo_sdk/port_handler.py) | 57 | `readPort()` → `read(fd)` |
| 字节→整数 | [group_sync_read.py](scservo_sdk/group_sync_read.py) | 146 | `getData()` → `SCS_MAKEWORD` |
| 符号解码 | [feetech.py](lerobot/src/lerobot/motors/feetech/feetech.py) | 351 | `_decode_sign()` |
| 归一化 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | 777 | `_normalize()` |

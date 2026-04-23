# scservo_sdk 写链路：`_setup_sync_writer` → `txPacket` → `ser.write()`

以写入 `Goal_Position`（地址 42，2 字节，6 个 STS-3215 电机）为例，完整追踪从 Lerobot 入口到 `ser.write()` 的每一步。写链路**不等任何回包**，发完即返回。

---

## 第一步：`_setup_sync_writer` — 告诉 SDK 写什么、写谁、写多少

**调用位置**：`motors_bus.py: _sync_write()`

```python
def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
    self.sync_writer.clearParam()
    self.sync_writer.start_address = addr   # 42（Goal_Position）
    self.sync_writer.data_length = length   # 2（2字节）
    for id_, value in ids_values.items():
        data = self._serialize_data(value, length)
        # _serialize_data(1891, 2) → [0x83, 0x07]（小端：低字节先）
        self.sync_writer.addParam(id_, data)
```

`self.sync_writer` 是 `scs.GroupSyncWrite` 实例，在 `feetech.py.__init__` 里创建：

```python
# feetech.py
self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
```

### `clearParam()` — 清空上次注册的电机

```python
# group_sync_write.py
def clearParam(self):
    self.data_dict.clear()
    # data_dict = {}
```

`data_dict` 是写链路的核心数据结构：
```
data_dict = {
    1: [0x83, 0x07],   # addParam 后存入，txPacket 发送时读取
    2: [0x99, 0x0A],
    ...
}
```

### `_serialize_data(value, length)` — 整数拆成小端字节列表

这一步在 setup 里调用，把每个电机的目标位置整数拆成 SDK 需要的字节列表。

```python
# motors_bus.py（调用 scservo_sdk 宏）
def _serialize_data(self, value: int, length: int) -> list[int]:
    import scservo_sdk as scs

    if length == 1:
        return [value]
    elif length == 2:
        return [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        # SCS_LOBYTE(1891) = 1891 & 0xFF = 0x83（低字节）
        # SCS_HIBYTE(1891) = (1891 >> 8) & 0xFF = 0x07（高字节）
    elif length == 4:
        return [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),   # 低16位的低字节
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),    # 低16位的高字节
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),    # 高16位的低字节
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),    # 高16位的高字节
        ]
```

`SCS_LOBYTE` / `SCS_HIBYTE` 在 `scservo_def.py` 里定义，受 `SCS_END` 字节序控制：
```python
def SCS_LOBYTE(w):
    if SCS_END == 0:    # Protocol 0 小端
        return w & 0xFF
    else:               # Protocol 1 大端
        return (w >> 8) & 0xFF
```

### `addParam(id_, data)` — 注册一个电机的 ID 和数据

```python
# group_sync_write.py
def addParam(self, scs_id, data):
    if scs_id in self.data_dict:        # 已存在则拒绝（clearParam 后不触发）
        return False
    if len(data) > self.data_length:    # 数据超长则拒绝
        return False
    self.data_dict[scs_id] = data       # 存入，例如 data_dict[1] = [0x83, 0x07]
    self.is_param_changed = True        # 标记"下次发送前要重新 makeParam()"
    return True
```

与读链路的 `addParam` 相比：读链路只存 `data_dict[id] = []`（空列表，等接收填充），写链路存 `data_dict[id] = data`（已知数据）。

6 次调用后 `data_dict` 状态（以目标位置 `{1:1891, 2:2713, 3:2150, 4:1946, 5:2423, 6:3436}` 为例）：
```
data_dict = {
    1: [0x83, 0x07],   # 1891 = 0x0783，小端：[0x83, 0x07]
    2: [0x99, 0x0A],   # 2713 = 0x0A99，小端：[0x99, 0x0A]
    3: [0x66, 0x08],   # 2150 = 0x0866，小端：[0x66, 0x08]
    4: [0x9A, 0x07],   # 1946 = 0x079A，小端：[0x9A, 0x07]
    5: [0x77, 0x09],   # 2423 = 0x0977，小端：[0x77, 0x09]
    6: [0x6C, 0x0D],   # 3436 = 0x0D6C，小端：[0x6C, 0x0D]
}
is_param_changed = True
```

---

## 第二步：`sync_writer.txPacket()` — 组装并发送 Sync Write 广播包

**调用位置**：`motors_bus.py: _sync_write()`

```python
comm = self.sync_writer.txPacket()
```

```python
# group_sync_write.py
def txPacket(self):
    if not self.data_dict:
        return COMM_NOT_AVAILABLE

    if self.is_param_changed or not self.param:
        self.makeParam()   # 把 data_dict 展开为连续字节列表

    # param_length = 电机数 × (1字节ID + data_length字节数据) = 6 × (1+2) = 18
    return self.ph.syncWriteTxOnly(
        self.port,
        self.start_address,            # 42
        self.data_length,              # 2
        self.param,                    # [1,0x83,0x07, 2,0x99,0x0A, ...]
        len(self.data_dict) * (1 + self.data_length)  # 6 × 3 = 18
    )
```

### `makeParam()` — 展开 data_dict 为 `[id, b0, b1, id, b0, b1, ...]`

```python
# group_sync_write.py
def makeParam(self):
    self.param = []
    for scs_id in self.data_dict:
        self.param.append(scs_id)                # 先放 ID（1字节）
        self.param.extend(self.data_dict[scs_id])  # 再放数据（data_length字节）
```

展开结果：
```
param = [
    1, 0x83, 0x07,    # 电机1：ID=1, 位置=1891
    2, 0x99, 0x0A,    # 电机2：ID=2, 位置=2713
    3, 0x66, 0x08,    # 电机3
    4, 0x9A, 0x07,    # 电机4
    5, 0x77, 0x09,    # 电机5
    6, 0x6C, 0x0D,    # 电机6
]
# 共 6 × 3 = 18 字节
```

注意：读链路的 `makeParam` 只有 `[id1, id2, ...]`，写链路的 `makeParam` 是 `[id1, b0, b1, id2, b0, b1, ...]`。这是 Sync Read 和 Sync Write 的关键结构差异。

### `syncWriteTxOnly()` — 组装 Sync Write 指令包

```python
# protocol_packet_handler.py
def syncWriteTxOnly(self, port, start_address, data_length, param, param_length):
    # 包总长 = param_length(18) + 8（FF FF ID LEN INST ADDR DLEN CHK）
    txpacket = [0] * (param_length + 8)

    txpacket[PKT_ID]          = BROADCAST_ID     # 0xFE，广播
    txpacket[PKT_LENGTH]      = param_length + 4  # LEN = 18 + 4 = 22
    txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE   # 0x83
    txpacket[PKT_PARAMETER0 + 0] = start_address  # 42，0x2A
    txpacket[PKT_PARAMETER0 + 1] = data_length    # 2

    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param  # 18字节参数

    # txRxPacket 检测到 PKT_ID=BROADCAST_ID → 发完即返回，不等响应
    _, result, _ = self.txRxPacket(port, txpacket)
    return result
```

组装后的原始帧（十六进制）：
```
FF FF FE 16 83 2A 02  01 83 07  02 99 0A  03 66 08  04 9A 07  05 77 09  06 6C 0D  CS
│  │  │  │  │  │  │   └─────────────────────────────────────────────────────────── ID+数据 × 6
│  │  │  │  │  │  └── data_length = 2
│  │  │  │  │  └───── start_address = 0x2A = 42
│  │  │  │  └──────── INST_SYNC_WRITE = 0x83
│  │  │  └─────────── LEN = 0x16 = 22（param_length=18 + 4）
│  │  └────────────── BROADCAST_ID = 0xFE
└──┴───────────────── Header 0xFF 0xFF
CS = (~sum(0xFE+0x16+0x83+0x2A+0x02+...所有ID和数据字节)) & 0xFF
```

总帧长 = 22 + 4 = **26 字节**。

### `txRxPacket()` → `txPacket()` — 广播检测，发完即返回

```python
# protocol_packet_handler.py
def txRxPacket(self, port, txpacket):
    result = self.txPacket(port, txpacket)   # 计算校验和、写串口
    if result != COMM_SUCCESS:
        return None, result, 0

    # 广播包：PKT_ID == 0xFE → 发完直接释放锁返回，不等任何回包
    if txpacket[PKT_ID] == BROADCAST_ID:
        port.is_using = False   # 释放并发锁
        return None, result, 0

    # 单播包才会走下面的 rxPacket（写链路不走这里）
    ...
```

### `txPacket()` — 填校验和、写串口

```python
# protocol_packet_handler.py
def txPacket(self, port, txpacket):
    if port.is_using:
        return COMM_PORT_BUSY
    port.is_using = True   # 加并发锁

    total_packet_length = txpacket[PKT_LENGTH] + 4  # 22 + 4 = 26 字节

    txpacket[PKT_HEADER0] = 0xFF
    txpacket[PKT_HEADER1] = 0xFF

    checksum = 0
    for idx in range(2, total_packet_length - 1):
        checksum += txpacket[idx]
    txpacket[total_packet_length - 1] = ~checksum & 0xFF

    port.clearPort()   # ser.flush()，避免粘包
    written = port.writePort(txpacket)   # ★ ser.write(packet)
    if total_packet_length != written:
        port.is_using = False
        return COMM_TX_FAIL
    # 注意：此处不释放 is_using，由 txRxPacket 检测 BROADCAST_ID 后释放
    return COMM_SUCCESS
```

`writePort` 最终调用：
```python
# port_handler.py
def writePort(self, packet):
    return self.ser.write(packet)   # pyserial → write(fd, buf, len) 系统调用
```

26 字节一次性写入串口缓冲区，内核通过 UART 以 1Mbps 发出。6 个舵机同时收到，各自更新 `Goal_Position`（地址 42，写 2 字节），驱动电机转动。**主机不等任何回包，函数直接返回。**

---

## 完整调用链一览

```
motors_bus._sync_write(addr=42, length=2, ids_values={1:1891,...,6:3436})
│   # [motors_bus.py] Lerobot 入口：向 6 个电机地址 42 写入 2 字节目标位置
│
├── _setup_sync_writer(ids_values, addr=42, length=2)
│   │   # [motors_bus.py] 【注册阶段】把每个电机的目标值序列化后注册到 GroupSyncWrite
│   ├── sync_writer.clearParam()
│   │   └── data_dict.clear()              # [group_sync_write.py] 清空上次残留的电机 ID 和数据
│   ├── sync_writer.start_address = 42     # [group_sync_write.py] 记录寄存器起始地址（Goal_Position）
│   ├── sync_writer.data_length = 2        # [group_sync_write.py] 记录每个电机写入的字节数
│   └── for id_, value in ids_values:
│       ├── _serialize_data(value=1891, length=2)   # [motors_bus.py]
│       │   └── [scs.SCS_LOBYTE(1891), scs.SCS_HIBYTE(1891)] = [0x83, 0x07]
│       │       # [scservo_def.py] 整数拆成小端两字节：低字节 0x83，高字节 0x07
│       └── sync_writer.addParam(id_=1, data=[0x83, 0x07])  # [group_sync_write.py]
│           └── data_dict[1] = [0x83, 0x07]；is_param_changed = True
│               # 存入字节数据，标记 param 需要重新生成
│       （重复 6 次）
│
└── sync_writer.txPacket()                 # [group_sync_write.py]
    │   # 【发送阶段】把所有电机数据打包成一帧广播写指令发出，不等回包
    ├── makeParam()                        # [group_sync_write.py]
    │   └── self.param = [1,0x83,0x07, 2,0x99,0x0A, ..., 6,0x6C,0x0D]
    │       # 展开 data_dict 为 [id, data_L, data_H, ...] 交错排列，共 18 字节（6×3）
    └── ph.syncWriteTxOnly(port, 42, 2, param, 18)  # [protocol_packet_handler.py]
        │   # 组装 Sync Write 广播帧（26字节）并写串口
        └── txRxPacket(port, txpacket)     # [protocol_packet_handler.py]
            └── txPacket(port, txpacket)   # [protocol_packet_handler.py] list[int]，26 字节
                ├── port.is_using = True   # 半双工加锁，发送期间禁止接收
                ├── txpacket[0]=0xFF, txpacket[1]=0xFF   # 填包头
                ├── checksum = ~sum(0xFE..所有数据) & 0xFF   # 计算校验和
                ├── port.clearPort()  → ser.flush()   # [port_handler.py] 冲刷缓冲，避免粘包
                └── port.writePort(txpacket)  → ser.write()  ★ 写串口  # [port_handler.py]
                    物理帧: FF FF FE 16 83 2A 02 [id+data×6] CS
            # 发完后检测 PKT_ID==BROADCAST_ID（广播帧），确认无需等回包
            └── port.is_using = False      # [port_handler.py] 释放锁，写链路到此结束
```

---

## 关键细节

| 细节 | 说明 |
|---|---|
| `makeParam` 结构 | `[id1, b0, b1, id2, b0, b1, ...]`（与读链路的 `[id1, id2, ...]` 不同） |
| 广播不等响应 | `txRxPacket` 检测到 `PKT_ID==0xFE` 后发完即释放锁返回 |
| 小端序 | `SCS_LOBYTE(v)=v&0xFF`（低字节先发），写入舵机寄存器时低地址存低字节 |
| `is_using` 锁 | `txPacket` 加锁，`txRxPacket` 检测广播后释放；写链路从不调用 `rxPacket` |
| 丢包风险 | 广播写无应答，主机不知道是否成功；总线噪声导致丢包时舵机静默不动 |
| 发送耗时 | 26字节 × 0.01ms/byte = 0.26ms 传输时间，加上 USB 延迟约 1~3ms 总耗时 |
| 6个舵机同步 | 广播帧所有舵机同时收到，同一时刻更新各自的 Goal_Position，同步精度高 |

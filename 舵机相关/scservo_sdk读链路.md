# scservo_sdk 读链路：`_setup_sync_reader` → `txRxPacket` → `getData`

以读取 `Present_Position`（地址 56，2 字节，6 个 STS-3215 电机）为例，完整追踪从 Lerobot 入口到 `ser.read()` 的每一步。

---

## 1. 第一步：`_setup_sync_reader` — 告诉 SDK 读什么、读谁

**调用位置**：`motors_bus.py: _sync_read()`

```python
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
    self.sync_reader.clearParam()
    self.sync_reader.start_address = addr   # 56（Present_Position）
    self.sync_reader.data_length = length   # 2（2字节）
    for id_ in motor_ids:
        self.sync_reader.addParam(id_)      # 依次注册 ID=1,2,3,4,5,6
```

`self.sync_reader` 是 `scs.GroupSyncRead` 实例，在 `feetech.py.__init__` 里创建：

```python
# feetech.py
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
```

### 1.1 `clearParam()` — 清空上次注册的电机

```python
# group_sync_read.py
def clearParam(self):
    self.data_dict.clear()
    # data_dict = {}，清空后面 rxPacket 写入的数据也一并清除
```

`data_dict` 是 GroupSyncRead 的核心数据结构：
```
data_dict = {
    1: [],   # addParam 后初始化为空列表，rxPacket 收到回包后填充
    2: [],
    ...
}
```

`start_address = addr` 和 `data_length = length` 是纯 Python 属性赋值，不调用任何 scs 代码，仅在后续 `txPacket()` 里读取。

### 1.2 `addParam(id_)` — 注册一个电机 ID

```python
# group_sync_read.py
def addParam(self, scs_id):
    if scs_id in self.data_dict:   # 已存在则拒绝（clearParam 后不会触发）
        return False
    self.data_dict[scs_id] = []    # 空列表占位，等 rxPacket 填充实际数据
    self.is_param_changed = True   # 标记"下次发送前要重新 makeParam()"
    return True
```

6 次调用后 `data_dict` 状态：
```
data_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
is_param_changed = True
```

---

## 2. 第二步：`sync_reader.txRxPacket()` — 发广播读包，等 6 个回包

**调用位置**：`motors_bus.py: _sync_read()`

```python
# motors_bus.py
comm = self.sync_reader.txRxPacket()
```

```python
# group_sync_read.py
def txRxPacket(self):
    result = self.txPacket()      # ① 先发广播读指令
    if result != COMM_SUCCESS:
        return result
    return self.rxPacket()        # ② 后逐个接收 6 个电机的应答
```

### 2.1 `txPacket()` — 组装并发送 Sync Read 广播包

```python
# group_sync_read.py
def txPacket(self):
    if not self.data_dict:
        return COMM_NOT_AVAILABLE

    if self.is_param_changed or not self.param:
        self.makeParam()    # 把 data_dict 的 key 展开为 ID 列表

    # param_length = 电机数量（每个 ID 占 1 字节）
    return self.ph.syncReadTx(  #PacketHandler
        self.port,              # PortHandler对象，包装了 serial.Serial
        self.start_address,   # 56
        self.data_length,     # 2
        self.param,           # [1, 2, 3, 4, 5, 6]
        len(self.data_dict) * 1  # 6
    )
```

#### 2.1.1 `makeParam()` — 展开字典为 ID 列表

```python
# group_sync_read.py
def makeParam(self):
    self.param = []
    for scs_id in self.data_dict:
        self.param.append(scs_id)   # 只有 ID，无数据：[1, 2, 3, 4, 5, 6]
```

注意：Sync Read 的 `param` 只包含 ID，**不包含数据**，因为我们还没读到数据。Sync Write 的 `param` 则是 `[id1, b0, b1, id2, b0, b1, ...]`（见写链路文档）。

#### 2.1.2 `syncReadTx()` — 组装 Sync Read 指令包

```python
# protocol_packet_handler.py
def syncReadTx(self, port, start_address, data_length, param, param_length):
    # 包总长 = param_length（ID列表）+ 8（FF FF ID LEN INST ADDR LEN CHK）
    txpacket = [0] * (param_length + 8)         # list[int]

    txpacket[PKT_ID]          = BROADCAST_ID    # 0xFE，广播
    txpacket[PKT_LENGTH]      = param_length + 4 # LEN = 参数长(6) + 4
    txpacket[PKT_INSTRUCTION] = INST_SYNC_READ   # 0x82
    txpacket[PKT_PARAMETER0 + 0] = start_address  # 56，0x38
    txpacket[PKT_PARAMETER0 + 1] = data_length    # 2
    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param  # [1,2,3,4,5,6]

    result = self.txPacket(port, txpacket)   # 计算校验和并写串口
    if result == COMM_SUCCESS:
        # 设置接收超时：每个电机回包 (6+2)=8 字节，6个电机共 48 字节
        port.setPacketTimeout((6 + data_length) * param_length)  # (6+2)*6=48
    return result
```

组装后的原始帧（十六进制）：
```
FF FF FE 0A 82 38 02 01 02 03 04 05 06 CS
│  │  │  │  │  │  │  └─────────────────── ID 列表：1~6
│  │  │  │  │  │  └── data_length = 2（每电机读 2 字节）
│  │  │  │  │  └───── start_address = 0x38 = 56
│  │  │  │  └──────── INST_SYNC_READ = 0x82
│  │  │  └─────────── LEN = 10（param_length=6 + 4）
│  │  └────────────── BROADCAST_ID = 0xFE
│  └───────────────── Header[1] = 0xFF
└──────────────────── Header[0] = 0xFF
CS = (~sum(0xFE+0x0A+0x82+0x38+0x02+1+2+3+4+5+6)) & 0xFF
```

#### 2.1.3 `txPacket()` — 填校验和、写串口

```python
# protocol_packet_handler.py
def txPacket(self, port, txpacket):
    if port.is_using:
        return COMM_PORT_BUSY
    port.is_using = True   # 加并发锁（半双工：发送期间不能接收）

    total_packet_length = txpacket[PKT_LENGTH] + 4  # 10 + 4 = 14 字节

    txpacket[PKT_HEADER0] = 0xFF
    txpacket[PKT_HEADER1] = 0xFF

    checksum = 0
    for idx in range(2, total_packet_length - 1):   # ID 到最后数据字节
        checksum += txpacket[idx]
    txpacket[total_packet_length - 1] = ~checksum & 0xFF   # 校验和

    port.clearPort()   # ser.flush()，避免粘包
    written = port.writePort(txpacket)   # ★ ser.write(packet)
    if total_packet_length != written:
        port.is_using = False
        return COMM_TX_FAIL
    # 注意：不在此释放 is_using，由 rxPacket 在收完后释放
    return COMM_SUCCESS
```

`writePort` 最终调用：
```python
# port_handler.py
def writePort(self, packet):
    return self.ser.write(packet)   # pyserial → write(fd, buf, len) 系统调用
```

至此，14 字节广播读包写入串口，6 个舵机同时收到，各自准备回包。

---

### 2.2 `rxPacket()` — 逐个电机接收应答

```python
# group_sync_read.py
def rxPacket(self):
    self.last_result = False

    for scs_id in self.data_dict:   # 按 ID 顺序逐个接收
        # readRx: 等待 scs_id 这个电机的回包，提取 data_length 字节数据
        self.data_dict[scs_id], result, _ = self.ph.readRx(
            self.port, scs_id, self.data_length   # scs_id=1, data_length=2
        )
        if result != COMM_SUCCESS:
            return result   # 某个电机超时/损坏则中止

    self.last_result = True
    return COMM_SUCCESS
```

#### 2.2.1 `readRx()` — 等待指定 ID 的回包

```python
# protocol_packet_handler.py
def readRx(self, port, scs_id, length):
    result = COMM_TX_FAIL
    error = 0
    data = []

    while True:
        rxpacket, result = self.rxPacket(port)   # 非阻塞循环读取一帧完整包
        # 通信失败（超时/损坏），或者收到了匹配 ID 的包：停止循环
        if result != COMM_SUCCESS or rxpacket[PKT_ID] == scs_id:
            break
        # 收到了其他 ID 的包（如 ID=2 先到），继续等 ID=1

    if result == COMM_SUCCESS and rxpacket[PKT_ID] == scs_id:
        error = rxpacket[PKT_ERROR]
        # 从参数段提取 length 字节（位置数据）
        data.extend(rxpacket[PKT_PARAMETER0: PKT_PARAMETER0 + length])
        # 例：rxpacket = [0xFF,0xFF,0x01,0x04,0x00,0xFD,0x08,CS]
        #     PKT_PARAMETER0=5，length=2
        #     data = [0xFD, 0x08]

    return data, result, error
```

#### 2.2.2 `rxPacket()` — 从串口非阻塞读取并验证一帧完整包

```python
# protocol_packet_handler.py
def rxPacket(self, port):
    rxpacket = []
    wait_length = 6   # 最小包 6 字节（FF FF ID LEN ERR CHK）

    while True:
        # 非阻塞读取（timeout=0，有多少读多少）
        rxpacket.extend(port.readPort(wait_length - len(rxpacket)))
        rx_length = len(rxpacket)

        if rx_length >= wait_length:
            # 搜索包头 0xFF 0xFF
            for idx in range(0, rx_length - 1):
                if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF:
                    break

            if idx == 0:   # 包头在起始位置
                # 字段合法性检查（ID/LEN/ERR 范围）
                if rxpacket[PKT_ID] > 0xFD or rxpacket[PKT_LENGTH] > 250 or rxpacket[PKT_ERROR] > 0x7F:
                    del rxpacket[0]   # 丢弃，重新对齐
                    continue

                # 按 PKT_LENGTH 字段精确计算包长
                # 完整包长 = PKT_LENGTH + 4（FF FF ID LENGTH 各1字节）
                wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1

                if rx_length < wait_length:
                    if port.isPacketTimeout():   # 超时检查
                        result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                        break
                    continue

                # 校验和验证
                checksum = 0
                for i in range(2, wait_length - 1):
                    checksum += rxpacket[i]
                checksum = ~checksum & 0xFF
                result = COMM_SUCCESS if rxpacket[wait_length - 1] == checksum else COMM_RX_CORRUPT
                break

            else:
                del rxpacket[0:idx]   # 丢弃包头前的垃圾字节
        else:
            if port.isPacketTimeout():
                result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                break

    port.is_using = False   # ★ 释放并发锁
    return rxpacket, result
```

`readPort` 最终调用：
```python
# port_handler.py
def readPort(self, length):
    return self.ser.read(length)   # pyserial → read(fd, buf, len) 系统调用
    # timeout=0，非阻塞，有多少读多少，可能返回空 bytes
```

**6 个回包的格式**（半双工，电机按 ID 顺序依次发）：
```
ID=1: [FF FF 01 04 00 FD 08 CS]
       │  │  │  │  │  │  │  └── checksum
       │  │  │  │  │  └──┘───── DATA_L=0xFD, DATA_H=0x08（小端：2301）
       │  │  │  │  └─────────── ERR=0x00（无错误）
       │  │  │  └────────────── LEN=4（ERR + DATA_L + DATA_H + CS）
       │  │  └───────────────── ID=1
       └──┘────────────────────  Header 0xFF 0xFF

ID=2: [FF FF 02 04 00 08 07 CS]  → DATA = [0x08, 0x07] → 0x0708 = 1800
```

6 次 `readRx` 循环结束后，`data_dict` 被填充：
```
data_dict = {
    1: [0xFD, 0x08],   # 2301
    2: [0x08, 0x07],   # 1800
    3: [0x00, 0x08],   # 2048
    4: [0x64, 0x08],   # 2148
    5: [0x00, 0x08],   # 2048
    6: [0xAC, 0x0D],   # 3500
}
```

---

## 3. 第三步：`getData()` — 从缓冲区提取整数

**调用位置**：`motors_bus.py: _sync_read()`

```python
# motors_bus.py
values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
# addr=56, length=2
```

```python
# group_sync_read.py
def getData(self, scs_id, address, data_length):
    if not self.isAvailable(scs_id, address, data_length):
        return 0

    # offset = address - start_address = 56 - 56 = 0（从头开始取）
    offset = address - self.start_address

    if data_length == 1:
        return self.data_dict[scs_id][offset]

    elif data_length == 2:
        # 小端序拼合：低字节 | (高字节 << 8)
        return SCS_MAKEWORD(
            self.data_dict[scs_id][offset],       # DATA_L = 0xFD
            self.data_dict[scs_id][offset + 1]    # DATA_H = 0x08
        )
        # SCS_MAKEWORD(0xFD, 0x08) = 0xFD | (0x08 << 8) = 0x08FD = 2301

    elif data_length == 4:
        return SCS_MAKEDWORD(
            SCS_MAKEWORD(self.data_dict[scs_id][offset + 0],
                         self.data_dict[scs_id][offset + 1]),
            SCS_MAKEWORD(self.data_dict[scs_id][offset + 2],
                         self.data_dict[scs_id][offset + 3])
        )
```

`SCS_MAKEWORD` 定义（小端，`SCS_END=0`）：
```python
# scservo_def.py
def SCS_MAKEWORD(a, b):
    if SCS_END == 0:
        return (a & 0xFF) | ((b & 0xFF) << 8)   # a=低字节，b=高字节
    else:
        return (b & 0xFF) | ((a & 0xFF) << 8)
```

最终返回：
```python
values = {1: 2301, 2: 1800, 3: 2048, 4: 2148, 5: 2048, 6: 3500}
```

---

## 4. 完整调用链一览

```
motors_bus._sync_read(addr=56, length=2, motor_ids=[1..6])
│   # [motors_bus.py] Lerobot 入口：读取 6 个电机地址 56 处的 2 字节（Present_Position）
│
├── _setup_sync_reader(motor_ids, addr=56, length=2)
│   │   # [motors_bus.py] 【注册阶段】告诉 GroupSyncRead 本次要读谁、读哪里、读多少
│   ├── sync_reader.clearParam()
│   │   └── data_dict.clear()              # [group_sync_read.py] 清空上次残留的电机 ID 和数据
│   ├── sync_reader.start_address = 56     # [group_sync_read.py] 记录寄存器起始地址
│   ├── sync_reader.data_length = 2        # [group_sync_read.py] 记录每个电机读取的字节数
│   └── sync_reader.addParam(id_) × 6     # [group_sync_read.py]
│       └── data_dict[id_] = []；is_param_changed = True
│           # 为每个 ID 占位，标记 param 需要重新生成
│
├── sync_reader.txRxPacket()               # [group_sync_read.py]
│   │   # 【通信阶段】先发一帧广播读指令，再逐个收 6 个电机的应答
│   │
│   ├── txPacket()                         # [group_sync_read.py] 准备发送参数
│   │   ├── makeParam()  → self.param = [1,2,3,4,5,6]
│   │   │   # [group_sync_read.py] 把 data_dict 的 key 展开为 ID 列表，供协议层组帧用
│   │   └── ph.syncReadTx(port, 56, 2, [1..6], 6)
│   │       │   # [protocol_packet_handler.py] 组装 Sync Read 广播帧并写串口
│   │       └── txPacket(port, txpacket)   # [protocol_packet_handler.py] list[int]，14 字节
│   │           ├── port.is_using = True   # 半双工加锁，发送期间禁止接收
│   │           ├── 填 Header: txpacket[0]=0xFF, txpacket[1]=0xFF
│   │           ├── 计算 checksum = ~sum(ID...DATA) & 0xFF
│   │           ├── port.clearPort()  → ser.flush()   # [port_handler.py] 冲刷缓冲，避免粘包
│   │           └── port.writePort(txpacket)  → ser.write()  ★ 写串口  # [port_handler.py]
│   │               物理帧: FF FF FE 0A 82 38 02 01 02 03 04 05 06 CS
│   │
│   └── rxPacket()  （for scs_id in data_dict）  # [group_sync_read.py]
│       │   # 按 ID 顺序循环接收 6 个电机应答
│       └── ph.readRx(port, scs_id=1, length=2)  # [protocol_packet_handler.py]
│           │   # 等待 ID=1 的完整回包
│           └── ph.rxPacket(port)  [while True，直到收到 ID=1 的包]  # [protocol_packet_handler.py]
│               │   # 从串口非阻塞读字节，拼包 → 验包头 → 验校验和
│               └── port.readPort(wait_length - len(rxpacket))  → ser.read()  ★ 读串口  # [port_handler.py]
│                   # timeout=0 非阻塞，有多少读多少；超时由 isPacketTimeout() 检测
│                   搜索 0xFF 0xFF → 校验 PKT_LENGTH/ERR → 校验 checksum
│                   data_dict[1] = [0xFD, 0x08]；port.is_using = False（释放锁）
│           （重复 6 次，每次 readRx 收一个电机，收完才进入下一个）
│
└── getData(id_, addr=56, length=2) × 6    # [group_sync_read.py]
    │   # 【解析阶段】从 data_dict 的字节列表还原为整数位置值
    └── SCS_MAKEWORD(data_dict[id_][0], data_dict[id_][1])  # [scservo_def.py]
        # 小端拼合：DATA_L | (DATA_H << 8)
        = {1:2301, 2:1800, 3:2048, 4:2148, 5:2048, 6:3500}
```

---

## 5. 关键细节

| 细节          | 说明                                                                 |
| ------------- | -------------------------------------------------------------------- |
| `is_using` 锁 | `txPacket` 加锁，`rxPacket` 释放；半双工串口不能同时收发             |
| `timeout=0`   | `ser.read()` 非阻塞，while 循环 + `isPacketTimeout()` 组合保护       |
| 包头搜索      | rxPacket 中对 `0xFF 0xFF` 搜索，丢弃前面垃圾字节，处理粘包           |
| 校验和        | `~sum(ID..最后数据字节) & 0xFF`，不含 Header 和校验和本身            |
| 小端序        | DATA_L 先收到，`SCS_MAKEWORD(L, H) = L                               | (H<<8)` |
| readRx 循环   | 可能收到其他 ID 的包（如 ID=2 先到），继续等直到 ID 匹配             |
| 接收超时      | `setPacketTimeout((6 + data_length) * param_length)`，1Mbps 下约 5ms |

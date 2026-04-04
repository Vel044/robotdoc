# Feetech 舵机库 (scservo_sdk) 读写执行链路指南

**文档位置**: `robotdoc/Analysis/Feetech_舵机库执行指南.md`

---

## 核心调用链路（细化到内核系统调用）

### 读取链路：`get_observation` → `Present_Position`

```
record_loop()                                         (lerobot/src/lerobot/record.py)
  │
  └─► robot.get_observation()
        └─► self.bus.sync_read("Present_Position")    (lerobot/src/lerobot/motors/motors_bus.py)
              └─► self.sync_reader.txRxPacket()
  │
  ┌──────────────────────────────────────────────────────────────────────────
  │  GroupSyncRead.txRxPacket()         (scservo_sdk/group_sync_read.py:114)
  ├──────────────────────────────────────────────────────────────────────────
  │
  ├─► txPacket()                        (group_sync_read.py:74)
  │       │  检查 data_dict 非空，调用 makeParam() 整理 ID 列表
  │       └─► self.ph.syncReadTx(port, start_address, data_length,
  │                               param, param_length)
  │                 │              (protocol_packet_handler.py:431)
  │                 │  构造 INST_SYNC_READ 数据包：
  │                 │    [0xFF][0xFF][0xFE(BROADCAST)][LEN]
  │                 │    [INST_SYNC_READ][start_addr][data_len]
  │                 │    [id1][id2]...[idN][checksum]
  │                 │  设置超时：setPacketTimeout((6+data_length)*N)
  │                 └─► self.txPacket(port, txpacket)
  │                           │     (protocol_packet_handler.py:69)
  │                           │  填写 Header (0xFF 0xFF)
  │                           │  计算 Checksum = ~sum(ID..PARAM) & 0xFF
  │                           │  port.clearPort()  → ser.flush()
  │                           └─► port.writePort(txpacket)
  │                                     │   (port_handler.py:63)
  │                                     └─► self.ser.write(packet)
  │                                               │  (pyserial)
  │                                               └─► [系统调用] write(fd, buf, len)
  │                                                   → UART/USB tty 驱动写串口
  │
  └─► rxPacket()                        (group_sync_read.py:91)
          │  遍历所有舵机 ID，对每个 ID 调用一次 readRx
          └─► for scs_id in self.data_dict:
                  self.ph.readRx(port, scs_id, data_length)
                          │       (protocol_packet_handler.py:262)
                          │  循环调用 rxPacket 直到收到目标 ID 的回包
                          └─► self.rxPacket(port)
                                    │   (protocol_packet_handler.py:103)
                                    │
                                    │  循环读取，直到收到完整包或超时：
                                    └─► port.readPort(wait_length - rx_length)
                                              │   (port_handler.py:57)
                                              └─► self.ser.read(length)
                                                        │  (pyserial, timeout=0 非阻塞)
                                                        └─► [系统调用] read(fd, buf, len)
                                                            → UART/USB tty 驱动读缓冲区
                                    │
                                    │  搜索包头 [0xFF][0xFF]
                                    │  读取 PKT_ID / PKT_LENGTH / PKT_ERROR
                                    │  验证 Checksum
                                    └─► 返回 (rxpacket[], COMM_SUCCESS/TIMEOUT/CORRUPT)
```

---

### 写入链路：`send_action` → `Goal_Position`

```
record_loop()                                         (lerobot/src/lerobot/record.py)
  │
  └─► robot.send_action()
        └─► self.bus.sync_write("Goal_Position")      (lerobot/src/lerobot/motors/motors_bus.py)
              └─► self.sync_writer.txPacket()
  │
  ┌──────────────────────────────────────────────────────────────────────────
  │  GroupSyncWrite.txPacket()          (scservo_sdk/group_sync_write.py:66)
  ├──────────────────────────────────────────────────────────────────────────
  │
  │  makeParam()：将 data_dict {id: [lo, hi]} 展开为
  │    param = [id1, lo1, hi1, id2, lo2, hi2, ...]
  │
  └─► self.ph.syncWriteTxOnly(port, start_address, data_length,
                               param, param_length)
              │                (protocol_packet_handler.py:450)
              │  构造 INST_SYNC_WRITE 数据包：
              │    [0xFF][0xFF][0xFE(BROADCAST)][LEN]
              │    [INST_SYNC_WRITE][start_addr][data_len]
              │    [id1][lo1][hi1][id2][lo2][hi2]...[checksum]
              └─► self.txRxPacket(port, txpacket)
                          │     (protocol_packet_handler.py:177)
                          │  BROADCAST_ID → 无需等待响应包
                          │  → port.is_using = False 后直接返回
                          └─► self.txPacket(port, txpacket)
                                    │   (protocol_packet_handler.py:69)
                                    │  填写 Header (0xFF 0xFF)
                                    │  计算 Checksum
                                    │  port.clearPort() → ser.flush()
                                    └─► port.writePort(txpacket)
                                              │   (port_handler.py:63)
                                              └─► self.ser.write(packet)
                                                        │  (pyserial)
                                                        └─► [系统调用] write(fd, buf, len)
                                                            → UART/USB tty 驱动写串口
```

---

## 📦 各层源码详解

### 1. `GroupSyncRead` — 同步读取业务层

**源码位置**：`scservo_sdk/group_sync_read.py`

```python
# group_sync_read.py:5
class GroupSyncRead:
    def __init__(self, port, ph, start_address, data_length):
        self.port = port
        self.ph = ph
        self.start_address = start_address  # 寄存器起始地址（如 0x38 = Present_Position）
        self.data_length = data_length      # 每个舵机读取的字节数（如 2 字节）
        self.last_result = False
        self.is_param_changed = False
        self.param = []        # 展开后的 ID 列表，用于构建数据包
        self.data_dict = {}    # {scs_id: [byte0, byte1, ...]}，存储读回的原始数据
        self.clearParam()

    # group_sync_read.py:29
    def makeParam(self):
        """将 data_dict 的所有 key（舵机ID）展平到 self.param 列表"""
        if not self.data_dict:
            return
        self.param = []
        for scs_id in self.data_dict:
            self.param.append(scs_id)  # 每个ID占1字节

    # group_sync_read.py:74
    def txPacket(self):
        """发送 SYNC_READ 指令（广播到所有ID，让各舵机准备回包）"""
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed is True or not self.param:
            self.makeParam()
        # param_length = N（舵机数量），每个舵机仅贡献1字节ID
        return self.ph.syncReadTx(self.port, self.start_address, self.data_length,
                                  self.param, len(self.data_dict.keys()) * 1)

    # group_sync_read.py:91
    def rxPacket(self):
        """按顺序逐个接收每个舵机的响应包"""
        self.last_result = False
        result = COMM_RX_FAIL
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        for scs_id in self.data_dict:
            # readRx：循环调 rxPacket 直到收到 scs_id 对应的包
            self.data_dict[scs_id], result, _ = self.ph.readRx(
                self.port, scs_id, self.data_length)
            if result != COMM_SUCCESS:
                return result           # 任一舵机失败则中止
        if result == COMM_SUCCESS:
            self.last_result = True
        return result

    # group_sync_read.py:114
    def txRxPacket(self):
        """先发后收：txPacket() → rxPacket()"""
        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result
        return self.rxPacket()

    # group_sync_read.py:146
    def getData(self, scs_id, address, data_length):
        """从 data_dict 中提取数据，按 1/2/4 字节拼接为整数"""
        if not self.isAvailable(scs_id, address, data_length):
            return 0
        offset = address - self.start_address
        if data_length == 1:
            return self.data_dict[scs_id][offset]
        elif data_length == 2:
            return SCS_MAKEWORD(self.data_dict[scs_id][offset],
                                self.data_dict[scs_id][offset + 1])
        elif data_length == 4:
            return SCS_MAKEDWORD(
                SCS_MAKEWORD(self.data_dict[scs_id][offset + 0],
                             self.data_dict[scs_id][offset + 1]),
                SCS_MAKEWORD(self.data_dict[scs_id][offset + 2],
                             self.data_dict[scs_id][offset + 3]))
        return 0
```

---

### 2. `GroupSyncWrite` — 同步写入业务层

**源码位置**：`scservo_sdk/group_sync_write.py`

```python
# group_sync_write.py:5
class GroupSyncWrite:
    def __init__(self, port, ph, start_address, data_length):
        self.port = port
        self.ph = ph
        self.start_address = start_address  # 寄存器起始地址（如 0x2E = Goal_Position）
        self.data_length = data_length      # 每个舵机写入的字节数（如 2 字节）
        self.is_param_changed = False
        self.param = []         # 展开后：[id1, lo1, hi1, id2, lo2, hi2, ...]
        self.data_dict = {}     # {scs_id: [lo_byte, hi_byte]}
        self.clearParam()

    # group_sync_write.py:18
    def makeParam(self):
        """将 data_dict 展平：每个舵机贡献 [ID, data_bytes...]"""
        if not self.data_dict:
            return
        self.param = []
        for scs_id in self.data_dict:
            if not self.data_dict[scs_id]:
                return
            self.param.append(scs_id)              # 1 字节 ID
            self.param.extend(self.data_dict[scs_id])  # data_length 字节数据

    # group_sync_write.py:66
    def txPacket(self):
        """构建并发送 SYNC_WRITE 指令（广播，无需等待回包）"""
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE
        if self.is_param_changed is True or not self.param:
            self.makeParam()
        # param_length = N * (1 + data_length)，每舵机 ID + 数据
        return self.ph.syncWriteTxOnly(
            self.port, self.start_address, self.data_length,
            self.param, len(self.data_dict.keys()) * (1 + self.data_length))
```

---

### 3. `protocol_packet_handler` — 协议组包/解包层

**源码位置**：`scservo_sdk/protocol_packet_handler.py`

#### `txPacket` — 发送数据包（行69）

```python
# protocol_packet_handler.py:69
def txPacket(self, port, txpacket):
    checksum = 0
    total_packet_length = txpacket[PKT_LENGTH] + 4  # +4: HEADER0 HEADER1 ID LENGTH

    if port.is_using:            # 防止并发占用串口
        return COMM_PORT_BUSY
    port.is_using = True

    if total_packet_length > TXPACKET_MAX_LEN:   # 最大 250 字节
        port.is_using = False
        return COMM_TX_ERROR

    # 填充包头
    txpacket[PKT_HEADER0] = 0xFF
    txpacket[PKT_HEADER1] = 0xFF

    # 计算校验和：对 ID + LENGTH + INSTRUCTION + PARAM 求和取反
    for idx in range(2, total_packet_length - 1):
        checksum += txpacket[idx]
    txpacket[total_packet_length - 1] = ~checksum & 0xFF

    port.clearPort()                     # ser.flush()：清空发送缓冲
    written_packet_length = port.writePort(txpacket)   # → ser.write()

    if total_packet_length != written_packet_length:
        port.is_using = False
        return COMM_TX_FAIL

    return COMM_SUCCESS
    # 注意：txPacket 不释放 port.is_using，由调用方或 rxPacket 结尾释放
```

**数据包格式（Feetech 协议）**：
```
[0xFF][0xFF][ID][LENGTH][INSTRUCTION][PARAM0]...[PARAMN][CHECKSUM]
  ^     ^    ^    ^          ^                              ^
Header Header 舵机ID 后续字节数  指令码(SYNC_READ=0x82)    ~sum(ID..PARAM)&0xFF
```

---

#### `rxPacket` — 接收数据包（行103）

```python
# protocol_packet_handler.py:103
def rxPacket(self, port):
    rxpacket = []
    result = COMM_TX_FAIL
    checksum = 0
    rx_length = 0
    wait_length = 6   # 最小包长：HEADER0 HEADER1 ID LENGTH ERROR CHKSUM

    while True:
        # 非阻塞读（timeout=0），每次尽量读满 wait_length - rx_length 字节
        rxpacket.extend(port.readPort(wait_length - rx_length))
        rx_length = len(rxpacket)

        if rx_length >= wait_length:
            # 搜索包头 [0xFF, 0xFF]
            for idx in range(0, rx_length - 1):
                if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF:
                    break

            if idx == 0:   # 包头在最前面
                # 校验 ID/LENGTH/ERROR 合法性
                if (rxpacket[PKT_ID] > 0xFD or
                        rxpacket[PKT_LENGTH] > RXPACKET_MAX_LEN or
                        rxpacket[PKT_ERROR] > 0x7F):
                    del rxpacket[0]      # 丢弃第一个字节，重新搜索
                    rx_length -= 1
                    continue

                # 根据 LENGTH 字段确定完整包长
                if wait_length != rxpacket[PKT_LENGTH] + PKT_LENGTH + 1:
                    wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                    continue

                if rx_length < wait_length:
                    if port.isPacketTimeout():   # 超时判断
                        result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                        break
                    else:
                        continue         # 还没收完，继续读

                # 校验 Checksum
                for i in range(2, wait_length - 1):
                    checksum += rxpacket[i]
                checksum = ~checksum & 0xFF
                result = COMM_SUCCESS if rxpacket[wait_length - 1] == checksum \
                         else COMM_RX_CORRUPT
                break
            else:
                # 包头不在最前，丢弃前缀脏数据
                del rxpacket[0:idx]
                rx_length -= idx
        else:
            if port.isPacketTimeout():
                result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                break
            # 数据不足，继续 readPort

    port.is_using = False    # 释放串口占用标志
    return rxpacket, result
```

---

#### `syncReadTx` — 构造并发送 SYNC_READ 指令（行431）

```python
# protocol_packet_handler.py:431
def syncReadTx(self, port, start_address, data_length, param, param_length):
    txpacket = [0] * (param_length + 8)
    # 包结构: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN [id1..idN] CHKSUM

    txpacket[PKT_ID]          = BROADCAST_ID       # 0xFE，广播给所有舵机
    txpacket[PKT_LENGTH]      = param_length + 4   # INST + START_ADDR + DATA_LEN + CHKSUM
    txpacket[PKT_INSTRUCTION] = INST_SYNC_READ     # 0x82

    txpacket[PKT_PARAMETER0 + 0] = start_address   # 要读的寄存器起始地址
    txpacket[PKT_PARAMETER0 + 1] = data_length      # 每个舵机读取的字节数
    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0:param_length]
    # param = [id1, id2, ..., idN]

    result = self.txPacket(port, txpacket)   # 发送
    if result == COMM_SUCCESS:
        # 设置接收超时 = (6 + data_length) × N 字节时间
        port.setPacketTimeout((6 + data_length) * param_length)
    return result
```

---

#### `syncWriteTxOnly` — 构造并发送 SYNC_WRITE 指令（行450）

```python
# protocol_packet_handler.py:450
def syncWriteTxOnly(self, port, start_address, data_length, param, param_length):
    txpacket = [0] * (param_length + 8)
    # 包结构: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN [id1 data1...] CHKSUM

    txpacket[PKT_ID]          = BROADCAST_ID       # 0xFE，广播
    txpacket[PKT_LENGTH]      = param_length + 4
    txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE    # 0x83

    txpacket[PKT_PARAMETER0 + 0] = start_address
    txpacket[PKT_PARAMETER0 + 1] = data_length
    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0:param_length]
    # param = [id1, lo1, hi1, id2, lo2, hi2, ...]

    _, result, _ = self.txRxPacket(port, txpacket)
    # txRxPacket 检测到 BROADCAST_ID → 发完直接返回，不等待回包
    return result
```

---

#### `txRxPacket` — 发送并等待回包（行177）

```python
# protocol_packet_handler.py:177
def txRxPacket(self, port, txpacket):
    rxpacket = None
    error = 0

    result = self.txPacket(port, txpacket)   # 先发
    if result != COMM_SUCCESS:
        return rxpacket, result, error

    # BROADCAST_ID（0xFE）：广播包不需要等回包，直接释放并返回
    if txpacket[PKT_ID] == BROADCAST_ID:
        port.is_using = False
        return rxpacket, result, error

    # 设置接收超时
    if txpacket[PKT_INSTRUCTION] == INST_READ:
        port.setPacketTimeout(txpacket[PKT_PARAMETER0 + 1] + 6)
    else:
        port.setPacketTimeout(6)

    # 循环接收，直到收到 ID 匹配的回包
    while True:
        rxpacket, result = self.rxPacket(port)
        if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
            break

    if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
        error = rxpacket[PKT_ERROR]

    return rxpacket, result, error
```

---

### 4. `PortHandler` — 串口物理层

**源码位置**：`scservo_sdk/port_handler.py`

```python
# port_handler.py:12
class PortHandler(object):
    def __init__(self, port_name):
        self.is_open = False
        self.baudrate = DEFAULT_BAUDRATE    # 1,000,000 bps
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0
        self.is_using = False               # 互斥标志，防止并发读写
        self.port_name = port_name          # 如 "/dev/ttyUSB0"
        self.ser = None                     # pyserial.Serial 实例

    # port_handler.py:91
    def setupPort(self, cflag_baud):
        """初始化 pyserial.Serial，timeout=0（非阻塞读）"""
        if self.is_open:
            self.closePort()
        self.ser = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            timeout=0                       # 非阻塞：read() 立即返回已有数据
        )
        self.is_open = True
        self.ser.reset_input_buffer()
        # 每字节传输时间 (ms) = 10 位 / 波特率 × 1000
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    # port_handler.py:57
    def readPort(self, length):
        """非阻塞读取最多 length 字节"""
        if sys.version_info > (3, 0):
            return self.ser.read(length)    # → [系统调用] read(fd, buf, length)
        else:
            return [ord(ch) for ch in self.ser.read(length)]

    # port_handler.py:63
    def writePort(self, packet):
        """写入字节串到串口"""
        return self.ser.write(packet)       # → [系统调用] write(fd, buf, len)

    # port_handler.py:66
    def setPacketTimeout(self, packet_length):
        """以字节数估算超时时间 (ms)"""
        self.packet_start_time = self.getCurrentTime()
        # 超时 = 传输时间 + 2 × LATENCY_TIMER(16ms) + 2ms
        self.packet_timeout = (self.tx_time_per_byte * packet_length) + \
                              (LATENCY_TIMER * 2.0) + 2.0

    # port_handler.py:74
    def isPacketTimeout(self):
        """检查是否超时（getTimeSinceStart > packet_timeout）"""
        if self.getTimeSinceStart() > self.packet_timeout:
            self.packet_timeout = 0
            return True
        return False
```

---

### 5. `pyserial SerialBase (POSIX)` — Python 串口驱动层

**源码位置**：`serial/serialposix.py`

> pyserial 不在 CPython 标准库中，是独立的第三方包，源码已拷贝到工作区 `serial/` 目录。
> 在 Linux/树莓派5 上实际使用的是 `serialposix.py`（POSIX 实现），内部通过 `os.write` / `os.read` 调用系统调用。

#### `write()` — 串口写入（serialposix.py:612）

```python
# serial/serialposix.py:612
def write(self, data):
    """Output the given byte string over the serial port."""
    if not self.is_open:
        raise PortNotOpenError()

    d = to_bytes(data)              # 确保是 bytes 类型
    tx_len = length = len(d)
    timeout = Timeout(self._write_timeout)

    while tx_len > 0:
        try:
            n = os.write(self.fd, d)    # ← 关键：调用 os.write，即 POSIX write(2)
                                        # self.fd 是 open("/dev/ttyUSB0") 返回的文件描述符

            if timeout.is_non_blocking:
                # write_timeout=0（非阻塞）：写了多少返回多少，不重试
                return n

            elif not timeout.is_infinite:
                # 有超时设置：用 select 等待 fd 可写后继续发剩余数据
                if timeout.expired():
                    raise SerialTimeoutException('Write timeout')
                abort, ready, _ = select.select(
                    [self.pipe_abort_write_r], [self.fd], [], timeout.time_left())
                if abort:
                    os.read(self.pipe_abort_write_r, 1000)
                    break
                if not ready:
                    raise SerialTimeoutException('Write timeout')
            else:
                # 无限等待：select 不设超时，等到 fd 可写
                abort, ready, _ = select.select(
                    [self.pipe_abort_write_r], [self.fd], [], None)
                if abort:
                    os.read(self.pipe_abort_write_r, 1)
                    break
                if not ready:
                    raise SerialException('write failed (select)')

            d = d[n:]       # 已发送的部分切掉
            tx_len -= n

        except OSError as e:
            # 忽略 EAGAIN/EINTR 等非致命错误，其余重新抛出
            if e.errno not in (errno.EAGAIN, errno.EALREADY,
                               errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                raise SerialException('write failed: {}'.format(e))

    return length - tx_len  # 返回实际发送字节数
```

> **lerobot 场景**：`PortHandler.setupPort()` 中 `write_timeout` 默认为 `None`（无限等待），
> 所以走的是 `timeout.is_infinite` 分支，每次 `os.write` 后用 `select` 等待 fd 可写再发剩余数据。

---

#### `read()` — 串口读取（serialposix.py:553）

```python
# serial/serialposix.py:553
def read(self, size=1):
    """\
    Read size bytes from the serial port. If a timeout is set it may
    return less characters as requested. With no timeout it will block
    until the requested number of bytes is read.
    """
    if not self.is_open:
        raise PortNotOpenError()

    read = bytearray()
    timeout = Timeout(self._timeout)    # _timeout 由 serial.Serial(timeout=0) 设置

    while len(read) < size:
        try:
            # select 等待 fd 可读（timeout=0 → time_left()=0 → 立即返回）
            ready, _, _ = select.select(
                [self.fd, self.pipe_abort_read_r], [], [], timeout.time_left())

            if self.pipe_abort_read_r in ready:
                # cancel_read() 被调用，中止读取
                os.read(self.pipe_abort_read_r, 1000)
                break

            if not ready:
                # timeout=0 且无数据：select 返回空列表，直接 break
                # 这就是 PortHandler 非阻塞读取的底层机制
                break

            buf = os.read(self.fd, size - len(read))  # ← POSIX read(2)

        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EALREADY,
                               errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                raise SerialException('read failed: {}'.format(e))
        else:
            if not buf:
                # fd 可读但返回空字节：设备断开
                raise SerialException(
                    'device reports readiness to read but returned no data '
                    '(device disconnected or multiple access on port?)')
            read.extend(buf)

        if timeout.expired():
            break

    return bytes(read)
```

> **关键细节**：`PortHandler.setupPort()` 传入 `timeout=0`，对应 pyserial 的 `_timeout=0`。
> `Timeout(0).time_left()` 返回 `0`，`select(..., 0)` 立即返回。
> 这意味着 `ser.read(n)` 是**非阻塞的**——只返回当前串口缓冲区中已有的数据（可能少于 n 字节）。
> 因此 `rxPacket()` 需要在 while 循环中反复调用 `readPort()`，并配合 `isPacketTimeout()` 做软超时。

---

### 6. `glibc 系统调用包装层` — 用户态到内核的桥梁

**源码位置**：`glibc-2.42/sysdeps/unix/sysv/linux/`

> pyserial 的 `os.write()` / `os.read()` 经过 CPython，最终调用 glibc 的这些包装函数，
> 再由 glibc 以 `svc 0` 指令陷入 Linux 内核。

#### ARM64 系统调用号（`glibc-2.42/sysdeps/unix/sysv/linux/aarch64/arch-syscall.h`）

```c
#define __NR_read      63   // x8=63
#define __NR_write     64   // x8=64
#define __NR_ioctl     29   // x8=29
#define __NR_pselect6  72   // x8=72  (select/pselect 在 ARM64 统一走 pselect6)
```

---

#### `write(2)` — glibc 包装（`glibc-2.42/sysdeps/unix/sysv/linux/write.c:24`）

```c
// glibc-2.42/sysdeps/unix/sysv/linux/write.c:24
ssize_t
__libc_write (int fd, const void *buf, size_t nbytes)
{
  // SYSCALL_CANCEL 宏：带线程取消检查的系统调用入口
  // 展开后调用 __syscall_cancel(fd, buf, nbytes, 0,0,0, __NR_write)
  return SYSCALL_CANCEL (write, fd, buf, nbytes);
}
// 符号别名：__libc_write → __write → write
weak_alias (__libc_write, write)
```

`SYSCALL_CANCEL` 宏展开路径（`glibc-2.42/sysdeps/unix/sysdep.h:251`）：

```c
// sysdep.h:251
#define SYSCALL_CANCEL(...)  __SYSCALL_CANCEL_CALL(__VA_ARGS__)
// ↓ 按参数个数展开为：
#define __SYSCALL_CANCEL3(name, a1, a2, a3)
  __syscall_cancel(__SSC(a1), __SSC(a2), __SSC(a3), 0, 0, 0, __NR_write)
```

最终落到 ARM64 汇编（`glibc-2.42/sysdeps/unix/sysv/linux/aarch64/syscall_cancel.S:31`）：

```asm
// syscall_cancel.S:31  __syscall_cancel_arch
__syscall_cancel_arch_start:
    // 检查线程取消标志，如果设置了 CANCELED_BITMASK → 跳转取消处理
    ldr   w0, [x0]
    tbnz  w0, TCB_CANCELED_BIT, 1f

    // 将参数从 x1-x7 移到系统调用寄存器 x0-x5，x8=syscall number
    mov   x8, x1    // x1 = __NR_write = 64
    mov   x0, x2    // x2 = fd
    mov   x1, x3    // x3 = buf
    mov   x2, x4    // x4 = nbytes
    ...
    svc   0x0        // ★ ARM64 系统调用指令，陷入内核
__syscall_cancel_arch_end:
    ret
```

---

#### `read(2)` — glibc 包装（`glibc-2.42/sysdeps/unix/sysv/linux/read.c:24`）

```c
// glibc-2.42/sysdeps/unix/sysv/linux/read.c:24
ssize_t
__libc_read (int fd, void *buf, size_t nbytes)
{
  // 同 write，带取消检查，展开后 __NR_read = 63
  return SYSCALL_CANCEL (read, fd, buf, nbytes);
}
weak_alias (__libc_read, read)
```

ARM64 上实际执行路径与 write 相同，最终：

```asm
// 关键寄存器赋值（INTERNAL_SYSCALL_RAW 内联路径）
// glibc-2.42/sysdeps/unix/sysv/linux/aarch64/sysdep.h:167
register long _x8 asm ("x8") = __NR_read;  // x8 = 63
// x0=fd, x1=buf, x2=nbytes
asm volatile ("svc 0  // syscall read"
              : "=r"(_x0) : "r"(_x8) ... : "memory");
```

---

#### `select(2)` / `pselect(2)` — glibc 包装（`glibc-2.42/sysdeps/unix/sysv/linux/select.c` & `pselect.c`）

> ARM64 上 **没有独立的 `select` 系统调用**，glibc 统一将 `select()` 和 `pselect()` 都路由到
> `pselect6_time64`（`__NR_pselect6 = 72`）。

```c
// glibc-2.42/sysdeps/unix/sysv/linux/select.c:32
int
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
            struct __timeval64 *timeout)
{
  // 将 timeval 转换为 timespec64（纳秒精度）
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL) {
    ts64.tv_sec  = s;
    ts64.tv_nsec = ns;     // us → ns
    pts64 = &ts64;
  }

  // ARM64 定义了 __ASSUME_TIME64_SYSCALLS，直接走 pselect6_time64
  int r = SYSCALL_CANCEL (pselect6_time64,  // __NR_pselect6 = 72
                           nfds, readfds, writefds, exceptfds,
                           pts64, NULL);     // NULL = 无信号掩码
  return r;
}
weak_alias (__select, select)
```

```c
// glibc-2.42/sysdeps/unix/sysv/linux/pselect.c:22
// pselect 的内部函数，带 sigmask 参数
static int
pselect64_syscall (int nfds, fd_set *readfds, fd_set *writefds,
                   fd_set *exceptfds, const struct __timespec64 *timeout,
                   const sigset_t *sigmask)
{
  __syscall_ulong_t data[2] = {
    (uintptr_t) sigmask, __NSIG_BYTES   // 打包 sigmask 指针+长度
  };
  // 最终也是 pselect6_time64，x8=72
  return SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds, exceptfds,
                         timeout, data);
}
```

**pyserial 实际传入的 timeout 值**（`serialposix.py` 中）：
- `ser.read(n)`：`timeout=0`，`time_left()=0` → `select(..., 0)` → `pselect6` 立即返回
- `ser.write(d)`（`write_timeout=None`）：`select(..., None)` → `pselect6` 无限等待直到 fd 可写

---

#### `ioctl(2)` — glibc ARM64 专属汇编（`glibc-2.42/sysdeps/unix/sysv/linux/aarch64/ioctl.S`）

> ioctl 在 pyserial 中用于串口参数配置（波特率、数据位、校验位等），
> 在 `setupPort()` / `setBaudRate()` 时调用，**不在数据收发的热路径上**。

```asm
// glibc-2.42/sysdeps/unix/sysv/linux/aarch64/ioctl.S:22
// ioctl 有 ARM64 平台专属实现（非通用 SYSCALL_CANCEL 路径）
ENTRY(__ioctl)
    mov   x8, #__NR_ioctl   // x8 = 29
    sxtw  x0, w0             // 将 fd（int）符号扩展为 64 位
    svc   #0x0               // ★ 直接陷入内核，无取消检查
    cmn   x0, #4095          // 检查返回值是否为错误码（-4095~-1）
    b.cs  .Lsyscall_error    // 如果是错误码，跳转处理 errno
    ret
PSEUDO_END (__ioctl)
weak_alias (__ioctl, ioctl)
```

> **注意**：`ioctl` 用的是 **直接 `svc`** 而不是 `SYSCALL_CANCEL`，
> 这是因为 ioctl 本身不是可安全取消的阻塞点（它通常很快返回）。

---

**内核系统调用层**：

```
─── 写路径 ──────────────────────────────────────────────────────────────
ser.write(packet)                  (serialposix.py:612)
    │
    ├─ select.select([pipe], [fd], [], None)   ★ 可能阻塞，但不占CPU
    │   (等待 tty fd 可写，通常 tty 写缓冲未满时立即返回，几乎不等待)
    │   └─► glibc select() → pselect64_syscall()  (select.c:32)
    │            └─► SYSCALL_CANCEL(pselect6_time64, ...)
    │                    └─► __syscall_cancel_arch: svc 0  x8=72(__NR_pselect6)
    │
    └─► os.write(self.fd, d)       (CPython posixmodule.c)
            └─► glibc write()      (write.c:24)
                    └─► SYSCALL_CANCEL(write, fd, buf, nbytes)
                            └─► __syscall_cancel_arch: svc 0  x8=__NR_write(64)
                    └─► [Linux kernel] ksys_write() → vfs_write()
                            └─► tty_write()            ← tty 层
                                    └─► usb_serial_generic_write()
                                            │   (generic.c:212)
                                            │
                                            ├─ kfifo_in_locked(buf → write_fifo)
                                            │   ★ 数据写入内核 kfifo 环形缓冲区
                                            │   ★ 耗时极短（内存拷贝，<1μs）
                                            │
                                            └─► usb_serial_generic_write_start()
                                                    │   (generic.c:153)
                                                    │
                                                    └─► usb_submit_urb(urb, GFP_ATOMIC)
                                                            │   ★ 提交 USB bulk OUT URB
                                                            │   ★ 立即返回（异步！不等传输完成）
                                                            │   ★ 不阻塞，不占CPU
                                                            │
                                                            └─► [USB 主控制器 DMA]
                                                                    ★ 硬件异步传输，CPU完全不参与
                                                                    ★ 1Mbps下12字节≈0.12ms传输时间
                                                                    └─► CH340 → UART TX 信号

    写路径完成后 os.write() 立即返回（数据在 kfifo，URB 已提交）
    ★ 写路径总耗时：<0.1ms（主要是 kfifo memcpy + URB 提交）
    ★ 无阻塞等待（USB 传输由硬件异步完成）

    [异步回调] usb_serial_generic_write_bulk_callback()  (generic.c:433)
        ← USB 传输完成后由中断触发，通知内核 URB 已完成
        → 继续发送 kfifo 中剩余数据（若有）

─── 读路径 ──────────────────────────────────────────────────────────────

    [硬件异步，与读路径并行]
    USB 中断 IN（周期性轮询，1ms/次）
        └─► usb_serial_generic_read_bulk_callback()  (generic.c:369)
                │   ★ 中断上下文，CPU 短暂被占用（<10μs）
                └─► process_read_urb(urb)
                        └─► tty_insert_flip_string()   ← 数据写入 TTY flip buffer
                                └─► tty_flip_buffer_push()  ← 唤醒等待读的进程
                                        └─► usb_submit_urb()  ← 重新提交 IN URB，等下一包

ser.read(length)                   (serialposix.py:553)
    │
    ├─► select.select([fd], [], [], 0)
    │       └─► glibc __select64() → pselect64_syscall()  (select.c:32)
    │               └─► SYSCALL_CANCEL(pselect6_time64, nfds, ..., 0)
    │                       └─► __syscall_cancel_arch: svc 0  x8=72(__NR_pselect6)
    │                               └─► [Linux kernel] 检查 TTY flip buffer 是否有数据
    │                               ★ timeout=0：立即返回，不阻塞，不占CPU等待
    │                               ★ 若无数据：ready=[] → rxPacket while循环继续轮询
    │
    └─► os.read(self.fd, n)        ← 仅在 select 报告可读时调用
            └─► glibc read()        (read.c:24)
                    └─► SYSCALL_CANCEL(read, fd, buf, nbytes)
                            └─► __syscall_cancel_arch: svc 0  x8=__NR_read(63)
                    └─► [Linux kernel] ksys_read() → vfs_read()
                            └─► tty_read() → n_tty_read()
                                    └─► 从 TTY line discipline 缓冲区复制数据到用户空间
                                    ★ 耗时：内存拷贝，<1μs

    读路径耗时分析：
    ★ 单次 ser.read() 本身极快（<5μs）
    ★ 真正的等待在 rxPacket() 的 while 循环轮询：
        - 每次循环：select(0) + read() ≈ 5-10μs
        - 舵机响应延迟：固件处理 ~100-500μs
        - USB IN 轮询间隔：1ms/次（USB Full Speed）
        ★ 总等待时间主要由 USB 1ms 轮询间隔决定
        ★ 等待期间 CPU 在 Python while 循环中空转（忙等！占用CPU）
        ★ 这是读路径最大的 CPU 开销来源

─── 耗时与CPU占用总结 ───────────────────────────────────────────────────

    操作                          耗时        CPU占用    阻塞类型
    ──────────────────────────────────────────────────────────────
    kfifo_in_locked (写缓冲)      <1μs        占用       无阻塞
    usb_submit_urb (提交URB)      <10μs       占用       无阻塞
    USB bulk OUT 传输             ~0.12ms     不占用     硬件异步
    write select (等tty可写)      ~0μs        不占用     阻塞但通常立即返回
    ──────────────────────────────────────────────────────────────
    USB IN 中断回调               <10μs       占用       中断
    USB IN 轮询间隔               1ms/次      不占用     硬件定时
    rxPacket while 轮询           N×5-10μs    ★占用CPU   忙等（无sleep）
    舵机固件响应延迟              ~100-500μs  不占用     等待外设
    ──────────────────────────────────────────────────────────────
    ★ 最大瓶颈：rxPacket 忙等轮询 × 6个舵机，每帧约 6-10ms CPU 空转
```

> **树莓派5 ARM 上的实际路径**：
> - USB 摄像头走 UVC，舵机走 CH340/CP2102 USB-Serial 适配器（`/dev/ttyUSB0`）
> - `ser.write()` → `write(2)` 系统调用 → `usb_serial_write()` → USB bulk OUT 传输
> - `ser.read()` → `read(2)` 系统调用，`timeout=0` 非阻塞，配合 `rxPacket` 的 `isPacketTimeout()` 软件超时轮询
> - 1Mbps 波特率下，每字节传输时间 = 10 / 1,000,000 × 1000 = **0.01 ms**

---

## 📊 数据包格式对照

### SYNC_READ 发送包（广播，无回包）

```
[0xFF][0xFF][0xFE][LEN][0x82][addr][dlen][id1][id2]...[idN][chksum]
             BCAST       INST
  LEN = N + 4（INST + addr + dlen + chksum）
```

### 各舵机回包（每个舵机单独回）

```
[0xFF][0xFF][ID][LEN][ERR][data0][data1]...[chksum]
              ^    ^    ^
            舵机ID  2+dlen  错误标志位（0=正常）
```

### SYNC_WRITE 发送包（广播，各舵机直接执行，无回包）

```
[0xFF][0xFF][0xFE][LEN][0x83][addr][dlen][id1][d10][d11][id2][d20][d21]...[chksum]
             BCAST       INST
  LEN = N*(1+dlen) + 4
```

---

## 📝 常见读写寄存器地址

**源码位置**：`lerobot/src/lerobot/motors/feetech/tables.py`

| 参数 | 地址 | 长度 | 说明 |
|------|------|------|------|
| Goal_Position | 0x2E (46) | 2 | 目标位置（写入） |
| Present_Position | 0x38 (56) | 2 | 当前位置（读取） |
| Present_Velocity | 0x3A (58) | 2 | 当前速度（读取） |
| Present_Load | 0x3C (60) | 2 | 当前负载（读取） |

---

## ⏱️ 一次完整读写的时序（1Mbps，6个舵机，2字节数据）

```
时间轴 ──────────────────────────────────────────────────────►

① syncReadTx 发送广播包（约 12 字节 × 0.01ms = 0.12ms）
② 每个舵机接收解析，准备回包（固件延迟 ~0.1ms）
③ 6个舵机顺序回包（每包 8 字节 × 0.01ms × 6 = 0.48ms）
④ rxPacket 轮询读取，软件超时 = (6+2)×6 × 0.01ms + 34ms ≈ 34.5ms

总计约 1-5ms（实际受 USB 延迟和 LATENCY_TIMER=16ms 影响较大）

⑤ syncWriteTxOnly 发送广播写包（约 20 字节 × 0.01ms = 0.2ms）
   BROADCAST → 不等回包，立即返回
```

---

## 🔬 实测拆分日志（去噪版）

下面这组数据来自对 `record_loop()` 的细粒度计时拆分。这里去掉了左侧时间戳、文件名、行号，只保留和瓶颈判断有关的字段。

### 首帧（冷启动）

```text
timestamp=1.6s
obs=7.7ms (hw=7.6, proc=0.0, frame=0.1)
inference=1548.7ms
action=2.1ms (proc=0.0, send=0.4, dataset=1.7, display=0.0)
total=1558.5ms
```

说明：

- 首帧主要是 `inference=1548.7ms` 的模型冷启动/预热
- 这一帧不能代表稳态性能

### 稳态帧（关键数据）

```text
obs=4.1ms (hw=4.1, proc=0.0, frame=0.0)action=6.7ms (proc=0.0, send=6.4, dataset=0.2, display=0.0)
obs=12.1ms (hw=12.1, proc=0.0, frame=0.0)action=3.2ms (proc=0.0, send=2.9, dataset=0.2, display=0.0)
obs=6.3ms (hw=6.2, proc=0.0, frame=0.0)action=0.6ms (proc=0.0, send=0.4, dataset=0.2, display=0.0)
obs=3.0ms (hw=3.0, proc=0.0, frame=0.0)action=2.3ms (proc=0.0, send=2.1, dataset=0.2, display=0.0)
obs=4.6ms (hw=4.6, proc=0.0, frame=0.0)action=0.7ms (proc=0.0, send=0.5, dataset=0.2, display=0.0) 

```

### 稳态平均值

基于上面 5 个稳态样本：

- `obs` 平均约 `6.0ms`
- `inference` 平均约 `50.9ms`
- `action` 平均约 `2.7ms`
- `send_action` 平均约 `2.5ms`
- `total` 平均约 `59.7ms`

稳态最大值：

- `action` 最大约 `6.7ms`
- `send_action` 最大约 `6.4ms`

### 结论

这组实测说明了两件事：

1. `action` 里的主要开销确实在 `send_action(...)`
   因为 `dataset` 基本稳定在 `0.2ms`，`proc` 几乎是 `0.0ms`

2. 即使如此，整体主瓶颈仍然是 `inference`
   稳态下 `inference≈43~63ms`，明显高于 `send_action≈0.4~6.4ms`

### 对写链路的新判断

以前仅看协议路径时，会以为 `SYNC_WRITE` 是广播发送、无回包，所以应该只有亚毫秒级延迟。  
但实际测下来，`send_action(...)` 会波动到 `2~6ms`，说明瓶颈不只是“串口发包时间”本身，还可能包含：

- `robot.send_action(...)` 内部的额外逻辑
- 串口驱动 / USB-Serial 适配器调度抖动
- 舵机总线瞬时拥塞或设备侧处理延迟

因此更准确的说法应该是：

- `写链路没有 rxPacket 忙等`
- `写链路的主要时间消耗在 send_action 整体路径`
- `但系统总瓶颈依然首先是 ACT 推理，不是舵机写入`

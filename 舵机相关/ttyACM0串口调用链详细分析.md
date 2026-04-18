# `/dev/ttyACM0` 串口调用链详细分析（带源码注释版）

本文将调用链每一层的真实源码贴出来，加上详细中文注释，逐参数讲透。
涉及文件均已在源码中添加注释，文档内容与源码保持同步。

---

## 一、全局常量与协议定义

`scservo_sdk/scservo_def.py`

```python
BROADCAST_ID = 0xFE  # 254，广播地址，Sync Read / Sync Write 时目标 ID 填此值
MAX_ID = 0xFC        # 252，单播地址最大值
SCS_END = 0          # 全局字节序标志：0 = 小端（Protocol 0），1 = 大端（Protocol 1）

# 指令码（写入数据包 PKT_INSTRUCTION 字段）
INST_PING       = 1    # 0x01：Ping，探测指定 ID 的电机是否在线
INST_READ       = 2    # 0x02：Read，读取单个电机的指定寄存器
INST_WRITE      = 3    # 0x03：Write，写入单个电机（等待状态包）
INST_SYNC_WRITE = 131  # 0x83：Sync Write，广播写多个电机（无响应）
INST_SYNC_READ  = 130  # 0x82：Sync Read，广播读多个电机（各电机顺序应答）

# 通信状态码
COMM_SUCCESS      =  0   # 收发成功
COMM_PORT_BUSY    = -1   # is_using=True，端口正被占用
COMM_TX_FAIL      = -2   # write() 实际写入字节数 ≠ 期望长度
COMM_RX_FAIL      = -3   # 接收失败
COMM_TX_ERROR     = -4   # 数据包字段错误（如超长）
COMM_RX_TIMEOUT   = -6   # isPacketTimeout() 触发
COMM_RX_CORRUPT   = -7   # 已收到数据但校验和不匹配
COMM_NOT_AVAILABLE = -9  # 当前协议版本不支持该指令
```

### 数据包字段偏移常量

`scservo_sdk/protocol_packet_handler.py`

```python
TXPACKET_MAX_LEN = 250  # 发送缓冲区上限（字节）
RXPACKET_MAX_LEN = 250  # 接收缓冲区上限（字节）

# 数据包各字段在 list 中的下标（偏移量）
PKT_HEADER0     = 0   # 固定 0xFF
PKT_HEADER1     = 1   # 固定 0xFF
PKT_ID          = 2   # 目标电机 ID（或广播 ID=0xFE）
PKT_LENGTH      = 3   # 后续字节数（从指令到校验和，不含 Header/ID/LENGTH 本身）
PKT_INSTRUCTION = 4   # 指令码（发送包用）
PKT_ERROR       = 4   # 错误字节（响应包用，与 INSTRUCTION 共用同一偏移位置）
PKT_PARAMETER0  = 5   # 第一个参数字节的起始下标

# 协议层错误位（PKT_ERROR 字段的位掩码）
ERRBIT_VOLTAGE  = 1   # 输入电压异常
ERRBIT_ANGLE    = 2   # 角度传感器异常
ERRBIT_OVERHEAT = 4   # 过热
ERRBIT_OVERELE  = 8   # 过电流
ERRBIT_OVERLOAD = 32  # 过载
```

**关键理解**：`PKT_INSTRUCTION` 和 `PKT_ERROR` 都是偏移量 4，因为发送包第 5 字节是"指令"，接收包第 5 字节是"错误"，两者复用同一下标。

---

## 二、`PortHandler`——串口的唯一入口

文件：`scservo_sdk/port_handler.py`

### 2.1 常量与初始化

```python
LATENCY_TIMER = 16      # USB Full-Speed SOF 轮询间隔（ms），串口固有延迟
DEFAULT_BAUDRATE = 1000000  # Feetech 舵机默认波特率：1 Mbps

class PortHandler(object):
    def __init__(self, port_name):
        """
        串口处理器，封装 pyserial Serial，提供超时管理和读写接口。

        参数:
            port_name (str): 串口设备路径，例如 "/dev/ttyACM0"
        """
        self.is_open = False            # 串口是否已打开
        self.baudrate = DEFAULT_BAUDRATE # 当前波特率，默认 1,000,000 bps
        self.packet_start_time = 0.0    # 最近一次包超时计时开始时刻（ms 浮点）
        self.packet_timeout = 0.0       # 当前包超时时限（ms）
        self.tx_time_per_byte = 0.0     # 每字节发送耗时（ms），用于动态超时计算

        self.is_using = False           # 端口是否正在被某次收发占用（软件并发锁）
        self.port_name = port_name      # 保存串口路径字符串，如 "/dev/ttyACM0"
        self.ser = None                 # pyserial Serial 实例，connect 前为 None
```

**此时什么都没有打开**——只是保存了路径字符串。`self.ser = None` 说明串口文件描述符还没有申请。

---

### 2.2 `openPort()` → `setBaudRate()` → `setupPort()`

这三个函数形成一条链，`openPort` 是入口，真正干活的是 `setupPort`。

```python
def openPort(self):
    """
    打开串口。实际通过 setBaudRate 触发 setupPort 完成物理打开。

    返回:
        bool: 打开成功返回 True，波特率不合法返回 False
    """
    return self.setBaudRate(self.baudrate)  # baudrate = 1,000,000


def setBaudRate(self, baudrate):
    """
    设置波特率并（重新）打开串口。

    参数:
        baudrate (int): 目标波特率，须在白名单内

    返回:
        bool: 成功返回 True；波特率不合法返回 False
    """
    baud = self.getCFlagBaud(baudrate)  # 先校验波特率

    if baud <= 0:
        return False  # 不支持的波特率，直接失败
    else:
        self.baudrate = baudrate
        return self.setupPort(baud)  # 白名单内则直接打开


def getCFlagBaud(self, baudrate):
    """
    校验波特率是否在支持的白名单中。

    参数:
        baudrate (int): 待校验的波特率

    返回:
        int: 合法则返回原值；不合法返回 -1
    """
    if baudrate in [4800, 9600, 14400, 19200, 38400, 57600,
                    115200, 128000, 250000, 500000, 1000000]:
        return baudrate
    else:
        return -1
```

```python
def setupPort(self, cflag_baud):
    """
    真正创建 serial.Serial 对象并打开串口设备文件。

    触发的系统调用：
        open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NONBLOCK)
        ioctl(fd, TCSETS, ...)  — 设置波特率 / 8N1 帧格式

    参数:
        cflag_baud (int): 已通过 getCFlagBaud 校验的波特率值（此处与 baudrate 相同）

    返回:
        bool: 始终返回 True（失败会由 pyserial 直接抛出异常）
    """
    if self.is_open:
        self.closePort()  # 如果已打开，先关闭再重新打开

    self.ser = serial.Serial(
        port=self.port_name,            # 串口设备路径，如 "/dev/ttyACM0"
        baudrate=self.baudrate,          # 波特率，通常为 1,000,000
        bytesize=serial.EIGHTBITS,       # 数据位：8 位（8N1 格式）
        timeout=0                        # 读超时：0 = 非阻塞，有多少读多少
        # parity 默认 NONE（无奇偶校验）
        # stopbits 默认 ONE（1 位停止位）
    )

    self.is_open = True

    self.ser.reset_input_buffer()
    # 清空内核接收缓冲，丢弃之前残留数据（等价于 tcflush(fd, TCIFLUSH)）

    # 计算每字节传输耗时（ms）：
    # 1 帧 = 1 起始位 + 8 数据位 + 1 停止位 = 10 位
    # 耗时 = 1000ms / 波特率 * 10位
    self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
    # 1 Mbps 下：(1000 / 1000000) * 10 = 0.01 ms/byte

    return True
```

**参数逐项解析**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `port` | `"/dev/ttyACM0"` | Linux 下 USB-CDC 串口设备文件路径 |
| `baudrate` | `1000000` | 1 Mbps，Feetech 协议标准速率 |
| `bytesize` | `EIGHTBITS` | 8 数据位，值为整数 8 |
| `timeout` | `0` | **非阻塞**：`read()` 有多少返回多少，不等待 |
| `parity` | `NONE`（默认） | 无奇偶校验 |
| `stopbits` | `ONE`（默认） | 1 位停止位 |

**为什么 timeout=0？** 上层 `rxPacket` 会在 while 循环中多次调用 `readPort`，配合 `isPacketTimeout()` 做整体超时控制，不需要 pyserial 自己阻塞等待。

---

### 2.3 超时管理

```python
def setPacketTimeout(self, packet_length):
    """
    根据数据包长度动态计算超时时限并启动计时器。

    超时公式：tx_time_per_byte * packet_length + LATENCY_TIMER * 2 + 2
    例：0.01ms * 10byte + 16*2ms + 2ms = 34.1ms

    参数:
        packet_length (int): 期望接收的数据包字节数
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (LATENCY_TIMER * 2.0) + 2.0


def setPacketTimeoutMillis(self, msec):
    """
    直接指定超时时限（ms）并启动计时器。

    参数:
        msec (float): 超时时限，单位毫秒
    """
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = msec


def getCurrentTime(self):
    """
    获取当前时刻（ms，纳秒精度）。

    返回:
        float: 当前时间戳，单位毫秒
    """
    return round(time.time() * 1000000000) / 1000000.0
    # time.time() 返回秒 → ×10^9 得纳秒 → 四舍五入 → ÷10^6 得毫秒


def isPacketTimeout(self):
    """
    检查自上次 setPacketTimeout 以来是否已超时。
    超时后将 packet_timeout 清零（防止重复触发）。

    返回:
        bool: 已超时返回 True，否则返回 False
    """
    if self.getTimeSinceStart() > self.packet_timeout:
        self.packet_timeout = 0
        return True
    return False
```

---

### 2.4 读写操作

```python
def readPort(self, length):
    """
    从串口读取最多 length 字节（非阻塞，有多少读多少）。

    参数:
        length (int): 期望读取的字节数

    返回:
        bytes (Python3) 或 list[int] (Python2)
        实际长度可能 < length（非阻塞正常现象）
    """
    if (sys.version_info > (3, 0)):
        return self.ser.read(length)
        # 系统调用：read(fd, buf, length)
    else:
        return [ord(ch) for ch in self.ser.read(length)]


def writePort(self, packet):
    """
    向串口写入一帧完整的协议数据包。

    参数:
        packet (bytearray | list[int]): 已组装好的完整协议帧

    返回:
        int: 实际写入的字节数
    """
    return self.ser.write(packet)
    # 系统调用：write(fd, buf, len)


def clearPort(self):
    """
    刷新发送缓冲区，确保之前残留数据已写入内核。
    在每次 txPacket 发送前调用，防止粘包。
    """
    self.ser.flush()
```

---

## 三、`txPacket`——封包与物理发送

文件：`scservo_sdk/protocol_packet_handler.py`

```python
def txPacket(self, port, txpacket):
    """
    将一帧完整的协议数据包写入串口。

    调用前由调用方填好除 Header 和 Checksum 外的所有字段，此方法负责：
        1. 检查并发锁
        2. 填写包头 0xFF 0xFF
        3. 计算校验和
        4. 调用 port.writePort() 触发 write(fd, ...) 系统调用

    参数:
        port     (PortHandler): 串口处理器（持有 serial.Serial）
        txpacket (list[int]):   预分配好的数据包列表

    返回:
        int: COMM_SUCCESS / COMM_PORT_BUSY / COMM_TX_ERROR / COMM_TX_FAIL
    """
    checksum = 0
    # 总字节数 = PKT_LENGTH字段值 + 4（Header0 Header1 ID LENGTH 各1字节）
    total_packet_length = txpacket[PKT_LENGTH] + 4

    # ── 并发锁检查 ─────────────────────────────────────────────────────────
    if port.is_using:
        return COMM_PORT_BUSY   # 端口正被另一次收发占用，直接拒绝
    port.is_using = True        # 加锁，整个发送+接收过程持有

    # ── 长度越界检查 ───────────────────────────────────────────────────────
    if total_packet_length > TXPACKET_MAX_LEN:  # 超出 250 字节上限
        port.is_using = False
        return COMM_TX_ERROR

    # ── 填写包头 ───────────────────────────────────────────────────────────
    txpacket[PKT_HEADER0] = 0xFF   # 索引 0：固定同步头
    txpacket[PKT_HEADER1] = 0xFF   # 索引 1：固定同步头

    # ── 计算累加校验和 ──────────────────────────────────────────────────────
    # 范围：从 ID（索引2）到倒数第二字节（校验和字节之前）
    for idx in range(2, total_packet_length - 1):
        checksum += txpacket[idx]
    txpacket[total_packet_length - 1] = ~checksum & 0xFF
    # ~checksum 是按位取反，& 0xFF 截断为 8 位

    # ── 物理发送 ───────────────────────────────────────────────────────────
    port.clearPort()                            # flush 发送缓冲，防止粘包
    written_packet_length = port.writePort(txpacket)  # ★ write(fd, buf, len)
    if total_packet_length != written_packet_length:
        port.is_using = False
        return COMM_TX_FAIL  # 实际写入字节数不符

    # 注意：此处不释放 is_using！由后续 rxPacket 在接收完成后释放。
    return COMM_SUCCESS
```

**校验和算法**：
```
checksum = (~(txpacket[2] + txpacket[3] + ... + txpacket[N-2])) & 0xFF

以数据包 [FF FF 01 04 02 38 02 CHK] 为例：
sum  = 0x01 + 0x04 + 0x02 + 0x38 + 0x02 = 0x41
~sum = 0xFFFFFFBE
&0xFF = 0xBE
```

---

## 四、`rxPacket`——从串口接收并验证一帧响应

```python
def rxPacket(self, port):
    """
    从串口读取一帧完整的状态响应包（电机回包）。

    响应包格式（最小 6 字节）：
        [0xFF] [0xFF] [ID] [LEN] [ERR] [DATA...] [CHECKSUM]
           0     1     2    3     4     5~N-2       N-1

    读取策略（非阻塞循环 + 超时）：
        timeout=0 导致 ser.read() 每次可能只返回部分字节，
        使用 while True 循环多次读取直到凑齐完整包。

    参数:
        port (PortHandler): 串口处理器

    返回:
        tuple[list[int], int]:
            rxpacket — 接收到的原始字节列表
            result   — 通信状态码
    """
    rxpacket  = []          # 接收缓冲，逐次 extend
    result    = COMM_TX_FAIL
    checksum  = 0
    rx_length = 0
    wait_length = 6  # 预期包长初值：最小包 6 字节（FF FF ID LEN ERR CHK）

    while True:
        # ── Step 1：非阻塞读，补充缓冲区 ──────────────────────────────────
        rxpacket.extend(port.readPort(wait_length - rx_length))
        # readPort 返回 bytes，extend 后 rxpacket 是 int 列表
        rx_length = len(rxpacket)

        if rx_length >= wait_length:
            # ── Step 2：搜索包头 0xFF 0xFF ──────────────────────────────
            for idx in range(0, (rx_length - 1)):
                if (rxpacket[idx] == 0xFF) and (rxpacket[idx + 1] == 0xFF):
                    break
            # 循环结束后 idx = 0xFF 0xFF 起始位置

            if idx == 0:  # 包头正好在缓冲区起始 → 格式正常

                # ── Step 3：校验字段合法性 ─────────────────────────────
                if (rxpacket[PKT_ID]     > 0xFD  or   # ID 非法（FE=广播，FF=包头）
                    rxpacket[PKT_LENGTH] > RXPACKET_MAX_LEN or  # 长度超限
                    rxpacket[PKT_ERROR]  > 0x7F):      # 错误字节超 7 位
                    del rxpacket[0]   # 丢弃首字节，重新对齐
                    rx_length -= 1
                    continue

                # ── Step 4：按 LEN 字段更新精确包长 ───────────────────
                if wait_length != (rxpacket[PKT_LENGTH] + PKT_LENGTH + 1):
                    wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                    # PKT_LENGTH + 1 = 偏移量3 + 1 = 4，加上 LEN 字段值本身
                    continue

                if rx_length < wait_length:
                    # ── Step 5：还没读完，检查超时 ────────────────────
                    if port.isPacketTimeout():
                        result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                        break
                    else:
                        continue

                # ── Step 6：计算并验证校验和 ───────────────────────────
                for i in range(2, wait_length - 1):
                    checksum += rxpacket[i]
                checksum = ~checksum & 0xFF

                if rxpacket[wait_length - 1] == checksum:
                    result = COMM_SUCCESS   # 校验通过
                else:
                    result = COMM_RX_CORRUPT  # 校验失败
                break

            else:
                # 包头不在起始 → 丢弃前面的垃圾字节
                del rxpacket[0: idx]
                rx_length -= idx

        else:
            # 字节数不够，检查超时
            if port.isPacketTimeout():
                result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                break

    port.is_using = False  # ★ 释放并发锁（txPacket 中加的锁）

    return rxpacket, result
```

**`wait_length` 的精确计算**：
```
响应包结构：[FF][FF][ID][LEN][ERR][DATA×N][CHK]
索引:         0   1   2   3   4   5~4+N   5+N

wait_length = PKT_LENGTH字段值 + PKT_LENGTH偏移量 + 1
            = rxpacket[3]      + 3               + 1
            = rxpacket[3] + 4

例：LEN字段=4（ERR + DATA1 + DATA2 + CHK = 4字节）
wait_length = 4 + 3 + 1 = 8（FF FF ID LEN ERR D0 D1 CHK = 8字节）
```

---

## 五、`txRxPacket`——发送并有条件地接收

```python
def txRxPacket(self, port, txpacket):
    """
    发送指令包，并（如需要）等待状态响应包。

    广播包（PKT_ID=0xFE）：发完即返回，不等响应（Sync Write 走此路径）
    单播包（PKT_ID≠0xFE）：发完后启动超时，循环读取直到收到匹配 ID 的响应

    参数:
        port     (PortHandler): 串口处理器
        txpacket (list[int]):   待发送的完整指令帧

    返回:
        tuple[list[int]|None, int, int]:
            rxpacket — 响应包，广播时为 None
            result   — 通信状态码
            error    — 电机侧错误位（PKT_ERROR 字段），0 表示无错
    """
    rxpacket = None
    error = 0

    # ── 1. 发送指令包 ───────────────────────────────────────────────────────
    result = self.txPacket(port, txpacket)
    if result != COMM_SUCCESS:
        return rxpacket, result, error

    # ── 2. 广播包：释放锁并直接返回 ─────────────────────────────────────────
    if txpacket[PKT_ID] == BROADCAST_ID:
        port.is_using = False  # txPacket 加了锁，这里释放
        return rxpacket, result, error

    # ── 3. 单播包：根据指令类型设置接收超时 ──────────────────────────────────
    if txpacket[PKT_INSTRUCTION] == INST_READ:
        # Read 指令：PKT_PARAMETER0+1 存的是请求的 length 字段
        # 期望响应：length字节数据 + 6字节固定开销（FF FF ID LEN ERR CHK）
        port.setPacketTimeout(txpacket[PKT_PARAMETER0 + 1] + 6)
    else:
        port.setPacketTimeout(6)  # 其他指令：固定 6 字节响应

    # ── 4. 循环接收，过滤掉不匹配 ID 的包 ─────────────────────────────────
    while True:
        rxpacket, result = self.rxPacket(port)
        if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
            break

    # ── 5. 提取电机错误字节 ────────────────────────────────────────────────
    if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
        error = rxpacket[PKT_ERROR]

    return rxpacket, result, error
```

---

## 六、`syncReadTx`——组装并发送 Sync Read 指令

```python
def syncReadTx(self, port, start_address, data_length, param, param_length):
    """
    组装并发送 Sync Read 广播指令包。

    数据包结构（以读 2 个电机 ID=1,2 的 Present_Position 为例）：
        [FF][FF][FE][06][82][38][02][01][02][CHK]
          H0  H1 广播 LEN INST ADDR SZ  ID1 ID2  校验

    LEN = param_length + 4（指令1 + 地址1 + 长度1 + 校验和1）
    发送成功后启动超时计时：(6 + data_length) * param_length ms

    参数:
        port          (PortHandler): 串口处理器
        start_address (int):  目标寄存器起始地址（如 Present_Position=56）
        data_length   (int):  每个电机读取字节数（如 2）
        param         (list[int]): 电机 ID 列表，如 [1, 2, 3]
        param_length  (int):  param 的长度（即电机数量）

    返回:
        int: 通信状态码
    """
    txpacket = [0] * (param_length + 8)
    # 总长 = 电机数（param_length）+ 8 固定字节
    # 8 = FF FF ID LEN INST ADDR DLEN CHK

    txpacket[PKT_ID]          = BROADCAST_ID     # 0xFE，广播给所有电机
    txpacket[PKT_LENGTH]      = param_length + 4 # LEN = 电机数 + 4
    txpacket[PKT_INSTRUCTION] = INST_SYNC_READ   # 0x82
    txpacket[PKT_PARAMETER0 + 0] = start_address # 寄存器地址
    txpacket[PKT_PARAMETER0 + 1] = data_length   # 每电机读取字节数

    # 填入电机 ID 列表
    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0: param_length]

    result = self.txPacket(port, txpacket)
    if result == COMM_SUCCESS:
        # 超时 = 每个响应包大小 × 电机数
        # 每个响应包 = 6字节固定 + data_length字节数据
        port.setPacketTimeout((6 + data_length) * param_length)

    return result
```

---

## 七、`syncWriteTxOnly`——组装并发送 Sync Write 指令

```python
def syncWriteTxOnly(self, port, start_address, data_length, param, param_length):
    """
    组装并发送 Sync Write 广播指令包（不等待响应）。

    数据包结构（以 2 个电机各写 2 字节为例）：
        [FF][FF][FE][09][83][2A][02][01][DC][05][02][D0][07][CHK]
          H0  H1  广播 LEN  指令 地址  长  ID1 D1L D1H ID2 D2L D2H 校验

    参数部分格式：[id1, b0, b1, id2, b0, b1, ...]
    param_length = 电机数 * (1 + data_length)

    由于 PKT_ID=BROADCAST_ID，txRxPacket 发完即返回，不等响应。

    参数:
        port          (PortHandler): 串口处理器
        start_address (int):  目标寄存器起始地址（如 Goal_Position=42）
        data_length   (int):  每个电机写入字节数（如 2）
        param         (list[int]): [id1,b0,b1, id2,b0,b1, ...]
        param_length  (int):  param 总长度（= 电机数 × (1+data_length)）

    返回:
        int: 通信状态码
    """
    txpacket = [0] * (param_length + 8)

    txpacket[PKT_ID]          = BROADCAST_ID    # 广播 ID
    txpacket[PKT_LENGTH]      = param_length + 4
    txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE  # 0x83
    txpacket[PKT_PARAMETER0 + 0] = start_address
    txpacket[PKT_PARAMETER0 + 1] = data_length

    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0: param_length]

    # txRxPacket 检测到广播 ID 后，发完即释放锁并返回
    _, result, _ = self.txRxPacket(port, txpacket)

    return result
```

---

## 八、`GroupSyncRead`——管理同步读的电机列表和数据缓存

文件：`scservo_sdk/group_sync_read.py`

```python
class GroupSyncRead:
    def __init__(self, port, ph, start_address, data_length):
        """
        同步读组，一次广播读取多个电机相同的寄存器。

        参数:
            port          (PortHandler):   串口处理器
            ph            (PacketHandler): 协议包处理器
            start_address (int): 目标寄存器起始地址
            data_length   (int): 每个电机读取的字节数

        注意：motors_bus.py 中创建时传入 start_address=0, data_length=0，
              每次 _setup_sync_reader 调用时再动态更新这两个值。
        """
        self.port = port
        self.ph = ph
        self.start_address = start_address  # 寄存器地址（动态设置）
        self.data_length = data_length      # 读取字节数（动态设置）

        self.last_result = False        # 上一次通信是否成功
        self.is_param_changed = False   # 参数（电机ID列表）是否变更
        self.param = []                 # 发包用参数列表（只含ID）
        self.data_dict = {}             # {电机ID: [接收到的字节列表]}

        self.clearParam()


    def addParam(self, scs_id):
        """
        将一个电机 ID 添加到同步读组（预留接收槽）。

        参数:
            scs_id (int): 电机 ID（0~252）

        返回:
            bool: 成功 True；已存在 False
        """
        if scs_id in self.data_dict:   # 已添加过，不重复
            return False

        self.data_dict[scs_id] = []    # 预留空列表，接收后填充
        self.is_param_changed = True
        return True


    def makeParam(self):
        """
        将 data_dict 中的电机 ID 提取为 param 列表（只含 ID，不含数据）。

        结果：self.param = [id1, id2, id3, ...]
        """
        if not self.data_dict:
            return

        self.param = []
        for scs_id in self.data_dict:
            self.param.append(scs_id)


    def txPacket(self):
        """
        发送 Sync Read 指令包。

        步骤：
            1. 若 data_dict 为空，返回 COMM_NOT_AVAILABLE
            2. 若参数变更，重新 makeParam()
            3. 调用 ph.syncReadTx() 发送广播读指令

        每个电机在 param 中只占 1 字节（ID），故 param_length = 电机数 * 1

        返回:
            int: 通信状态码
        """
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or not self.param:
            self.makeParam()

        return self.ph.syncReadTx(self.port, self.start_address, self.data_length, self.param,
                                  len(self.data_dict.keys()) * 1)
        # param_length = 电机数 * 1（每个 ID 占 1 字节）


    def rxPacket(self):
        """
        接收所有电机的响应包，填充 data_dict。

        逐个电机调用 ph.readRx()，每个电机响应一个独立状态包。

        返回:
            int: 最后一次读取的通信状态码
        """
        self.last_result = False
        result = COMM_RX_FAIL

        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        for scs_id in self.data_dict:
            # readRx 等待并解析一个 PKT_ID == scs_id 的响应包
            # 返回：(data: list[int], result: int, error: int)
            self.data_dict[scs_id], result, _ = self.ph.readRx(self.port, scs_id, self.data_length)
            if result != COMM_SUCCESS:
                return result  # 任何一个电机失败立即返回

        if result == COMM_SUCCESS:
            self.last_result = True

        return result


    def txRxPacket(self):
        """先 txPacket 再 rxPacket，是 _sync_read 的核心入口。"""
        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result
        return self.rxPacket()


    def getData(self, scs_id, address, data_length):
        """
        从 data_dict 中提取指定电机指定地址的已接收数据，拼装为整数。

        参数:
            scs_id      (int): 电机 ID
            address     (int): 要提取的寄存器地址
            data_length (int): 数据字节数（1/2/4）

        返回:
            int: 拼装后的整数值（小端序，使用 SCS_MAKEWORD/SCS_MAKEDWORD）
        """
        if not self.isAvailable(scs_id, address, data_length):
            return 0  # 数据不可用（未读成功或地址越界）

        # 从接收到的字节列表中按偏移量取字节
        # address - self.start_address 得到在 data_dict 列表中的偏移
        if data_length == 1:
            return self.data_dict[scs_id][address - self.start_address]

        elif data_length == 2:
            # SCS_MAKEWORD(低字节, 高字节) → Protocol 0 小端：低字节在前
            return SCS_MAKEWORD(
                self.data_dict[scs_id][address - self.start_address],      # 低字节
                self.data_dict[scs_id][address - self.start_address + 1]   # 高字节
            )
            # 例：[0xAB, 0xCD] → 0xCDAB（小端）

        elif data_length == 4:
            return SCS_MAKEDWORD(
                SCS_MAKEWORD(self.data_dict[scs_id][address - self.start_address + 0],
                             self.data_dict[scs_id][address - self.start_address + 1]),
                SCS_MAKEWORD(self.data_dict[scs_id][address - self.start_address + 2],
                             self.data_dict[scs_id][address - self.start_address + 3])
            )
        else:
            return 0
```

---

## 九、`GroupSyncWrite`——管理同步写的电机数据

文件：`scservo_sdk/group_sync_write.py`

```python
class GroupSyncWrite:
    def __init__(self, port, ph, start_address, data_length):
        """
        同步写组，一次广播写多个电机相同寄存器。

        参数:
            port          (PortHandler):   串口处理器
            ph            (PacketHandler): 协议包处理器
            start_address (int): 目标寄存器地址（动态设置）
            data_length   (int): 每电机写入字节数（动态设置）
        """
        self.port = port
        self.ph = ph
        self.start_address = start_address
        self.data_length = data_length

        self.is_param_changed = False  # 参数是否已变更
        self.param = []                # 已组装的参数字节：[id1,b0,b1, id2,b0,b1, ...]
        self.data_dict = {}            # {电机ID: [byte0, byte1, ...]}


    def addParam(self, scs_id, data):
        """
        添加一个电机的写入数据。

        参数:
            scs_id (int):       电机 ID（0~252）
            data   (list[int]): 要写入的字节列表，长度须 ≤ data_length
                                例：写 Goal_Position=1500 → [0xDC, 0x05]

        返回:
            bool: 成功 True；已存在或超长 False
        """
        if scs_id in self.data_dict:       # 已添加，拒绝重复
            return False
        if len(data) > self.data_length:   # 数据超长
            return False

        self.data_dict[scs_id] = data      # 存储 {id: [b0, b1, ...]}
        self.is_param_changed = True
        return True


    def makeParam(self):
        """
        将 data_dict 展开成 param 字节列表。

        结果格式：[id1, data1_b0, data1_b1, id2, data2_b0, data2_b1, ...]
        每个电机占用 1 + data_length 字节。
        """
        if not self.data_dict:
            return

        self.param = []
        for scs_id in self.data_dict:
            if not self.data_dict[scs_id]:  # 数据为空则停止
                return
            self.param.append(scs_id)               # 先放 ID（1 字节）
            self.param.extend(self.data_dict[scs_id]) # 再放数据字节


    def txPacket(self):
        """
        将所有电机的写入数据打包为 Sync Write 指令并发送。

        param_length = 电机数 * (1 + data_length)

        返回:
            int: 通信状态码
        """
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or not self.param:
            self.makeParam()

        return self.ph.syncWriteTxOnly(
            self.port,
            self.start_address,
            self.data_length,
            self.param,
            len(self.data_dict.keys()) * (1 + self.data_length)
            # param_length = 电机数 * (1字节ID + data_length字节数据)
        )
```

---

## 十、上层：`MotorsBus` 与 `FeetechMotorsBus`

### 10.1 `FeetechMotorsBus.__init__`

文件：`lerobot/src/lerobot/motors/feetech/feetech.py:116-140`

```python
def __init__(
    self,
    port: str,                              # "/dev/ttyACM0"
    motors: dict[str, Motor],               # {"shoulder_pan": Motor(id=1, model="sts3215"), ...}
    calibration: dict[str, MotorCalibration] | None = None,
    protocol_version: int = DEFAULT_PROTOCOL_VERSION,  # 0 = 小端协议
):
    super().__init__(port, motors, calibration)  # 调用 MotorsBus.__init__
    self.protocol_version = protocol_version
    self._assert_same_protocol()
    import scservo_sdk as scs

    # ★ 创建底层对象（串口未打开）
    self.port_handler = scs.PortHandler(self.port)  # 只保存路径，不打开

    # HACK：monkeypatch 修复官方 SDK 的超时计算 bug
    # 原版公式：(tx_time_per_byte * length) + LATENCY_TIMER * 2 + 2
    # 修复版本：(tx_time_per_byte * length) + tx_time_per_byte * 3 + 50
    self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
        self.port_handler, scs.PortHandler
    )

    self.packet_handler = scs.PacketHandler(protocol_version)  # 协议包处理器
    # GroupSyncRead/Write 初始化时 start_address=0, data_length=0
    # 每次 _setup_sync_reader/writer 时动态更新
    self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
    self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

    self._comm_success = scs.COMM_SUCCESS  # = 0
    self._no_error = 0x00
```

---

### 10.2 `MotorsBus.sync_read`

文件：`lerobot/src/lerobot/motors/motors_bus.py:1052-1103`

```python
def sync_read(
    self,
    data_name: str,                         # 寄存器名称，如 "Present_Position"
    motors: str | list[str] | None = None,  # 要读的电机名列表，None=全部
    *,
    normalize: bool = True,                 # 是否将原始值归一化到 [-100, 100]
    num_retry: int = 0,                     # 失败重试次数
) -> dict[str, Value]:                      # 返回 {"shoulder_pan": 45.2, ...}
    """
    读取多个电机相同寄存器的值，一次完整的 Sync Read 操作。

    处理流程：
        1. 检查连接状态
        2. 解析电机名 → ID 列表 + 型号列表
        3. 查控制表获得寄存器地址和字节长度
        4. 调用 _sync_read 执行总线读取
        5. _decode_sign：符号位解码（Feetech 特有）
        6. _normalize：原始编码值 → 用户可用范围
        7. 返回 {电机名: 数值} 字典
    """
    self._assert_protocol_is_compatible("sync_read")  # 协议兼容性检查

    # ── 1. 电机名 → ID + 型号 ─────────────────────────────────────────────
    names  = self._get_motors_list(motors)            # ["shoulder_pan", "elbow_flex", ...]
    ids    = [self.motors[motor].id for motor in names]     # [1, 2, 3, ...]
    models = [self.motors[motor].model for motor in names]  # ["sts3215", "sts3215", ...]

    # ── 2. 查控制表：寄存器地址 + 字节长度 ──────────────────────────────────
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)
    # 例：Present_Position → addr=56, length=2（sts3215）

    # ── 3. 执行总线读取 ──────────────────────────────────────────────────────
    ids_values, _ = self._sync_read(addr, length, ids, num_retry=num_retry,
                                    raise_on_error=True, err_msg=...)
    # 返回：{1: 2048, 2: 1500, ...}（原始编码值）

    # ── 4. 符号位解码 ────────────────────────────────────────────────────────
    ids_values = self._decode_sign(data_name, ids_values)
    # 例：若寄存器值第11位=1，则该值为负数

    # ── 5. 归一化 ────────────────────────────────────────────────────────────
    if normalize and data_name in self.normalized_data:
        ids_values = self._normalize(ids_values)
    # 原始 0~4095 → 用户侧 -100.0 ~ 100.0

    # ── 6. 返回 {电机名: 数值} ────────────────────────────────────────────────
    return {self._id_to_name(id_): value for id_, value in ids_values.items()}
```

---

### 10.3 `MotorsBus._sync_read`

```python
def _sync_read(
    self,
    addr: int,         # 寄存器地址，如 56
    length: int,       # 读取字节数，如 2
    motor_ids: list[int],  # 电机 ID 列表，如 [1, 2, 3]
    *,
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> tuple[dict[int, int], int]:
    # 返回 ({电机ID: 原始值}, 通信状态码)

    # ── A. 配置 GroupSyncRead ──────────────────────────────────────────────
    self._setup_sync_reader(motor_ids, addr, length)
    # 等价于：
    #   sync_reader.clearParam()
    #   sync_reader.start_address = addr
    #   sync_reader.data_length = length
    #   for id_ in motor_ids: sync_reader.addParam(id_)

    # ── B. 执行收发（主要耗时段：5~20ms） ─────────────────────────────────
    for n_try in range(1 + num_retry):
        comm = self.sync_reader.txRxPacket()  # ★ 发送广播读 + 逐个接收响应
        if self._is_comm_success(comm):
            break

    # ── C. 从回包提取各电机的值 ────────────────────────────────────────────
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
    # getData 使用 SCS_MAKEWORD 将字节列表拼成整数

    return values, comm
```

---

### 10.4 `MotorsBus.sync_write`

```python
def sync_write(
    self,
    data_name: str,                           # 寄存器名，如 "Goal_Position"
    values: Value | dict[str, Value],         # {"shoulder_pan": 0.0, ...} 或单个值
    *,
    normalize: bool = True,                   # 是否将用户值反归一化到原始编码
    num_retry: int = 0,
) -> None:
    """写多个电机相同寄存器，无响应（广播写）。"""

    # ── 1. 规整为 {motor_id: value} ──────────────────────────────────────
    ids_values = self._get_ids_values_dict(values)  # {1: 0.0, 2: -50.0, ...}
    models = [self._id_to_model(id_) for id_ in ids_values]

    # ── 2. 查控制表 ──────────────────────────────────────────────────────
    model = next(iter(models))
    addr, length = get_address(self.model_ctrl_table, model, data_name)
    # 例：Goal_Position → addr=42, length=2（sts3215）

    # ── 3. 反归一化：用户值 → 原始编码值 ─────────────────────────────────
    if normalize and data_name in self.normalized_data:
        ids_values = self._unnormalize(ids_values)
    # -100.0~100.0 → 0~4095

    # ── 4. 符号位编码 ─────────────────────────────────────────────────────
    ids_values = self._encode_sign(data_name, ids_values)
    # 负数：取绝对值后置第11位为1

    # ── 5. 执行总线写 ─────────────────────────────────────────────────────
    self._sync_write(addr, length, ids_values, num_retry=num_retry,
                     raise_on_error=True, err_msg=...)
```

---

### 10.5 `MotorsBus._sync_write`

```python
def _sync_write(
    self,
    addr: int,                  # 寄存器地址，如 42
    length: int,                # 写入字节数，如 2
    ids_values: dict[int, int], # {电机ID: 原始编码值}，如 {1: 2048, 2: 1500}
    num_retry: int = 0,
    raise_on_error: bool = True,
    err_msg: str = "",
) -> int:

    # ── A. 配置 GroupSyncWrite ─────────────────────────────────────────────
    self._setup_sync_writer(ids_values, addr, length)
    # 等价于：
    #   sync_writer.clearParam()
    #   sync_writer.start_address = addr
    #   sync_writer.data_length = length
    #   for id_, value in ids_values.items():
    #       data = _serialize_data(value, length)  → [0xDC, 0x05]
    #       sync_writer.addParam(id_, data)

    # ── B. 发送（主要耗时段：5~15ms） ─────────────────────────────────────
    for n_try in range(1 + num_retry):
        comm = self.sync_writer.txPacket()  # ★ 发送广播写，不等响应
        if self._is_comm_success(comm):
            break

    return comm
```

---

### 10.6 字节序列化：`_serialize_data` → `_split_into_byte_chunks`

`motors_bus.py:844` → `feetech.py:69`

```python
def _serialize_data(self, value: int, length: int) -> list[int]:
    """
    将无符号整数转成字节列表，准备放入数据包。

    参数:
        value  (int): 非负整数
        length (int): 字节数（1/2/4）

    返回:
        list[int]: 字节列表，顺序由协议字节序决定
    """
    # 校验范围
    max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
    if value > max_value:
        raise ValueError(...)

    return self._split_into_byte_chunks(value, length)  # Feetech 实现


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    """
    Feetech 协议（小端/Protocol 0）的字节分割实现。

    参数:
        value  (int): 原始编码值，如 1500 = 0x05DC
        length (int): 字节数（1/2/4）

    返回:
        list[int]: 小端字节序，如 length=2 → [0xDC, 0x05]（低字节在前）
    """
    import scservo_sdk as scs

    if length == 1:
        data = [value]
        # 单字节：直接放

    elif length == 2:
        data = [scs.SCS_LOBYTE(value), scs.SCS_HIBYTE(value)]
        # SCS_LOBYTE(1500) = 0xDC（低字节）
        # SCS_HIBYTE(1500) = 0x05（高字节）
        # 小端：[0xDC, 0x05]

    elif length == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),  # 字节0（最低）
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),  # 字节1
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),  # 字节2
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),  # 字节3（最高）
        ]
    return data
```

---

### 10.7 符号位编解码

```python
# feetech.py
def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
    """
    将带符号整数编码为 Feetech 符号幅度格式（发送前调用）。

    参数:
        data_name  (str): 寄存器名称（用于查 encoding_table 判断是否需要编码）
        ids_values (dict): {电机ID: 整数值}，此时 value 可能为负

    返回:
        dict: {电机ID: 编码后的无符号整数}
    """
    for id_ in ids_values:
        model = self._id_to_model(id_)
        encoding_table = self.model_encoding_table.get(model)
        if encoding_table and data_name in encoding_table:
            sign_bit = encoding_table[data_name]  # 符号位位置，通常为 11
            ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)
            # 负数示例：value=-500, sign_bit=11
            # encode → 500 | (1<<11) = 500 | 2048 = 2548
    return ids_values


def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
    """
    将 Feetech 符号幅度编码还原为带符号整数（接收后调用）。

    参数:
        data_name  (str): 寄存器名称
        ids_values (dict): {电机ID: 原始寄存器值}

    返回:
        dict: {电机ID: 带符号整数}
    """
    for id_ in ids_values:
        model = self._id_to_model(id_)
        encoding_table = self.model_encoding_table.get(model)
        if encoding_table and data_name in encoding_table:
            sign_bit = encoding_table[data_name]
            ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)
            # 示例：value=2548, sign_bit=11
            # 第11位=1 → 负数 → -(2548 & ~2048) = -(2548-2048) = -500
    return ids_values
```

---

### 10.8 归一化与反归一化

`lerobot/src/lerobot/motors/motors_bus.py:777-834`

```python
def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
    """
    将原始寄存器值转换为用户侧归一化数值。

    参数:
        ids_values (dict): {电机ID: 原始编码值}，范围如 0~4095

    返回:
        dict: {电机ID: 归一化浮点值}

    归一化公式（RANGE_M100_100 模式）：
        norm = ((val - min) / (max - min)) * 200 - 100
        范围：[-100.0, 100.0]

    drive_mode=True（方向翻转）：norm = -norm
    """
    for id_, val in ids_values.items():
        motor = self._id_to_name(id_)
        min_ = self.calibration[motor].range_min  # 标定最小值
        max_ = self.calibration[motor].range_max  # 标定最大值
        drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode

        bounded_val = min(max_, max(min_, val))   # 限幅

        if norm_mode is RANGE_M100_100:
            norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
            normalized_values[id_] = -norm if drive_mode else norm


def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
    """
    将用户侧归一化数值反变换为原始寄存器编码值（写入前调用）。

    反归一化公式（RANGE_M100_100 模式）：
        raw = int(((val + 100) / 200) * (max - min) + min)
    """
    for id_, val in ids_values.items():
        motor = self._id_to_name(id_)
        min_ = self.calibration[motor].range_min
        max_ = self.calibration[motor].range_max
        drive_mode = ...

        if norm_mode is RANGE_M100_100:
            val = -val if drive_mode else val
            bounded_val = min(100.0, max(-100.0, val))
            unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
```

---

## 十一、完整数据包内存布局图

### Sync Read 请求包（读 3 个电机 Present_Position，addr=56, len=2）

```
字节索引：  0     1     2     3     4     5     6     7     8     9     10
内容：    [FF]  [FF]  [FE]  [07]  [82]  [38]  [02]  [01]  [02]  [03]  [CHK]
字段名：   H0    H1   广播  LEN  INST  ADDR   SZ   ID1   ID2   ID3   校验

LEN = param_length + 4 = 3 + 4 = 7
CHK = ~(0xFE + 0x07 + 0x82 + 0x38 + 0x02 + 0x01 + 0x02 + 0x03) & 0xFF
```

### Sync Read 响应包（电机1回包，Present_Position=2048=0x0800）

```
字节索引：  0     1     2     3     4     5     6     7
内容：    [FF]  [FF]  [01]  [04]  [00]  [00]  [08]  [CHK]
字段名：   H0    H1   ID1   LEN   ERR   D0    D1    校验

LEN = 4（ERR1 + D0 + D1 + CHK1）
D0 = 0x00（低字节），D1 = 0x08（高字节）
position = SCS_MAKEWORD(0x00, 0x08) = 0x0800 = 2048

CHK = ~(0x01 + 0x04 + 0x00 + 0x00 + 0x08) & 0xFF
    = ~0x0D & 0xFF = 0xF2
```

### Sync Write 请求包（写 2 个电机 Goal_Position，addr=42，motor1=1500, motor2=2000）

```
字节索引：  0    1    2    3    4    5    6    7    8    9    10   11   12   13
内容：    [FF] [FF] [FE] [09] [83] [2A] [02] [01] [DC] [05] [02] [D0] [07] [CHK]
字段名：   H0   H1  广播 LEN  INST  地址  长   ID1 D1L  D1H  ID2  D2L  D2H  校验

LEN = param_length + 4 = 6 + 4 - 1 = 9
param_length = 2*(1+2) = 6
motor1: ID=1, position=1500=0x05DC → [0xDC, 0x05]（小端）
motor2: ID=2, position=2000=0x07D0 → [0xD0, 0x07]
广播写：无响应包，发完即返回
```

---

## 十二、调用时序与耗时分布

```
sync_read("Present_Position", 6 个关节) 典型时序：

t=0ms      MotorsBus.sync_read() 入口
t=0.1ms    get_address() 查控制表，addr=56, length=2
t=0.2ms    _setup_sync_reader() 配置 GroupSyncRead
t=0.3ms    GroupSyncRead.txRxPacket() 进入
             ↓ txPacket()
t=0.5ms    txPacket() 组装 10 字节数据包
t=0.6ms    port.clearPort()（flush）
t=0.7ms    port.writePort() → write(fd, buf, 10)
             ↓ USB 传输（Full-Speed，1ms 间隔）
t=1.7ms    Feetech 总线开始响应
             ↓ rxPacket() 开始
t=2~18ms   非阻塞循环读取，每次 read(fd, buf, N)
           6 个电机各回 8 字节，共 48 字节
             ↓
t=~18ms    所有 6 个响应包接收完毕，校验通过
t=18.1ms   getData() 拼装 6 个整数值
t=18.2ms   _decode_sign() 符号位解码
t=18.3ms   _normalize() 归一化，返回 {名称: 浮点值}
```

**瓶颈**：`rxPacket` 循环（`read` 系统调用 + USB 延迟 + 电机响应时间），占总耗时 90%。

---

## 十三、文件索引

| 层级 | 文件 | 关键内容 |
|------|------|---------|
| 配置 | [config_so101_follower.py](lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py) | `port="/dev/ttyACM0"` |
| 总线抽象 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | `sync_read`(L1052), `sync_write`(L1193), `_sync_read`(L1105), `_sync_write`(L1238) |
| Feetech 实现 | [feetech.py](lerobot/src/lerobot/motors/feetech/feetech.py) | `__init__`(L116), `_encode_sign`(L341), `_decode_sign`(L351), `_split_into_byte_chunks`(L69) |
| 串口处理器 | [port_handler.py](scservo_sdk/port_handler.py) | `setupPort`(L91), `readPort`(L57), `writePort`(L63) |
| 协议封包 | [protocol_packet_handler.py](scservo_sdk/protocol_packet_handler.py) | `txPacket`(L69), `rxPacket`(L103), `txRxPacket`(L177), `syncReadTx`(L431), `syncWriteTxOnly`(L450) |
| 同步读组 | [group_sync_read.py](scservo_sdk/group_sync_read.py) | `addParam`(L42), `txPacket`(L74), `rxPacket`(L91), `getData`(L146) |
| 同步写组 | [group_sync_write.py](scservo_sdk/group_sync_write.py) | `addParam`(L31), `makeParam`(L18), `txPacket`(L66) |
| 协议常量 | [scservo_def.py](scservo_sdk/scservo_def.py) | 指令码、错误码、字节序宏 |

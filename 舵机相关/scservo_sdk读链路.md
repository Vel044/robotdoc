# scservo_sdk 读链路：`_setup_sync_reader` → `txRxPacket` → `getData`

以读取 `Present_Position`（地址 56，2 字节，6 个 STS-3215 电机）为例，完整追踪从 Lerobot 入口到 `ser.read()` 的每一步。

---

## 1. 第一步：`_setup_sync_reader` — 告诉 SDK 读什么、读谁

**调用位置**：`motors_bus.py: _sync_read()`

来源：lerobot/src/lerobot/motors/motors_bus.py:_setup_sync_reader（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:  # 定义本链路要说明的函数入口
    self.sync_reader.clearParam()  # 清理上一轮注册参数，避免旧 ID 或旧数据残留
    self.sync_reader.start_address = addr   # 56（Present_Position）  # 保存本链路后续步骤需要使用的中间状态或参数
    self.sync_reader.data_length = length   # 2（2字节）
    for id_ in motor_ids:  # 遍历本次链路涉及的元素
        self.sync_reader.addParam(id_)      # 依次注册 ID=1,2,3,4,5,6
```

`self.sync_reader` 是 `scs.GroupSyncRead` 实例，在 `feetech.py.__init__` 里创建：

来源：lerobot/src/lerobot/motors/feetech/feetech.py:__init__.sync_reader 初始化（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# feetech.py
self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)  # 保存本链路后续步骤需要使用的中间状态或参数
```

### 1.1 `clearParam()` — 清空上次注册的电机

来源：scservo_sdk/group_sync_read.py:clearParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def clearParam(self):  # 定义本链路要说明的函数入口
    self.data_dict.clear()  # 维护电机 ID 到数据字节的映射
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

来源：scservo_sdk/group_sync_read.py:addParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def addParam(self, scs_id):  # 定义本链路要说明的函数入口
    if scs_id in self.data_dict:   # 已存在则拒绝（clearParam 后不会触发）
        return False  # 把本层处理结果返回给调用方
    self.data_dict[scs_id] = []    # 空列表占位，等 rxPacket 填充实际数据
    self.is_param_changed = True   # 标记"下次发送前要重新 makeParam()"
    return True  # 把本层处理结果返回给调用方
```

6 次调用后 `data_dict` 状态：
```
data_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
is_param_changed = True
```

---

## 2. 第二步：`sync_reader.txRxPacket()` — 发广播读包，等 6 个回包

**调用位置**：`motors_bus.py: _sync_read()`

来源：lerobot/src/lerobot/motors/motors_bus.py:_sync_read 调用点（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# motors_bus.py
comm = self.sync_reader.txRxPacket()  # 保存本链路后续步骤需要使用的中间状态或参数
```

来源：scservo_sdk/group_sync_read.py:txRxPacket（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def txRxPacket(self):  # 定义本链路要说明的函数入口
    result = self.txPacket()      # ① 先发广播读指令
    if result != COMM_SUCCESS:  # 检查条件，决定是否进入该分支
        return result  # 把本层处理结果返回给调用方
    return self.rxPacket()        # ② 后逐个接收 6 个电机的应答
```

### 2.1 `txPacket()` — 组装并发送 Sync Read 广播包

来源：scservo_sdk/group_sync_read.py:txPacket（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def txPacket(self):  # 定义本链路要说明的函数入口
    if not self.data_dict:  # 检查条件，决定是否进入该分支
        return COMM_NOT_AVAILABLE  # 把本层处理结果返回给调用方

    if self.is_param_changed or not self.param:  # 检查条件，决定是否进入该分支
        self.makeParam()    # 把 data_dict 的 key 展开为 ID 列表

    # param_length = 电机数量（每个 ID 占 1 字节）
    return self.ph.syncReadTx(  #PacketHandler  # 把本层处理结果返回给调用方
        self.port,              # PortHandler对象，包装了 serial.Serial
        self.start_address,   # 56  # 本行参与当前链路的控制流或数据准备
        self.data_length,     # 2  # 本行参与当前链路的控制流或数据准备
        self.param,           # [1, 2, 3, 4, 5, 6]  # 本行参与当前链路的控制流或数据准备
        len(self.data_dict) * 1  # 6  # 维护电机 ID 到数据字节的映射
    )
```

#### 2.1.1 `makeParam()` — 展开字典为 ID 列表

来源：scservo_sdk/group_sync_read.py:makeParam（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def makeParam(self):  # 定义本链路要说明的函数入口
    self.param = []  # 保存本链路后续步骤需要使用的中间状态或参数
    for scs_id in self.data_dict:  # 遍历本次链路涉及的元素
        self.param.append(scs_id)   # 只有 ID，无数据：[1, 2, 3, 4, 5, 6]
```

注意：Sync Read 的 `param` 只包含 ID，**不包含数据**，因为我们还没读到数据。Sync Write 的 `param` 则是 `[id1, b0, b1, id2, b0, b1, ...]`（见写链路文档）。

#### 2.1.2 `syncReadTx()` — 组装 Sync Read 指令包

来源：scservo_sdk/protocol_packet_handler.py:syncReadTx（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# protocol_packet_handler.py
def syncReadTx(self, port, start_address, data_length, param, param_length):  # 定义本链路要说明的函数入口
    # 包总长 = param_length（ID列表）+ 8（FF FF ID LEN INST ADDR LEN CHK）
    txpacket = [0] * (param_length + 8)         # list[int]  # 保存本链路后续步骤需要使用的中间状态或参数

    txpacket[PKT_ID]          = BROADCAST_ID    # 0xFE，广播
    txpacket[PKT_LENGTH]      = param_length + 4 # LEN = 参数长(6) + 4
    txpacket[PKT_INSTRUCTION] = INST_SYNC_READ   # 0x82  # 读写 Feetech 协议帧的固定字段
    txpacket[PKT_PARAMETER0 + 0] = start_address  # 56，0x38  # 读写 Feetech 协议帧的固定字段
    txpacket[PKT_PARAMETER0 + 1] = data_length    # 2  # 读写 Feetech 协议帧的固定字段
    txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param  # [1,2,3,4,5,6]  # 读写 Feetech 协议帧的固定字段

    result = self.txPacket(port, txpacket)   # 计算校验和并写串口
    if result == COMM_SUCCESS:  # 检查条件，决定是否进入该分支
        # 设置接收超时：每个电机回包 (6+2)=8 字节，6个电机共 48 字节
        port.setPacketTimeout((6 + data_length) * param_length)  # (6+2)*6=48  # 根据预期回包长度设置通信超时时间
    return result  # 把本层处理结果返回给调用方
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

来源：scservo_sdk/protocol_packet_handler.py:txPacket（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# protocol_packet_handler.py
def txPacket(self, port, txpacket):  # 定义本链路要说明的函数入口
    if port.is_using:  # 检查条件，决定是否进入该分支
        return COMM_PORT_BUSY  # 把本层处理结果返回给调用方
    port.is_using = True   # 加并发锁（半双工：发送期间不能接收）

    total_packet_length = txpacket[PKT_LENGTH] + 4  # 10 + 4 = 14 字节

    txpacket[PKT_HEADER0] = 0xFF  # 读写 Feetech 协议帧的固定字段
    txpacket[PKT_HEADER1] = 0xFF  # 读写 Feetech 协议帧的固定字段

    checksum = 0  # 计算或保存 Feetech 协议校验和
    for idx in range(2, total_packet_length - 1):   # ID 到最后数据字节
        checksum += txpacket[idx]  # 计算或保存 Feetech 协议校验和
    txpacket[total_packet_length - 1] = ~checksum & 0xFF   # 校验和

    port.clearPort()   # ser.flush()，避免粘包
    written = port.writePort(txpacket)   # ★ ser.write(packet)  # 保存本链路后续步骤需要使用的中间状态或参数
    if total_packet_length != written:  # 检查条件，决定是否进入该分支
        port.is_using = False  # 保存本链路后续步骤需要使用的中间状态或参数
        return COMM_TX_FAIL  # 把本层处理结果返回给调用方
    # 注意：不在此释放 is_using，由 rxPacket 在收完后释放
    return COMM_SUCCESS  # 把本层处理结果返回给调用方
```

`writePort` 最终调用：
来源：scservo_sdk/port_handler.py:writePort（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# port_handler.py
def writePort(self, packet):  # 定义本链路要说明的函数入口
    return self.ser.write(packet)   # pyserial → write(fd, buf, len) 系统调用
```

读链路的发送阶段也会先进入 pyserial 的 POSIX 后端，把 14 字节 Sync Read 广播帧写到 `/dev/ttyACM0`：

来源：serial/serialposix.py:write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# serial/serialposix.py
def write(self, data):  # 定义本链路要说明的函数入口
    if not self.is_open:  # 检查条件，决定是否进入该分支
        raise PortNotOpenError()  # 调用下一层函数继续完成当前链路动作

    d = to_bytes(data)  # 保存本链路后续步骤需要使用的中间状态或参数
    tx_len = length = len(d)  # 保存本链路后续步骤需要使用的中间状态或参数
    timeout = Timeout(self._write_timeout)  # 维护本次串口读写的超时控制状态

    while tx_len > 0:  # 循环等待或持续处理直到满足退出条件
        try:  # 进入可能触发异常的系统/IO 调用保护区
            # 这里是真正的数据写系统调用：
            # d 是 14 字节 Sync Read 请求帧。
            n = os.write(self.fd, d)   # ★ write(fd, buf, count) syscall  # 触发 write 系统调用，把用户态协议帧交给内核 fd

            if timeout.is_non_blocking:  # 检查条件，决定是否进入该分支
                return n  # 把本层处理结果返回给调用方
            elif not timeout.is_infinite:  # 检查前一分支未命中后的备选条件
                abort, ready, _ = select.select(  # 触发 select/pselect6 就绪检查，等待 fd 可读或可写
                    [self.pipe_abort_write_r],  # 本行参与当前链路的控制流或数据准备
                    [self.fd],  # 本行参与当前链路的控制流或数据准备
                    [],  # 本行参与当前链路的控制流或数据准备
                    timeout.time_left()  # 维护本次串口读写的超时控制状态
                )   # 可能触发 pselect6，等待 fd 可写
                if abort:  # 检查条件，决定是否进入该分支
                    os.read(self.pipe_abort_write_r, 1000)  # 触发 read 系统调用，从内核 fd 取回已到达字节
                    break  # 本行参与当前链路的控制流或数据准备
                if not ready:  # 检查条件，决定是否进入该分支
                    raise SerialTimeoutException('Write timeout')  # 维护本次串口读写的超时控制状态
            else:  # 处理前面条件都不满足的情况
                abort, ready, _ = select.select(  # 触发 select/pselect6 就绪检查，等待 fd 可读或可写
                    [self.pipe_abort_write_r],  # 本行参与当前链路的控制流或数据准备
                    [self.fd],  # 本行参与当前链路的控制流或数据准备
                    [],  # 本行参与当前链路的控制流或数据准备
                    None  # 本行参与当前链路的控制流或数据准备
                )   # 默认 write_timeout=None 时阻塞等待 fd 可写
                if abort:  # 检查条件，决定是否进入该分支
                    os.read(self.pipe_abort_write_r, 1)  # 触发 read 系统调用，从内核 fd 取回已到达字节
                    break  # 本行参与当前链路的控制流或数据准备
                if not ready:  # 检查条件，决定是否进入该分支
                    raise SerialException('write failed (select)')  # 调用下一层函数继续完成当前链路动作

            d = d[n:]  # 保存本链路后续步骤需要使用的中间状态或参数
            tx_len -= n  # 保存本链路后续步骤需要使用的中间状态或参数
        except OSError as e:  # 捕获底层调用失败并转换为上层错误
            if e.errno not in (  # 检查条件，决定是否进入该分支
                errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK,  # 本行参与当前链路的控制流或数据准备
                errno.EINPROGRESS, errno.EINTR  # 本行参与当前链路的控制流或数据准备
            ):  # 本行参与当前链路的控制流或数据准备
                raise SerialException('write failed: {}'.format(e))  # 调用下一层函数继续完成当前链路动作

    return length - len(d)  # 把本层处理结果返回给调用方
```

另外，发送前的 `port.clearPort()` 会进入 `serial/serialposix.py` 的 `flush()`：

来源：serial/serialposix.py:flush（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# serial/serialposix.py
def flush(self):  # 定义本链路要说明的函数入口
    if not self.is_open:  # 检查条件，决定是否进入该分支
        raise PortNotOpenError()  # 调用下一层函数继续完成当前链路动作
    termios.tcdrain(self.fd)   # ★ Linux 上最终进入 ioctl(fd, TCSBRK, 1)
```

至此，14 字节广播读包写入串口，6 个舵机同时收到，各自准备回包。

---

### 2.2 `rxPacket()` — 逐个电机接收应答

来源：scservo_sdk/group_sync_read.py:rxPacket（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def rxPacket(self):  # 定义本链路要说明的函数入口
    self.last_result = False  # 保存本链路后续步骤需要使用的中间状态或参数

    for scs_id in self.data_dict:   # 按 ID 顺序逐个接收
        # readRx: 等待 scs_id 这个电机的回包，提取 data_length 字节数据
        self.data_dict[scs_id], result, _ = self.ph.readRx(  # 维护电机 ID 到数据字节的映射
            self.port, scs_id, self.data_length   # scs_id=1, data_length=2  # 保存本链路后续步骤需要使用的中间状态或参数
        )
        if result != COMM_SUCCESS:  # 检查条件，决定是否进入该分支
            return result   # 某个电机超时/损坏则中止

    self.last_result = True  # 保存本链路后续步骤需要使用的中间状态或参数
    return COMM_SUCCESS  # 把本层处理结果返回给调用方
```

#### 2.2.1 `readRx()` — 等待指定 ID 的回包

来源：scservo_sdk/protocol_packet_handler.py:readRx（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# protocol_packet_handler.py
def readRx(self, port, scs_id, length):  # 定义本链路要说明的函数入口
    result = COMM_TX_FAIL  # 保存本链路后续步骤需要使用的中间状态或参数
    error = 0  # 保存本链路后续步骤需要使用的中间状态或参数
    data = []  # 保存本链路后续步骤需要使用的中间状态或参数

    while True:  # 循环等待或持续处理直到满足退出条件
        rxpacket, result = self.rxPacket(port)   # 非阻塞循环读取一帧完整包
        # 通信失败（超时/损坏），或者收到了匹配 ID 的包：停止循环
        if result != COMM_SUCCESS or rxpacket[PKT_ID] == scs_id:  # 检查条件，决定是否进入该分支
            break  # 本行参与当前链路的控制流或数据准备
        # 收到了其他 ID 的包（如 ID=2 先到），继续等 ID=1

    if result == COMM_SUCCESS and rxpacket[PKT_ID] == scs_id:  # 检查条件，决定是否进入该分支
        error = rxpacket[PKT_ERROR]  # 读写 Feetech 协议帧的固定字段
        # 从参数段提取 length 字节（位置数据）
        data.extend(rxpacket[PKT_PARAMETER0: PKT_PARAMETER0 + length])  # 读写 Feetech 协议帧的固定字段
        # 例：rxpacket = [0xFF,0xFF,0x01,0x04,0x00,0xFD,0x08,CS]
        #     PKT_PARAMETER0=5，length=2
        #     data = [0xFD, 0x08]

    return data, result, error  # 把本层处理结果返回给调用方
```

#### 2.2.2 `rxPacket()` — 从串口非阻塞读取并验证一帧完整包

来源：scservo_sdk/protocol_packet_handler.py:rxPacket（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# protocol_packet_handler.py
def rxPacket(self, port):  # 定义本链路要说明的函数入口
    rxpacket = []  # 保存本链路后续步骤需要使用的中间状态或参数
    wait_length = 6   # 最小包 6 字节（FF FF ID LEN ERR CHK）

    while True:  # 循环等待或持续处理直到满足退出条件
        # 非阻塞读取（timeout=0，有多少读多少）
        rxpacket.extend(port.readPort(wait_length - len(rxpacket)))  # 调用下一层函数继续完成当前链路动作
        rx_length = len(rxpacket)  # 保存本链路后续步骤需要使用的中间状态或参数

        if rx_length >= wait_length:  # 检查条件，决定是否进入该分支
            # 搜索包头 0xFF 0xFF
            for idx in range(0, rx_length - 1):  # 遍历本次链路涉及的元素
                if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF:  # 检查条件，决定是否进入该分支
                    break  # 本行参与当前链路的控制流或数据准备

            if idx == 0:   # 包头在起始位置
                # 字段合法性检查（ID/LEN/ERR 范围）
                if rxpacket[PKT_ID] > 0xFD or rxpacket[PKT_LENGTH] > 250 or rxpacket[PKT_ERROR] > 0x7F:  # 检查条件，决定是否进入该分支
                    del rxpacket[0]   # 丢弃，重新对齐
                    continue  # 本行参与当前链路的控制流或数据准备

                # 按 PKT_LENGTH 字段精确计算包长
                # 完整包长 = PKT_LENGTH + 4（FF FF ID LENGTH 各1字节）
                wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1  # 读写 Feetech 协议帧的固定字段

                if rx_length < wait_length:  # 检查条件，决定是否进入该分支
                    if port.isPacketTimeout():   # 超时检查
                        result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT  # 保存本链路后续步骤需要使用的中间状态或参数
                        break  # 本行参与当前链路的控制流或数据准备
                    continue  # 本行参与当前链路的控制流或数据准备

                # 校验和验证
                checksum = 0  # 计算或保存 Feetech 协议校验和
                for i in range(2, wait_length - 1):  # 遍历本次链路涉及的元素
                    checksum += rxpacket[i]  # 计算或保存 Feetech 协议校验和
                checksum = ~checksum & 0xFF  # 计算或保存 Feetech 协议校验和
                result = COMM_SUCCESS if rxpacket[wait_length - 1] == checksum else COMM_RX_CORRUPT  # 计算或保存 Feetech 协议校验和
                break  # 本行参与当前链路的控制流或数据准备

            else:  # 处理前面条件都不满足的情况
                del rxpacket[0:idx]   # 丢弃包头前的垃圾字节
        else:  # 处理前面条件都不满足的情况
            if port.isPacketTimeout():  # 检查条件，决定是否进入该分支
                result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT  # 保存本链路后续步骤需要使用的中间状态或参数
                break  # 本行参与当前链路的控制流或数据准备

    port.is_using = False   # ★ 释放并发锁
    return rxpacket, result  # 把本层处理结果返回给调用方
```

`readPort` 最终调用：
来源：scservo_sdk/port_handler.py:readPort（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# port_handler.py
def readPort(self, length):  # 定义本链路要说明的函数入口
    return self.ser.read(length)   # pyserial → read(fd, buf, len) 系统调用
    # timeout=0，非阻塞，有多少读多少，可能返回空 bytes
```

继续往下进入 `serial/serialposix.py`。本项目 `serial.Serial(..., timeout=0)`，所以 `select.select()` 的 timeout 是 0：它只做一次立即轮询，fd 可读才继续 `os.read()`。

来源：serial/serialposix.py:read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# serial/serialposix.py
def read(self, size=1):  # 定义本链路要说明的函数入口
    """Read size bytes from the serial port."""  # 本行参与当前链路的控制流或数据准备
    if not self.is_open:  # 检查条件，决定是否进入该分支
        raise PortNotOpenError()  # 调用下一层函数继续完成当前链路动作

    read = bytearray()  # 保存本链路后续步骤需要使用的中间状态或参数
    timeout = Timeout(self._timeout)   # 本项目 timeout=0，非阻塞

    while len(read) < size:  # 循环等待或持续处理直到满足退出条件
        try:  # 进入可能触发异常的系统/IO 调用保护区
            # 先检查串口 fd 是否已有数据可读。
            # 在 Linux 上，select.select 最终通常表现为 pselect6 syscall。
            ready, _, _ = select.select(  # 触发 select/pselect6 就绪检查，等待 fd 可读或可写
                [self.fd, self.pipe_abort_read_r],  # 本行参与当前链路的控制流或数据准备
                [],  # 本行参与当前链路的控制流或数据准备
                [],  # 本行参与当前链路的控制流或数据准备
                timeout.time_left()  # 维护本次串口读写的超时控制状态
            )   # ★ pselect6/select，就绪检查

            if self.pipe_abort_read_r in ready:  # 检查条件，决定是否进入该分支
                os.read(self.pipe_abort_read_r, 1000)  # 触发 read 系统调用，从内核 fd 取回已到达字节
                break  # 本行参与当前链路的控制流或数据准备

            if not ready:  # 检查条件，决定是否进入该分支
                break   # timeout=0 时，无数据会立即走到这里

            # fd 可读后，真正把内核 TTY 缓冲区里的字节拷回用户态。
            buf = os.read(self.fd, size - len(read))   # ★ read(fd, buf, count) syscall  # 触发 read 系统调用，从内核 fd 取回已到达字节
        except OSError as e:  # 捕获底层调用失败并转换为上层错误
            if e.errno not in (  # 检查条件，决定是否进入该分支
                errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK,  # 本行参与当前链路的控制流或数据准备
                errno.EINPROGRESS, errno.EINTR  # 本行参与当前链路的控制流或数据准备
            ):  # 本行参与当前链路的控制流或数据准备
                raise SerialException('read failed: {}'.format(e))  # 调用下一层函数继续完成当前链路动作
        else:  # 处理前面条件都不满足的情况
            if not buf:  # 检查条件，决定是否进入该分支
                raise SerialException(  # 调用下一层函数继续完成当前链路动作
                    'device reports readiness to read but returned no data '  # 本行参与当前链路的控制流或数据准备
                    '(device disconnected or multiple access on port?)'  # 调用下一层函数继续完成当前链路动作
                )
            read.extend(buf)  # 调用下一层函数继续完成当前链路动作

        if timeout.expired():  # 检查条件，决定是否进入该分支
            break  # 本行参与当前链路的控制流或数据准备

    return bytes(read)  # 把本层处理结果返回给调用方
```

如果上层调用 `PortHandler.getBytesAvailable()` 查询接收缓冲区里已经到了多少字节，则走的是同一个 `serial` 文件里的 `ioctl`：

来源：scservo_sdk/port_handler.py:getBytesAvailable（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# port_handler.py
def getBytesAvailable(self):  # 定义本链路要说明的函数入口
    return self.ser.in_waiting  # 把本层处理结果返回给调用方

# serial/serialposix.py
@property  # 本行参与当前链路的控制流或数据准备
def in_waiting(self):  # 定义本链路要说明的函数入口
    s = fcntl.ioctl(self.fd, TIOCINQ, TIOCM_zero_str)  # ★ ioctl(fd, TIOCINQ)  # 触发 ioctl，向 TTY 驱动查询或设置设备状态
    return struct.unpack('I', s)[0]  # 把本层处理结果返回给调用方
```

所以读链路在 `serial` 层触发的关键系统调用是：

| 位置 | serial 代码 | syscall 作用 |
|---|---|---|
| 发送读请求 | `os.write(self.fd, d)` | 发出 14 字节 Sync Read 广播帧 |
| 发送前等待 | `termios.tcdrain(self.fd)` | 等上一帧输出排空，Linux 下最终是 `ioctl(TCSBRK, 1)` |
| 等待回包可读 | `select.select([self.fd, ...], [], [], 0)` | 立即轮询串口 fd，内核侧通常是 `pselect6` |
| 读取回包字节 | `os.read(self.fd, size - len(read))` | 从 `/dev/ttyACM0` 取回舵机状态包字节 |
| 查询可读字节数 | `fcntl.ioctl(self.fd, TIOCINQ, ...)` | 查询 TTY 输入队列已到达字节数 |

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

来源：lerobot/src/lerobot/motors/motors_bus.py:_sync_read 结果提取示意（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# motors_bus.py
values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}  # 保存本链路后续步骤需要使用的中间状态或参数
# addr=56, length=2
```

来源：scservo_sdk/group_sync_read.py:getData（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# group_sync_read.py
def getData(self, scs_id, address, data_length):  # 定义本链路要说明的函数入口
    if not self.isAvailable(scs_id, address, data_length):  # 检查条件，决定是否进入该分支
        return 0  # 把本层处理结果返回给调用方

    # offset = address - start_address = 56 - 56 = 0（从头开始取）
    offset = address - self.start_address  # 保存本链路后续步骤需要使用的中间状态或参数

    if data_length == 1:  # 检查条件，决定是否进入该分支
        return self.data_dict[scs_id][offset]  # 把本层处理结果返回给调用方

    elif data_length == 2:  # 检查前一分支未命中后的备选条件
        # 小端序拼合：低字节 | (高字节 << 8)
        return SCS_MAKEWORD(  # 把本层处理结果返回给调用方
            self.data_dict[scs_id][offset],       # DATA_L = 0xFD  # 维护电机 ID 到数据字节的映射
            self.data_dict[scs_id][offset + 1]    # DATA_H = 0x08  # 维护电机 ID 到数据字节的映射
        )
        # SCS_MAKEWORD(0xFD, 0x08) = 0xFD | (0x08 << 8) = 0x08FD = 2301

    elif data_length == 4:  # 检查前一分支未命中后的备选条件
        return SCS_MAKEDWORD(  # 把本层处理结果返回给调用方
            SCS_MAKEWORD(self.data_dict[scs_id][offset + 0],  # 维护电机 ID 到数据字节的映射
                         self.data_dict[scs_id][offset + 1]),  # 维护电机 ID 到数据字节的映射
            SCS_MAKEWORD(self.data_dict[scs_id][offset + 2],  # 维护电机 ID 到数据字节的映射
                         self.data_dict[scs_id][offset + 3])  # 维护电机 ID 到数据字节的映射
        )
```

`SCS_MAKEWORD` 定义（小端，`SCS_END=0`）：
来源：scservo_sdk/scservo_def.py:SCS_MAKEWORD（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# scservo_def.py
def SCS_MAKEWORD(a, b):  # 定义本链路要说明的函数入口
    if SCS_END == 0:  # 检查条件，决定是否进入该分支
        return (a & 0xFF) | ((b & 0xFF) << 8)   # a=低字节，b=高字节
    else:  # 处理前面条件都不满足的情况
        return (b & 0xFF) | ((a & 0xFF) << 8)  # 把本层处理结果返回给调用方
```

最终返回：
示意代码：为说明链路整理，不作为逐字源码；已按当前源码语义核对。
```python
values = {1: 2301, 2: 1800, 3: 2048, 4: 2148, 5: 2048, 6: 3500}  # 保存本链路后续步骤需要使用的中间状态或参数
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

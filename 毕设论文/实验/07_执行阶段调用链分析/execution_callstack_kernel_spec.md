# 图2规格：串口读写调用链图（直接复用 scservo_sdk 链路文档）

> 对应论文章节：第 3.3.2 节 串口到内核的系统调用路径
> 用途：论文插图，展示读链路和写链路的完整调用链
> 来源：直接复用 `robotdoc/舵机相关/scservo_sdk读链路.md` 和 `scservo_sdk写链路.md` 中的调用链
> 图输出路径：`3_性能分析/image/execution_callstack_kernel.pdf`

---

## 一、图的用途

展示读链路（`get_observation()` → `sync_read`）和写链路（`send_action()` → `sync_write`）的完整调用链。
复用已有文档中的调用链 ASCII 图，直接作为论文插图。

---

## 二、画图规范

| 项目 | 规范 |
|------|------|
| 工具 | draw.io / Mermaid / Excalidraw / Figma，导出 PDF |
| 尺寸 | 宽度与论文正文等宽（`\textwidth`），高度不超过 18cm |
| 布局 | 两个独立的调用链（写链路在上，读链路在下），或左右并排 |
| 节点风格 | 圆角矩形，层之间用箭头连接 |
| 配色 | 写链路蓝色，读链路红色 |

---

## 三、写链路调用链（来自 scservo_sdk写链路.md）

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

**关键帧格式**（图内标注）：
```
FF FF FE 16 83 2A 02  01 83 07  02 99 0A  03 66 08  04 9A 07  05 77 09  06 6C 0D  CS
│  │  │  │  │  │  │   └─────────────────────────────────────────────────────────── ID+数据 × 6
│  │  │  │  │  │  └── data_length = 2
│  │  │  │  │  └───── start_address = 0x2A = 42
│  │  │  │  └──────── INST_SYNC_WRITE = 0x83
│  │  │  └─────────── LEN = 0x16 = 22（param_length=18 + 4）
│  │  └────────────── BROADCAST_ID = 0xFE
└──┴───────────────── Header 0xFF 0xFF
总帧长 = 26 字节
```

---

## 四、读链路调用链（来自 scservo_sdk读链路.md）

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

**关键帧格式**（图内标注）：
```
读包（发送）：FF FF FE 0A 82 38 02 01 02 03 04 05 06 CS
│  │  │  │  │  │  │  └─────────────────── ID 列表：1~6
│  │  │  │  │  │  └── data_length = 2（每电机读 2 字节）
│  │  │  │  │  └───── start_address = 0x38 = 56
│  │  │  │  └──────── INST_SYNC_READ = 0x82
│  │  │  └─────────── LEN = 10（param_length=6 + 4）
│  │  └────────────── BROADCAST_ID = 0xFE
│  └───────────────── Header[1] = 0xFF
└──────────────────── Header[0] = 0xFF

回包（每个电机）：[FF FF ID 04 ERR DATA_L DATA_H CS]
例：ID=1: [FF FF 01 04 00 FD 08 CS] → 0x08FD = 2301
```

---

## 五、关键数据点（图内标注）

| 标注 | 内容 |
|------|------|
| 寄存器地址 | Present_Position=56（读），Goal_Position=42（写） |
| 帧长 | 读包 14 字节，写包 26 字节 |
| 指令码 | 0x82（读），0x83（写） |
| 耗时 | 读 ~1.2 ms/帧（含等待6个回包），写 ~1~3 ms |
| 关键差异 | 写链路发完即返回无回包，读链路半双工等待6个电机按序回包 |

---

## 六、输出

- 文件：`3_性能分析/image/execution_callstack_kernel.pdf`
- 同时生成 `.png` 预览图
- 数据来源：`robotdoc/舵机相关/scservo_sdk写链路.md`、`robotdoc/舵机相关/scservo_sdk读链路.md`
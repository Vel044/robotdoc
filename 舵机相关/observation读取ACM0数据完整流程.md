# SO101 Follower 读取 ACM0 数据完整流程（带传入传出详细版）

---

## 零、ACM0 参数传递链路（从命令行到串口打开）

```
命令行: lerobot-record --robot.port=/dev/ttyACM0
    ↓
[record.py:500] @parser.wrap() → parser 解析命令行参数
    传入: args = ["--robot.port=/dev/ttyACM0", ...]
    传出: cfg = RecordConfig(robot=SO101FollowerConfig(port="/dev/ttyACM0", ...))
    ↓
[record.py:506] robot = make_robot_from_config(cfg.robot)
    传入: config = SO101FollowerConfig(port="/dev/ttyACM0", ...)
    传出: SO101Follower 实例
    ↓
[utils.py:32-35] 根据 config.type="so101_follower" 创建实例
    return SO101Follower(config)
    ↓
[so101_follower.py:45-60] SO101Follower.__init__()
    self.bus = FeetechMotorsBus(port=self.config.port, ...)
    传入: port="/dev/ttyACM0"
    ↓
[feetech.py:128] FeetechMotorsBus.__init__()
    self.port_handler = scs.PortHandler(self.port)
    传入: port="/dev/ttyACM0"
    传出: PortHandler 实例（此时串口未打开，仅保存路径）
    ↓
[record.py:569] robot.connect()
    ↓
[so101_follower.py:93] self.bus.connect()
    ↓
[motors_bus.py:437] self._connect(handshake=True)
    ↓
[motors_bus.py:443] self.port_handler.openPort()
    ↓
[port_handler.py:61] setBaudRate(1_000_000)
    ↓
[port_handler.py:79] setupPort(cflag_baud)
    ↓
[port_handler.py:195-202] serial.Serial(
        port="/dev/ttyACM0",    ★ 这里才真正打开串口设备文件
        baudrate=1_000_000,
        bytesize=serial.EIGHTBITS,
        timeout=0
    )
    ↓
系统调用: open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NONBLOCK)
         ioctl(fd, TCSETS, ...)  # 设置波特率/8N1格式
    ↓
串口已连接，后续可通过 get_observation() 读取数据
```

### 详细代码：命令行参数解析

```python
# lerobot/src/lerobot/record.py:499-500
@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    # @parser.wrap() 装饰器会解析命令行参数
    # 用户执行: lerobot-record --robot.port=/dev/ttyACM0 ...
    # 解析后 cfg.robot = SO101FollowerConfig(port="/dev/ttyACM0", ...)
```

**详细解析流程：**

```python
# lerobot/src/lerobot/configs/parser.py:187-230
def wrap(config_path: Path | None = None):
    """装饰器：解析命令行参数并构造配置对象"""

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)  # 分析被装饰函数的参数
            # argspec.args[0] = 'cfg', argspec.annotations['cfg'] = RecordConfig
            argtype = argspec.annotations[argspec.args[0]]

            # 如果已传入 cfg 对象则直接使用（用于测试或直接调用）
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                # ★ 主要分支：从命令行解析
                cli_args = sys.argv[1:]  # ['--robot.port=/dev/ttyACM0', ...]

                # 1. 加载插件（如果有 --env.discover_packages_path 参数）
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                # PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"
                # 解析后 plugin_args = {}（无插件参数）

                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)  # 动态导入插件包
                    except PluginLoadError as e:
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)  # 移除已处理的参数

                # 2. 过滤掉 .path 参数（特殊处理，如 --policy.path）
                # RecordConfig.__get_path_fields__() 返回 ["policy"]（来自 PreTrainedConfig）
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    # path_fields = ["policy"]
                    cli_args = filter_path_args(path_fields, cli_args)
                    # 移除 --policy.path 参数（稍后在 RecordConfig.__post_init__ 中处理）

                # 3. 使用 draccus 解析命令行参数
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    # 如果指定了 --config_path 且类有 from_pretrained 方法
                    cli_args = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    # ★ 正常流程：调用 draccus.parse
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
                    # config_class = RecordConfig
                    # args = ['--robot.port=/dev/ttyACM0', '--robot.type=so101_follower', ...]

            # 4. 调用原始函数
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner
    return wrapper_outer
```

**draccus.parse 内部流程：**

```python
# venv/lib/python3.13/site-packages/draccus/__init__.py
def parse(
    config_class: Type[T],        # RecordConfig
    config_path: Optional[Union[Path, str]] = None,  # None
    args: Optional[Sequence[str]] = None,  # ['--robot.port=/dev/ttyACM0', '--robot.type=so101_follower', ...]
    prog: Optional[str] = None,
    exit_on_error: bool = True,
    preferred_help: str = HelpOrder.inline,
) -> T:
    """解析命令行参数并返回配置对象实例"""
    # 创建 ArgumentParser
    parser = ArgumentParser(
        config_class=config_class,    # RecordConfig
        config_path=config_path,       # None
        exit_on_error=exit_on_error,
        prog=prog,
        preferred_help=preferred_help,
    )
    # ArgumentParser.__init__ 会遍历 RecordConfig 的所有字段
    # 为每个字段注册 argparse action，包括 robot: RobotConfig
    return parser.parse_args(args)  # 返回 RecordConfig 实例
```

**ArgumentParser._set_dataclass 注册字段：**

```python
# venv/lib/python3.13/site-packages/draccus/argparsing.py
def _set_dataclass(
    self,
    dataclass: Union[Type[Dataclass], Dataclass],
    default: Optional[Union[Dataclass, Dict]] = None,
    dataclass_wrapper_class: Type[DataclassWrapper] = DataclassWrapper,
):
    """为 dataclass 的所有字段添加命令行参数"""
    if not isinstance(dataclass, type):
        default = dataclass if default is None else default
        dataclass = type(dataclass)

    # 创建 DataclassWrapper，它会遍历所有字段并注册到 parser
    new_wrapper = dataclass_wrapper_class(dataclass, default=default, preferred_help=self.preferred_help)
    new_wrapper.register_actions(parser=self.parser)
    # 注册后，解析器可以识别：
    #   --robot.port=/dev/ttyACM0
    #   --robot.type=so101_follower
    #   --robot.cameras="{...}"
    #   --dataset.repo_id=lerobot/test
    #   --dataset.fps=30
    #   --dataset.num_episodes=50
```

**DataclassWrapper 处理 ChoiceRegistry 类型：**

```python
# venv/lib/python3.13/site-packages/draccus/argparsing.py (DataclassWrapper.register_actions)
# 当字段类型是 ChoiceRegistry 时（如 robot: RobotConfig）
# DataclassWrapper 会检测到该类型有 _choice_registry 字典
# 添加 --robot.type 参数，枚举所有已注册的子类：
#   --robot.type {so100_follower,so101_follower,bi_so100_follower,bi_so101_follower,koch_follower,lekiwi,stretch3,viperx,hope_jr_hand,hope_jr_arm,reachy2,mock_robot}

# 子类注册流程：
# so101_follower.py:24
@RobotConfig.register_subclass("so101_follower")  # 装饰器
@dataclass
class SO101FollowerConfig(RobotConfig):
    port: str
# 装饰器执行：RobotConfig._choice_registry["so101_follower"] = SO101FollowerConfig
```

**解析 --robot.type 并创建子类实例：**

```python
# 当用户指定 --robot.type=so101_follower
# draccus 解析器会在 RobotConfig._choice_registry 中查找 "so101_follower"
# 找到 SO101FollowerConfig 类

# 然后继续解析该子类的字段：
#   --robot.port=/dev/ttyACM0  → SO101FollowerConfig.port = "/dev/ttyACM0"
#   --robot.id=black         → SO101FollowerConfig.id = "black"
#   --robot.disable_torque_on_disconnect=True → ...

# 最终构造：
# cfg = RecordConfig(
#     robot=SO101FollowerConfig(port="/dev/ttyACM0", id="black", ...),
#     dataset=DatasetRecordConfig(repo_id=lerobot/test, single_task="...", ...),
#     ...
# )
```

**完整命令行示例：**

```bash
# 用户执行：
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=black \
    --robot.cameras="{handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=lerobot/test \
    --dataset.single_task="Pick the cube" \
    --dataset.fps=30 \
    --dataset.num_episodes=50

# 解析流程：
# 1. sys.argv[1:] = [
#       '--robot.type=so101_follower',
#       '--robot.port=/dev/ttyACM0',
#       '--robot.id=black',
#       '--robot.cameras={handeye: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}',
#       '--dataset.repo_id=lerobot/test',
#       '--dataset.single_task=Pick the cube',
#       '--dataset.fps=30',
#       '--dataset.num_episodes=50'
#   ]

# 2. draccus.parse() 创建 RecordConfig 实例：
#    - 检测 --robot.type=so101_follower
#    - 从 RobotConfig._choice_registry["so101_follower"] 获取 SO101FollowerConfig
#    - 解析 --robot.port=/dev/ttyACM0 → SO101FollowerConfig.port = "/dev/ttyACM0"
#    - 解析 --robot.id=black → SO101FollowerConfig.id = "black"
#    - 解析 --robot.cameras=... → SO101FollowerConfig.cameras = {...}
#    - 解析 --dataset.* 参数到 DatasetRecordConfig
```

### 详细代码：SO101FollowerConfig

```python
# lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py:24-28
@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    port: str  # 串口设备路径，如 "/dev/ttyACM0"
```

### 详细代码：make_robot_from_config（工厂方法）

```python
# lerobot/src/lerobot/robots/utils.py:32-35
elif config.type == "so101_follower":
    from .so101_follower import SO101Follower
    return SO101Follower(config)
# 传入: config = SO101FollowerConfig(port="/dev/ttyACM0", ...)
# 传出: SO101Follower 实例
```

### 详细代码：SO101Follower.__init__（创建 MotorsBus）

```python
# lerobot/src/lerobot/robots/so101_follower/so101_follower.py:45-60
def __init__(self, config: SO101FollowerConfig):
    super().__init__(config)
    self.config = config
    norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

    self.bus = FeetechMotorsBus(
        port=self.config.port,        # ← 传入 "/dev/ttyACM0"
        motors={
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=self.calibration,
    )
```

### 详细代码：FeetechMotorsBus.__init__（创建 PortHandler）

```python
# lerobot/src/lerobot/motors/feetech/feetech.py:116-138
def __init__(
    self,
    port: str,                      # "/dev/ttyACM0"
    motors: dict[str, Motor],
    calibration: dict[str, MotorCalibration] | None = None,
    protocol_version: int = DEFAULT_PROTOCOL_VERSION,
):
    super().__init__(port, motors, calibration)
    self.protocol_version = protocol_version
    self._assert_same_protocol()
    import scservo_sdk as scs

    self.port_handler = scs.PortHandler(self.port)
    # ★ 此时 port_handler 保存了 "/dev/ttyACM0"，
    # 但串口尚未打开（is_open=False，ser=None）
```

### 详细代码：robot.connect()（触发串口打开）

```python
# lerobot/src/lerobot/robots/so101_follower/so101_follower.py:85-105
def connect(self, calibrate: bool = True) -> None:
    """连接机器人硬件：打开串口、校准电机、连接相机"""
    if self.is_connected:
        raise DeviceAlreadyConnectedError(...)

    self.bus.connect()      # ← 调用 MotorsBus.connect()
    # ...
```

### 详细代码：MotorsBus.connect()

```python
# lerobot/src/lerobot/motors/motors_bus.py:421-440
def connect(self, handshake: bool = True) -> None:
    """打开串口并初始化通信"""
    if self.is_connected:
        raise DeviceAlreadyConnectedError(...)

    self._connect(handshake)    # ← 调用内部 _connect
    self.set_timeout()
    logger.debug(f"{self.__class__.__name__} connected.")

def _connect(self, handshake: bool = True) -> None:
    try:
        if not self.port_handler.openPort():   # ← 调用 PortHandler.openPort()
            raise OSError(f"Failed to open port '{self.port}'.")
        elif handshake:
            self._handshake()
    except (FileNotFoundError, OSError, serial.SerialException) as e:
        raise ConnectionError(
            f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
            "\nTry running `lerobot-find-port`\n"
        ) from e
```

### 详细代码：PortHandler.openPort() → setupPort()（真正打开串口）

```python
# scservo_sdk/port_handler.py:32-39
def openPort(self):
    """打开串口。实际通过 setBaudRate 触发 setupPort 完成物理打开。"""
    return self.setBaudRate(self.baudrate)   # 默认 1_000_000 bps

def setBaudRate(self, baudrate):
    """设置波特率并（重新）打开串口"""
    baud = self.getCFlagBaud(baudrate)  # 校验波特率

    if baud <= 0:
        return False
    else:
        self.baudrate = baud
        return self.setupPort(baud)      # ← 调用 setupPort 真正打开

def setupPort(self, cflag_baud):
    """真正创建 serial.Serial 对象并打开串口设备文件"""
    if self.is_open:
        self.closePort()  # 先关闭再重新打开

    # ★ 这里才真正调用 pyserial 打开串口
    self.ser = serial.Serial(
        port=self.port_name,            # "/dev/ttyACM0"
        baudrate=self.baudrate,          # 1_000_000
        bytesize=serial.EIGHTBITS,       # 8 数据位
        timeout=0                        # 非阻塞读
    )

    self.is_open = True
    self.ser.reset_input_buffer()        # 清空接收缓冲

    # 计算每字节传输耗时（ms）：1 帧 = 1 起始位 + 8 数据位 + 1 停止位 = 10 位
    self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
    # 1Mbps 下 = (1000/1000000) * 10 = 0.01 ms/byte

    return True
```

**serial.Serial 内部流程（pyserial 源码）：**

```python
# pyserial/serial/serialposix.py（Linux/MacOS 串口实现）
class Serial(SerialBase):
    def __init__(
        self,
        port=None,
        baudrate=9600,
        bytesize=EIGHTBITS,
        parity=PARITY_NONE,
        stopbits=STOPBITS_ONE,
        timeout=None,
        ...
    ):
        # 1. 端口名称处理
        self.port = port  # "/dev/ttyACM0"

        # 2. 初始化默认参数
        self._baudrate = baudrate          # 1_000_000
        self._bytesize = bytesize          # 8
        self._parity = parity              # 无校验
        self._stopbits = stopbits          # 1

        # 3. timeout 处理
        self._timeout = timeout          # 0（非阻塞）
        self._inter_byte_timeout = None

        # 4. 打开端口（实际系统调用）
        self._isOpen = False
        self.open()

    def open(self):
        """打开串口设备文件"""
        # 4.1 打开设备文件
        self.fd = os.open(
            self.port,                          # "/dev/ttyACM0"
            os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK  # 读写 + 不作为终端 + 非阻塞
        )
        # 返回文件描述符 fd（整数）

        # 4.2 设置波特率
        self._reconfigure_port()

    def _reconfigure_port(self):
        """配置串口参数（通过 ioctl 系统调用）"""
        # 4.2.1 获取当前终端设置
        termios.tcgetattr(self.fd)
        # 返回 termios 结构体，包含 c_iflag, c_oflag, c_cflag, c_lflag, cc[]

        # 4.2.2 构造新配置
        # 清除 c_cflag（控制标志）
        self.cflag = 0
        # 设置波特率：1_000_000 是非标准值，需要特殊处理
        # Linux 下非标准波特率通过 termios.B38400 和自定义分频器实现
        self.cflag |= termios.B38400  # 占位，实际通过分频器设置 1Mbps

        # 设置数据位：8 位
        self.cflag |= termios.CS8

        # 清除校验位标志
        self.cflag &= ~termios.PARENB   # 无校验
        self.cflag &= ~termios.CSTOPB   # 1 停止位

        # 4.2.3 应用配置
        termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, cc])
        # 系统调用：ioctl(fd, TCSETS, ...)
        # TCSETS：立即应用新设置

        # 4.2.4 配置自定义波特率（Linux 专用）
        if self._baudrate == 1000000:
            # 需要设置 USB ACM 设备的波特率分频器
            # ioctl(fd, TIOCSSERIAL, ...)
            # 设置波特率 = 1000000
            pass
```

**系统调用（内核层）：**

```
open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NONBLOCK)
    ↓
[内核] usb_serial_open()
    ├─ 查找 USB ACM 设备
    ├─ 分配 tty 设备号（如 ttyACM0）
    ├─ 创建文件描述符 fd
    └─ 设置 f_op 为 tty 操作函数集
    ↓
ioctl(fd, TCSETS, termios_config)
    ↓
[内核] tty_termios_ioctl() → uart_set_termios()
    ├─ 设置数据位：CS8（8 数据位）
    ├─ 设置停止位：1 位
    ├─ 设置校验：无校验
    └─ 设置波特率：通过 ACM 设备请求
        ↓
[内核] acm_tty_set_termios() → acm_set_line()
    ├─ 构造 USB 控制请求：SET_LINE_CODING
    └─ 通过 USB 发送到设备
        ↓
[STM32 微控制器] 接收波特率配置
    └─ 配置 UART 硬件：1_000_000 bps

后续操作：
    write(fd, buf, len) → [内核] tty_write() → acm_write() → USB 发送
    read(fd, buf, len)  → [内核] tty_read() → acm_read()  → USB 接收
```

**内核层写操作（发送命令到电机）：**

```c
// Linux 内核：drivers/usb/class/cdc-acm.c:814-866
static ssize_t acm_tty_write(struct tty_struct *tty, const u8 *buf, size_t count)
{
    struct acm *acm = tty->driver_data;  // ACM 设备实例
    struct acm_wb *wb;                    // 写缓冲区
    int stat;

    // 1. 分配写缓冲区
    spin_lock_irqsave(&acm->write_lock, flags);
    wbn = acm_wb_alloc(acm);   // 分配一个空闲的写缓冲区（wb）
    if (wbn < 0) {
        spin_unlock_irqrestore(&acm->write_lock, flags);
        return 0;  // 没有可用缓冲区
    }
    wb = &acm->wb[wbn];

    // 2. 限制写入大小（不超过 ACM 块大小）
    count = (count > acm->writesize) ? acm->writesize : count;

    // 3. 复制用户数据到内核缓冲区
    memcpy(wb->buf, buf, count);  // 从用户空间复制
    wb->len = count;

    // 4. 启动 USB 传输
    stat = acm_start_wb(acm, wb);
    spin_unlock_irqrestore(&acm->write_lock, flags);

    if (stat < 0)
        return stat;
    return count;  // 返回写入字节数
}

static int acm_start_wb(struct acm *acm, struct acm_wb *wb)
{
    // 填充 USB URB（USB Request Block）
    usb_fill_bulk_urb(
        wb->urb,
        acm->dev,
        usb_sndbulkpipe(acm->dev, acm->data->out_ep),  // bulk-out 端点
        wb->buf,
        wb->len,
        acm_write_bulk,     // 回调函数（写入完成时调用）
        wb
    );

    // 提交 URB 到 USB 核心
    return usb_submit_urb(wb->urb, GFP_ATOMIC);
}
```

**内核层读操作（接收电机响应）：**

```c
// Linux 内核：drivers/usb/class/cdc-acm.c:516-587
// 读操作由 USB 端点触发，不直接通过 tty_read 调用

// 1. 设备打开时启动读 URB
static int acm_tty_open(struct tty_struct *tty, struct file *filp)
{
    struct acm *acm = tty->driver_data;

    // 为每个读缓冲区启动 URB
    for (i = 0; i < ACM_NR; i++) {
        struct acm_rb *rb = &acm->rb[i];

        usb_fill_bulk_urb(
            rb->urb,
            acm->dev,
            usb_rcvbulkpipe(acm->dev, acm->data->in_ep),  // bulk-in 端点
            rb->buf,
            acm->readsize,
            acm_read_bulk_callback,  // 回调函数
            rb
        );

        usb_submit_urb(rb->urb, GFP_KERNEL);
    }

    return 0;
}

// 2. USB 读取完成回调（异步触发）
static void acm_read_bulk_callback(struct urb *urb)
{
    struct acm_rb *rb = urb->context;
    struct acm *acm = rb->instance;
    int status = urb->status;

    switch (status) {
    case 0:  // 成功读取
        acm_process_read_urb(acm, urb);
        break;
    case -EPIPE:  // 端点阻塞
        set_bit(EVENT_RX_STALL, &acm->flags);
        break;
    // ... 其他错误处理
    }

    // 重新提交 URB（持续监听）
    if (!stopped && !stalled && !cooldown) {
        acm_submit_read_urb(acm, rb->index, GFP_ATOMIC);
    }
}

// 3. 处理读取的数据并推送到 TTY 缓冲区
static void acm_process_read_urb(struct acm *acm, struct urb *urb)
{
    unsigned long flags;

    if (!urb->actual_length)
        return;

    spin_lock_irqsave(&acm->read_lock, flags);

    // 将 USB 数据推送到 TTY 翻转缓冲区
    tty_insert_flip_string(&acm->port, urb->transfer_buffer, urb->actual_length);
    // 数据从 USB 缓冲区 → TTY 线路规程缓冲区

    spin_unlock_irqrestore(&acm->read_lock, flags);

    // 通知 TTY 层有新数据
    tty_flip_buffer_push(&acm->port);
    // 唤醒阻塞在 read(fd, ...) 的用户空间进程
}
```

**用户空间 read 调用流程：**

```
用户空间：
    bytes_read = read(fd, buf, N)  // 阻塞等待数据
    ↓
系统调用：
    sys_read(fd, buf, N)
    ↓
VFS 层：
    vfs_read() → tty_read()
    ↓
TTY 层：
    n_tty_read()
    ├─ 检查 tty 缓冲区是否有数据
    ├─ 如果无数据，进程阻塞（加入等待队列）
    └─ 当 tty_flip_buffer_push() 调用时，唤醒进程
    ↓
数据返回：
    复制 tty 缓冲区数据到用户空间 buf
    返回实际读取字节数
```

**STM32 微控制器（USB 固件）：**

```
[STM32 微控制器] 接收 USB 数据
    ↓
USB 中断服务程序（ISR）
    └─ 检测到 bulk-out 端点数据到达
    ↓
STM32 USB 驱动：
    usb_ep_receive_handler()
    └─ 读取 USB 数据到 DMA 缓冲区
    ↓
数据解析：
    usb_packet_callback()
    └─ 遍历接收到的包：
        - 检查包头：0xFF 0xFF
        - 检查 ID：是否匹配本设备 ID（1~6）
        - 检查指令码：
          * 0x82 = INST_SYNC_READ  → 读寄存器命令
          * 0x83 = INST_SYNC_WRITE → 写寄存器命令
    ↓
读寄存器处理（Present_Position=56，长度=2）：
    handle_sync_read(addr=56, length=2, data=[1,2,3,4,5,6])
    for motor_id in [1,2,3,4,5,6]:
        if motor_id == self.id:
            // 从 UART 硬件读取当前寄存器值
            position = read_uart_register(56)  // 读 2 字节
            break
    ↓
构造响应包：
    response = [0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08, CHK]
    // ID=1, LEN=4, ERR=0, DATA_LOW=0x00, DATA_HIGH=0x08

    // 计算校验和
    checksum = ~(0x01 + 0x04 + 0x00 + 0x00 + 0x08) & 0xFF
    response[7] = checksum  // 0xF2

    // 通过 USB bulk-in 端点发送
    usb_transmit(response)
    ↓
[主机] 通过 acm_read_bulk_callback() 接收响应
```

**STM32 UART 寄存器读取（底层硬件）：**

```
handle_sync_read(addr=56, length=2)
    ↓
检查地址 56 = Present_Position（2 字节）
    ↓
访问 UART/串口接收数据：
    - STM32 的 UART 外设接收来自电机的 RS485 数据
    - 或通过 I2C/SPI 接口读取
    ↓
返回位置编码：
    // sts3215 电机返回 12-bit 编码的位置
    // 范围 0~4095，对应 0°~360°
    return [low_byte, high_byte]  // 小端格式
```

**完整数据流向（主机 → 电机 → 主机）：**

```
用户空间 Python：
    obs = robot.get_observation()
    ↓
port_handler.writePort([FF FF FE 0A 82 38 02 01 02 03 04 05 06 20])
    ↓
sys_write(fd, buf, 14)
    ↓
[VFS] → [TTY 层] → [ACM 驱动]
    ↓
[USB 核心] usb_submit_urb() → [USB 主机]
    ↓
[USB 总线] +5V 数据线传输
    ↓
[STM32 USB 固件] 接收并解析
    ↓
[STM32 UART/总线] → [sts3215 舵机] 读取寄存器 56
    ↓
[sts3215 硬件] 返回 Present_Position 值 0x0800 (2048)
    ↓
[STM32] 构造响应包并 USB 发送
    ↓
[USB 总线] ← 接收响应包 [FF FF 01 04 00 00 08 F2]
    ↓
[USB 主机] → [ACM 驱动] acm_read_bulk_callback()
    ↓
[TTY 缓冲区] tty_insert_flip_string() → 唤醒 read()
    ↓
sys_read(fd, buf, N) → 返回数据到用户空间
    ↓
port_handler.readPort(8) → bytes b'\x00\x08'
    ↓
group_sync_read.getData(1, 56, 2) → SCS_MAKEWORD(0x00, 0x08) = 2048
    ↓
motors_bus._normalize() → ((2048-100)/(3900-100))*200-100 = 2.53
    ↓
最终返回：
    obs["shoulder_pan.pos"] = 2.53
```

**USB ACM 设备驱动流程：**

```c
// Linux 内核：drivers/usb/class/cdc-acm.c
static int acm_tty_open(struct tty_struct *tty, struct file *filp)
{
    // ACM 设备打开时执行
    struct acm *acm = tty->driver_data;

    // 初始化 USB 端点
    // - bulk-in 端点：接收数据（电机响应）
    // - bulk-out 端点：发送数据（命令）
    // - interrupt-in 端点：状态通知

    // 启动 USB 读取 URB
    usb_submit_urb(acm->readurb);

    return 0;
}

static int acm_set_line(struct acm *acm)
{
    // 设置串口线路编码（波特率、数据位、停止位、校验）
    struct usb_cdc_line_coding linecoding;

    linecoding.dwDTERate = cpu_to_le32(acm->line.dwDTERate);  // 1000000
    linecoding.bCharFormat = acm->line.bCharFormat;          // 停止位：0=1bit
    linecoding.bParityType = acm->line.bParityType;          // 校验：0=无
    linecoding.bDataBits = acm->line.bDataBits;            // 数据位：8

    // USB 控制传输：SET_LINE_CODING 请求
    return usb_control_msg(
        acm->dev,
        usb_sndctrlpipe(acm->dev, 0),
        USB_CDC_REQ_SET_LINE_CODING,  // 0x20
        USB_TYPE_CLASS | USB_RECIP_INTERFACE,
        0,
        acm->ctrlif,  // 接口号
        &linecoding,   // 编码参数
        7,            // 长度
        1000           // 超时 ms
    );
}
```

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

## 十二、ACM0 参数传递关键文件与行号

| 步骤 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 命令行解析 | [record.py](lerobot/src/lerobot/record.py) | 499-500 | `@parser.wrap()` 装饰器解析 --robot.port |
| 配置类 | [config_so101_follower.py](lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py) | 24-28 | `SO101FollowerConfig(port: str)` |
| 工厂方法 | [utils.py](lerobot/src/lerobot/robots/utils.py) | 32-35 | `make_robot_from_config()` |
| 机器人初始化 | [so101_follower.py](lerobot/src/lerobot/robots/so101_follower/so101_follower.py) | 45-60 | `FeetechMotorsBus(port="/dev/ttyACM0")` |
| MotorsBus 初始化 | [feetech.py](lerobot/src/lerobot/motors/feetech/feetech.py) | 116-138 | `PortHandler("/dev/ttyACM0")` |
| 机器人连接 | [so101_follower.py](lerobot/src/lerobot/robots/so101_follower/so101_follower.py) | 85-93 | `robot.connect()` → `bus.connect()` |
| 串口打开 | [motors_bus.py](lerobot/src/lerobot/motors/motors_bus.py) | 421-443 | `connect()` → `_connect()` → `openPort()` |
| 物理打开串口 | [port_handler.py](scservo_sdk/port_handler.py) | 178-212 | `setupPort()` → `serial.Serial("/dev/ttyACM0")` |

---

## 十三、数据读取关键文件与行号

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

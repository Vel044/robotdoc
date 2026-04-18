# Linux 内核 ioctl 系统调用完整流程分析

## 一、ioctl cmd 编码格式（_IOC 宏）

**文件路径**: `linux/include/uapi/asm-generic/ioctl.h`

```c
/* ioctl command encoding: 32 bits total */
#define _IOC_NRBITS   8      /* 命令号位数 */
#define _IOC_TYPEBITS 8      /* 类型位数 */
#define _IOC_SIZEBITS 14     /* 参数大小位数 */
#define _IOC_DIRBITS   2      /* 方向位数 */

#define _IOC_NRSHIFT   0
#define _IOC_TYPESHIFT (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT  (_IOC_SIZESHIFT + _IOC_SIZEBITS)

/* 方向定义 */
#define _IOC_NONE  0U  /* 无数据传输 */
#define _IOC_WRITE 1U  /* 用户态写，内核态读 */
#define _IOC_READ  2U  /* 用户态读，内核态写 */

/* 构造 ioctl cmd 的宏 */
#define _IOC(dir, type, nr, size) \
    (((dir)  << _IOC_DIRSHIFT) | \
     ((type) << _IOC_TYPESHIFT) | \
     ((nr)   << _IOC_NRSHIFT) | \
     ((size) << _IOC_SIZESHIFT))

/* 常用的 ioctl 构造宏 */
#define _IO(type, nr)        _IOC(_IOC_NONE, (type), (nr), 0)
#define _IOR(type, nr, size) _IOC(_IOC_READ, (type), (nr), (_IOC_TYPECHECK(size)))
#define _IOW(type, nr, size) _IOC(_IOC_WRITE, (type), (nr), (_IOC_TYPECHECK(size)))
#define _IOWR(type, nr, size) _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), (_IOC_TYPECHECK(size)))

/* 解码 ioctl cmd 的宏 */
#define _IOC_DIR(nr)   (((nr) >> _IOC_DIRSHIFT) & _IOC_DIRMASK)
#define _IOC_TYPE(nr)  (((nr) >> _IOC_TYPESHIFT) & _IOC_TYPEMASK)
#define _IOC_NR(nr)    (((nr) >> _IOC_NRSHIFT) & _IOC_NRMASK)
#define _IOC_SIZE(nr)  (((nr) >> _IOC_SIZESHIFT) & _IOC_SIZEMASK)
```

**32位 cmd 的内存布局**:
```
| 31 30 | 29 ... 16 | 15 ... 8 | 7 ... 0 |
| DIR   |   SIZE    |   TYPE   |   NR    |
```

### 常用 TTY ioctl cmd 示例

| cmd | 值 | 类型 | 方向 | 含义 |
|-----|-----|------|------|------|
| TCSBRK | 0x5409 | 'T'(0x54) | 无 | 发送 break / tcdrain |
| TIOCINQ | 0x541B | 'T'(0x54) | 读 | 获取输入队列字节数 |
| FIONREAD | 0x541B | 'T'(0x54) | 读 | 同 TIOCINQ |
| TIOCOUTQ | 0x5411 | 'T'(0x54) | 读 | 获取输出队列字节数 |
| TIOCMSET | 0x541C | 'T'(0x54) | 读写 | 设置调制解调器标志 |

---

## 二、从用户态到内核的完整调用链

### 2.1 系统调用入口

**文件路径**: `linux/fs/ioctl.c`

```c
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)
{
    struct fd f = fdget(fd);
    int error;

    if (!fd_file(f))
        return -EBADF;

    error = security_file_ioctl(fd_file(f), cmd, arg);
    if (error)
        goto out;

    error = do_vfs_ioctl(fd_file(f), fd, cmd, arg);
    if (error == -ENOIOCTLCMD)
        error = vfs_ioctl(fd_file(f), cmd, arg);

out:
    fdput(f);
    return error;
}
```

### 2.2 VFS 层处理

**文件路径**: `linux/fs/ioctl.c`

```c
long vfs_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    int error = -ENOTTY;

    if (!filp->f_op->unlocked_ioctl)
        goto out;

    error = filp->f_op->unlocked_ioctl(filp, cmd, arg);
    if (error == -ENOIOCTLCMD)
        error = -ENOTTY;
out:
    return error;
}
```

### 2.3 TTY 层的 ioctl 路由

对于 TTY 设备，`f_op->unlocked_ioctl` 指向 `tty_ioctl`。TTY 层的 ioctl 分发顺序为:

1. **tty 层** (`tty_ioctl.c`) - 处理通用 TTY ioctl
2. **tty 驱动层** (`serial_core.c`) - 处理 UART 特定 ioctl
3. **行规程层** (`n_tty_ioctl_helper`) - 处理行规程特定 ioctl

---

## 三、tty_struct、tty_ldisc、tty_port 的关系

### 3.1 tty_struct（TTY 设备核心结构）

**文件路径**: `linux/include/linux/tty.h`

```c
struct tty_struct {
    struct kref kref;              /* 引用计数 */
    int index;                     /* 设备索引号 */
    struct device *dev;             /* 设备对象 */
    struct tty_driver *driver;     /* TTY 驱动程序 */
    struct tty_port *port;         /* TTY 端口（持久存储） */
    const struct tty_operations *ops; /* 驱动程序操作集 */

    struct tty_ldisc *ldisc;       /* 当前行规程 */
    struct ld_semaphore ldisc_sem; /* 保护行规程切换 */

    /* 各种锁 */
    struct mutex atomic_write_lock;   /* 保护写操作 */
    struct mutex legacy_mutex;         /* 历史遗留锁 */
    struct mutex throttle_mutex;       /* 节流锁 */
    struct rw_semaphore termios_rwsem; /* 保护 termios */

    struct ktermios termios, termios_locked; /* 终端配置 */
    unsigned long flags;              /* TTY 标志位 */
    int count;                       /* 打开计数 */

    /* 缓冲区信息 */
    unsigned int receive_room;       /* 接收缓冲区剩余空间 */

    /* 等待队列 */
    wait_queue_head_t write_wait;    /* 写等待队列 */
    wait_queue_head_t read_wait;    /* 读等待队列 */

    void *disc_data;                 /* 行规程私有数据 */
    void *driver_data;               /* 驱动私有数据（如 uart_state） */

    /* 写缓冲区 */
    int write_cnt;
    u8 *write_buf;
};
```

### 3.2 tty_ldisc（行规程）

**文件路径**: `linux/include/linux/tty_ldisc.h`

```c
struct tty_ldisc_ops {
    char *name;
    int num;   /* N_TTY = 0, N_HDLC = 1, etc. */

    /* 从 TTY 层调用 */
    int  (*open)(struct tty_struct *tty);
    void (*close)(struct tty_struct *tty);
    void (*flush_buffer)(struct tty_struct *tty);
    ssize_t (*read)(struct tty_struct *tty, struct file *file, u8 *buf, size_t nr, ...);
    ssize_t (*write)(struct tty_struct *tty, struct file *file, const u8 *buf, size_t nr);
    int  (*ioctl)(struct tty_struct *tty, unsigned int cmd, unsigned long arg);
    void (*set_termios)(struct tty_struct *tty, const struct ktermios *old);
    __poll_t (*poll)(struct tty_struct *tty, struct file *file, struct poll_table_struct *wait);
    void (*hangup)(struct tty_struct *tty);

    /* 从驱动层调用 */
    void (*receive_buf)(struct tty_struct *tty, const u8 *cp, const u8 *fp, size_t count);
    void (*write_wakeup)(struct tty_struct *tty);
    size_t (*receive_buf2)(struct tty_struct *tty, const u8 *cp, const u8 *fp, size_t count);

    struct module *owner;
};

struct tty_ldisc {
    struct tty_ldisc_ops *ops;
    struct tty_struct *tty;
};
```

### 3.3 tty_port（端口抽象）

**文件路径**: `linux/include/linux/tty_port.h`

```c
struct tty_port {
    struct tty_bufhead buf;        /* TTY 缓冲区头 */
    struct tty_struct *tty;        /* 关联的 tty 结构 */
    struct tty_struct *itty;        /* 内部使用的 tty */
    const struct tty_port_operations *ops; /* 端口操作 */

    spinlock_t lock;               /* 保护 tty */
    int blocked_open;               /* 等待打开的进程数 */
    int count;                      /* 使用计数 */

    wait_queue_head_t open_wait;   /* 等待打开完成 */
    wait_queue_head_t delta_msr_wait; /* 调制解调器状态变化等待 */

    unsigned long flags;           /* 用户标志 ASYNC_ */
    unsigned long iflags;          /* 内部标志 TTY_PORT_ */

    struct mutex mutex;             /* 打开/关闭互斥 */
    struct mutex buf_mutex;         /* 缓冲区互斥 */

    u8 *xmit_buf;                  /* 传输缓冲区 */
    DECLARE_KFIFO_PTR(xmit_fifo, u8); /* 传输 FIFO */

    int drain_delay;                /* 排空延迟 */
    struct kref kref;              /* 引用计数 */
};
```

### 3.4 三者关系图

```
┌─────────────────────────────────────────────────────────────┐
│                         tty_struct                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │    driver    │──▶│    port     │──▶│      ldisc        │  │
│  │ (tty_driver)│  │ (tty_port)  │  │   (tty_ldisc)     │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│         │                 │                    │             │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 硬件寄存器   │  │   缓冲区     │  │   n_tty_data      │  │
│  │ (uart_port)  │  │ (tty_buffer) │  │ (读/写缓冲区)     │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、TCSBRK、TIOCINQ、FIONREAD 等 cmd 的处理

### 4.1 TIOCINQ / FIONREAD（获取输入队列字节数）

**文件路径**: `linux/drivers/tty/n_tty.c`

```c
static int n_tty_ioctl(struct tty_struct *tty, unsigned int cmd,
		       unsigned long arg)
{
    struct n_tty_data *ldata = tty->disc_data;
    unsigned int num;

    switch (cmd) {
    case TIOCOUTQ:
        return put_user(tty_chars_in_buffer(tty), (int __user *) arg);
    case TIOCINQ:
        down_write(&tty->termios_rwsem);
        if (L_ICANON(tty) && !L_EXTPROC(tty))
            num = inq_canon(ldata);  /* 规范模式：计算换行符之间的字节 */
        else
            num = read_cnt(ldata);   /* 非规范模式：直接返回计数 */
        up_write(&tty->termios_rwsem);
        return put_user(num, (unsigned int __user *) arg);
    default:
        return n_tty_ioctl_helper(tty, cmd, arg);
    }
}
```

### 4.2 TCSBRK（发送 break 信号）

**文件路径**: `linux/drivers/tty/tty_ioctl.c`

```c
long tty_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct tty_struct *tty = file_tty(file);
    switch (cmd) {
    case TCSBRK:   /* SVID version: non-zero arg --> no break */
        if (!arg)
            tty->ops->break_ctl(tty, -1);  // arg=0 → 发送 break 信号
        // arg ≠ 0 → 不发送 break，只等待输出排空（tcdrain 走这个分支）
        return tty_wait_until_sent(tty, 0);
    case TIOCINQ:  /* FIONREAD */
        return n_tty_ioctl(tty, cmd, arg);
    }
}
```

实际的 break 控制由 UART 驱动处理：

**文件路径**: `linux/drivers/tty/serial/serial_core.c`

```c
static int uart_break_ctl(struct tty_struct *tty, int break_state)
{
    struct uart_state *state = tty->driver_data;
    struct uart_port *uport;

    uport = uart_port_check(state);
    if (!uport)
        return -EIO;

    if (uport->type != PORT_UNKNOWN && uport->ops->break_ctl)
        uport->ops->break_ctl(uport, break_state);  /* 调用驱动发送 break */

    return 0;
}
```

---

## 五、tty_wait_until_sent 函数分析

**文件路径**: `linux/drivers/tty/tty_ioctl.c`

```c
void tty_wait_until_sent(struct tty_struct *tty, long timeout)
{
    if (!timeout)
        timeout = MAX_SCHEDULE_TIMEOUT;

    /* 等待直到输出缓冲区为空 */
    timeout = wait_event_interruptible_timeout(tty->write_wait,
            !tty_chars_in_buffer(tty), timeout);
    if (timeout <= 0)
        return;

    if (timeout == MAX_SCHEDULE_TIMEOUT)
        timeout = 0;

    /* 让驱动完成传输 */
    if (tty->ops->wait_until_sent)
        tty->ops->wait_until_sent(tty, timeout);
}
```

**UART 层的实现**:

**文件路径**: `linux/drivers/tty/serial/serial_core.c`

```c
static void uart_wait_until_sent(struct tty_struct *tty, int timeout)
{
    struct uart_state *state = tty->driver_data;
    struct uart_port *port;
    unsigned long char_time, expire, fifo_timeout;

    port = uart_port_ref(state);
    if (!port)
        return;

    if (port->type == PORT_UNKNOWN || port->fifosize == 0) {
        uart_port_deref(port);
        return;
    }

    /*
     * 计算每个字符的时间：
     * char_time = (帧大小 * 10) / 波特率
     * 帧大小 = 数据位 + 校验位 + 停止位
     * 例如 115200 波特率，8N1：char_time = 10/115200 ≈ 87us
     */
    char_time = (uart_fifo_timeout(port) + port->fifosize) / port->fifosize;

    fifo_timeout = uart_fifo_timeout(port);
    if (timeout == 0 || timeout > 2 * fifo_timeout)
        timeout = 2 * fifo_timeout;

    expire = jiffies + timeout;

    /* 轮询检查发送器是否为空 */
    while (!port->ops->tx_empty(port)) {
        msleep_interruptible(jiffies_to_msecs(char_time));
        if (signal_pending(current))
            break;
        if (time_after(jiffies, expire))
            break;
    }

    uart_port_deref(port);
}
```

---

## 六、n_tty_read 和 n_tty_write 的实现

### 6.1 n_tty_read（读函数）

**文件路径**: `linux/drivers/tty/n_tty.c`

```c
static ssize_t n_tty_read(struct tty_struct *tty, struct file *file, u8 *kbuf,
              size_t nr, void **cookie, unsigned long offset)
{
    struct n_tty_data *ldata = tty->disc_data;
    // ...

    /* 1. 检查作业控制 */
    retval = job_control(tty, file);
    if (retval < 0)
        return retval;

    /* 2. 获取原子读锁 */
    if (file->f_flags & O_NONBLOCK) {
        if (!mutex_trylock(&ldata->atomic_read_lock))
            return -EAGAIN;
    } else {
        if (mutex_lock_interruptible(&ldata->atomic_read_lock))
            return -ERESTARTSYS;
    }

    down_read(&tty->termios_rwsem);

    /* 3. 设置超时（规范模式下的 MIN 和 TIME） */
    minimum = time = 0;
    timeout = MAX_SCHEDULE_TIMEOUT;
    if (!ldata->icanon) {
        minimum = MIN_CHAR(tty);
        if (minimum) {
            time = (HZ / 10) * TIME_CHAR(tty);
        } else {
            timeout = (HZ / 10) * TIME_CHAR(tty);
            minimum = 1;
        }
    }

    add_wait_queue(&tty->read_wait, &wait);
    while (nr) {
        /* 4. 检查数据包模式状态变化 */
        if (packet && tty->link->ctrl.pktstatus) {
            break;
        }

        /* 5. 检查是否有可用数据 */
        if (!input_available_p(tty, 0)) {
            continue;
        }

        /* 6. 从规范模式或非规范模式缓冲区复制数据 */
        if (ldata->icanon && !L_EXTPROC(tty)) {
            if (canon_copy_from_read_buf(tty, &kb, &nr))
                goto more_to_be_read;
        } else {
            if (copy_from_read_buf(tty, &kb, &nr) && kb - kbuf >= minimum)
                goto more_to_be_read;
        }

        if (kb - kbuf >= minimum)
            break;
        if (time)
            timeout = time;
    }
}
```

### 6.2 n_tty_write（写函数）

**文件路径**: `linux/drivers/tty/n_tty.c`

```c
static ssize_t n_tty_write(struct tty_struct *tty, struct file *file,
               const u8 *buf, size_t nr)
{
    const u8 *b = buf;
    DEFINE_WAIT_FUNC(wait, woken_wake_function);
    ssize_t num, retval = 0;

    /* 1. 检查作业控制 */
    if (L_TOSTOP(tty) && file->f_op->write_iter != redirected_tty_write) {
        retval = tty_check_change(tty);
        if (retval)
            return retval;
    }

    down_read(&tty->termios_rwsem);

    /* 2. 处理待处理的回显字符 */
    process_echoes(tty);

    add_wait_queue(&tty->write_wait, &wait);
    while (1) {
        if (signal_pending(current)) {
            retval = -ERESTARTSYS;
            break;
        }
        if (tty_hung_up_p(file) || (tty->link && !tty->link->count)) {
            retval = -EIO;
            break;
        }

        /* 3. 如果启用 OPOST，进行输出处理 */
        if (O_OPOST(tty)) {
            while (nr > 0) {
                num = process_output_block(tty, b, nr);
                if (num < 0) {
                    if (num == -EAGAIN)
                        break;
                    retval = num;
                    goto break_out;
                }
                b += num;
                nr -= num;
                if (nr == 0)
                    break;
                if (process_output(*b, tty) < 0)
                    break;
                b++; nr--;
            }
            if (tty->ops->flush_chars)
                tty->ops->flush_chars(tty);
        } else {
            /* 直接调用驱动写 */
            while (nr > 0) {
                mutex_lock(&ldata->output_lock);
                num = tty->ops->write(tty, b, nr);
                mutex_unlock(&ldata->output_lock);
                if (num < 0) {
                    retval = num;
                    goto break_out;
                }
                if (!num)
                    break;
                b += num;
                nr -= num;
            }
        }
    }
}
```

---

## 七、uart_write 和 UART TX 中断处理

### 7.1 uart_write（UART 写函数）

**文件路径**: `linux/drivers/tty/serial/serial_core.c`

```c
static ssize_t uart_write(struct tty_struct *tty, const u8 *buf, size_t count)
{
    struct uart_state *state = tty->driver_data;
    struct uart_port *port;
    unsigned long flags;
    int ret = 0;

    if (WARN_ON(!state))
        return -EL3HLT;

    port = uart_port_lock(state, flags);
    if (!state->port.xmit_buf) {
        uart_port_unlock(port, flags);
        return 0;
    }

    /* 将数据放入传输 FIFO */
    if (port)
        ret = kfifo_in(&state->port.xmit_fifo, buf, count);

    /* 启动传输 */
    __uart_start(state);
    uart_port_unlock(port, flags);
    return ret;
}
```

### 7.2 __uart_start（启动传输）

**文件路径**: `linux/drivers/tty/serial/serial_core.c`

```c
static void __uart_start(struct uart_state *state)
{
    struct uart_port *port = state->uart_port;

    if (!pm_runtime_enabled(port->dev) || pm_runtime_active(port->dev))
        port->ops->start_tx(port);  /* 启动 TX 中断 */
}
```

### 7.3 UART TX 中断处理流程

```
硬件中断触发
    │
    ▼
serial8250_interrupt()  (8250驱动中断处理)
    │
    ▼
handle_rx_interrupt() / handle_tx_interrupt()
    │
    ├──► 接收：readb(port->uart_port.mcr) → 放入 tty_buffer
    │
    └──► 发送：port->ops->start_tx(port)
                    │
                    ▼
            从 xmit_fifo 取数据写入硬件 TX FIFO
                    │
                    ▼
            UART TX 中断 → TX FIFO 空 → 触发 uart_write_wakeup()
                    │
                    ▼
            tty_port_tty_wakeup() → 唤醒 write_wait 队列
```

**uart_write_wakeup 函数**:

```c
void uart_write_wakeup(struct uart_port *port)
{
    struct uart_state *state = port->state;

    BUG_ON(!state);
    tty_port_tty_wakeup(&state->port);  /* 唤醒等待写的进程 */
}
```

---

## 八、读缓冲区和写缓冲区的实现

### 8.1 TTY 读缓冲区（tty_buffer）

**文件路径**: `linux/drivers/tty/tty_buffer.c`

TTY 使用**页面链式缓冲区**：

```c
struct tty_buffer {
    struct tty_buffer *next;   /* 下一个缓冲区 */
    int used;                  /* 已使用的字节数 */
    int size;                  /* 缓冲区总大小 */
    int commit;                /* 已提交的数据量 */
    int read;                  /* 已读取的数据量 */
    int lookahead;             /* 预读位置 */
    bool flags;                /* 是否包含标志 */
    u8 data[0];                /* 数据区（柔性数组） */
};
```

**flush_to_ldisc**: 从硬件接收的数据通过此工作队列推送到行规程

```c
static void flush_to_ldisc(struct work_struct *work)
{
    struct tty_port *port = container_of(work, struct tty_port, buf.work);

    mutex_lock(&buf->lock);
    while (1) {
        struct tty_buffer *head = buf->head;
        size_t count = smp_load_acquire(&head->commit) - head->read;

        if (!count) {
            if (next == NULL)
                break;
            continue;
        }

        /* 调用行规程的 receive_buf */
        rcvd = receive_buf(port, head, count);
        head->read += rcvd;
    }
    mutex_unlock(&buf->lock);
}
```

### 8.2 N_TTY 行规程读缓冲区（n_tty_data）

**文件路径**: `linux/drivers/tty/n_tty.c`

```c
struct n_tty_data {
    /* 生产者发布 */
    size_t read_head;      /* 读缓冲区头 */
    size_t commit_head;    /* 已提交头 */
    size_t canon_head;     /* 规范模式头（换行符位置） */
    size_t echo_head;      /* 回显缓冲区头 */
    DECLARE_BITMAP(char_map, 256);  /* 特殊字符映射 */

    unsigned char lnext:1, erasing:1, raw:1, real_raw:1, icanon:1;
    unsigned char push:1;

    /* 生产者和消费者共享 */
    u8 read_buf[N_TTY_BUF_SIZE];   /* 读缓冲区 - 4KB 环形缓冲区 */
    DECLARE_BITMAP(read_flags, N_TTY_BUF_SIZE);
    u8 echo_buf[N_TTY_BUF_SIZE];    /* 回显缓冲区 */

    /* 消费者发布 */
    size_t read_tail;       /* 读缓冲区尾 */
    size_t line_start;      /* 行开始位置 */

    struct mutex atomic_read_lock;  /* 读操作原子锁 */
    struct mutex output_lock;        /* 输出锁 */
};
```

### 8.3 UART 传输缓冲区

```c
struct uart_port {
    DECLARE_KFIFO_PTR(xmit_fifo, u8);  /* 传输 FIFO - 内核 FIFO 实现 */
    // ...
};
```

---

## 九、关键数据结构关系图

```
用户进程
    │
    ▼
sys_ioctl()  ─────────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            │
do_vfs_ioctl()                                                  │
    │                                                            │
    ▼                                                            │
vfs_ioctl()  ──▶ filp->f_op->unlocked_ioctl                     │
    │                          │                                 │
    │                          ▼                                 │
    │               tty_ioctl() (tty_io.c)                      │
    │                          │                                 │
    │         ┌────────────────┼────────────────┐               │
    │         ▼                ▼                ▼               │
    │   tty_mode_ioctl   uart_ioctl     n_tty_ioctl_helper       │
    │   (TTY 层通用)    (UART 驱动层)    (行规程层)               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  tty_struct     │
                    │  ├─ driver      │──▶ tty_operations
                    │  ├─ port       │──▶ tty_port
                    │  │              │     ├─ xmit_fifo (UART TX FIFO)
                    │  │              │     └─ buf (tty_buffer 链)
                    │  └─ ldisc      │──▶ tty_ldisc
                    │                 │     ├─ ops (tty_ldisc_ops)
                    │                 │     └─ disc_data (n_tty_data)
                    │                                 │
                    │                                 ▼
                    │                    ┌────────────────────┐
                    │                    │   n_tty_data       │
                    │                    │   ├─ read_buf[4K]  │ ← 读缓冲区
                    │                    │   ├─ echo_buf[4K]  │ ← 回显缓冲区
                    │                    │   └─ char_map[256] │
                    │                    └────────────────────┘
                    └─────────────────────────│───────────────────
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   uart_port     │
                                    │   (硬件抽象)     │
                                    │   ├─ ops        │──▶ uart_ops
                                    │   │   start_tx  │    (硬件操作)
                                    │   │   stop_tx   │
                                    │   │   tx_empty  │
                                    │   └─ xmit_fifo │──▶ 硬件 TX FIFO
                                    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   UART 硬件     │
                                    │   (8250/16550)  │
                                    │   TX Register   │
                                    └─────────────────┘
```

---

## 十、关键文件路径总结

| 功能 | 文件路径 |
|------|----------|
| sys_ioctl 入口 | `linux/fs/ioctl.c` |
| VFS ioctl | `linux/fs/ioctl.c` |
| TTY ioctl 分发 | `linux/drivers/tty/tty_ioctl.c` |
| TTY 结构定义 | `linux/include/linux/tty.h` |
| TTY 行规程 | `linux/include/linux/tty_ldisc.h` |
| TTY 端口 | `linux/include/linux/tty_port.h` |
| N_TTY 实现 | `linux/drivers/tty/n_tty.c` |
| N_TTY 缓冲区 | `linux/drivers/tty/tty_buffer.c` |
| 串口核心 | `linux/drivers/tty/serial/serial_core.c` |
| 串口头文件 | `linux/include/linux/serial_core.h` |
| ioctl 宏定义 | `linux/include/uapi/asm-generic/ioctl.h` |
| TTY 驱动定义 | `linux/include/linux/tty_driver.h` |

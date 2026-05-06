# wait 阶段用处与保留必要性研究

## 问题背景

在 `record_loop` 的每一帧末尾，程序会先计算本帧已经消耗的时间，然后调用：

```python
dt_s = time.perf_counter() - start_loop_t
busy_wait(1 / fps - dt_s)
```

这段代码的含义不是额外增加一个业务阶段，而是把当前帧补齐到目标帧间隔。以 `fps=30`
为例，单帧周期应接近 33.3 ms；如果观测、推理、动作下发和数据写入总共只用了 20 ms，
剩下的约 13.3 ms 就会进入 wait。如果前面的实际耗时已经超过 33.3 ms，
传给 `busy_wait()` 的时间为负数或接近 0，wait 基本不会再等待。

因此，wait 更像是控制循环的节拍器。它不参与模型推理，也不直接生成数据，但会影响每一帧之间的时间距离。

## wait 是否会进入内核

在树莓派 5 的 Linux 环境下，`record_loop` 调用的是 LeRobot 封装的 `busy_wait()`，
但这个函数在 Linux 上并不是真的一直空转。它的实现逻辑是按操作系统分支处理：

```python
def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        if seconds > 0:
            time.sleep(seconds)
```

也就是说，在 Mac 和 Windows 上，它为了更精确地控制时间，会用 Python 循环持续检查 `perf_counter()`，
这种方式主要消耗用户态 CPU，不主动把线程睡眠交给内核。树莓派运行的是 Linux，所以会走 `time.sleep(seconds)`。

`time.sleep()` 在 Linux 上最终会进入内核的定时睡眠路径，通常对应 `clock_nanosleep` 或 `nanosleep` 一类系统调用。
线程进入睡眠后会从可运行队列中让出 CPU，内核调度其他线程或进程运行；等定时器到期后，内核再把这个 Python 线程唤醒，
后续由调度器安排它重新获得 CPU。

因此，Linux 上的 wait 是会进内核的，也会带来上下文切换。大致过程可以理解为：

1. Python 主线程调用 `time.sleep()`，从用户态进入内核；
2. 内核把该线程标记为睡眠状态，并设置定时器；
3. CPU 被让给其他可运行任务，例如相机后台线程、系统服务或空闲线程；
4. 定时器到期后内核唤醒该线程；
5. 调度器在合适时机把 CPU 切回 `record_loop` 主线程。

这里确实有系统调用和上下文切换成本，但它通常不是这段 wait 的主要问题。对于 30 fps 控制循环，
单帧周期是 33.3 ms，wait 往往是毫秒级；一次睡眠/唤醒的调度开销通常远小于这个量级。
相比之下，如果不用睡眠而在用户态空转，虽然可以少一次主动睡眠唤醒，但会持续占用一个 CPU 核，
反而可能挤压 PyTorch 推理、OpenCV 读线程和串口 I/O 的调度空间。

所以在树莓派上，wait 的内核路径不是纯粹损失。它一方面带来系统调用和调度开销，
另一方面也主动释放 CPU，让后台读线程和系统 I/O 有机会运行。对于这个项目，真正需要关注的是：
睡眠唤醒后是否导致控制周期抖动，以及释放 CPU 后是否改善相机读线程和系统整体负载。

## `time.sleep()` 到内核的源码调用栈

结合本项目中的 CPython、glibc 和 Linux 源码，`record_loop` 的 wait 在树莓派 Linux 上可以画成下面这条静态调用链：

```text
record_loop()
  └─ busy_wait(1 / fps - dt_s)
      └─ time.sleep(seconds)                         # Python 标准库接口
          └─ time_sleep()                            # cpython/Modules/timemodule.c
              └─ pysleep()
                  ├─ Py_BEGIN_ALLOW_THREADS          # 释放 GIL，睡眠期间不占着 Python 解释器锁
                  ├─ clock_nanosleep(
                  │     CLOCK_MONOTONIC,
                  │     TIMER_ABSTIME,
                  │     &timeout_abs,
                  │     NULL)
                  └─ Py_END_ALLOW_THREADS
                      ↓
                  glibc clock_nanosleep()
                      └─ __clock_nanosleep_time64()
                          └─ INTERNAL_SYSCALL_CANCEL(clock_nanosleep_time64 / clock_nanosleep, ...)
                              ↓
                          Linux syscall: clock_nanosleep
                              └─ SYSCALL_DEFINE4(clock_nanosleep, ...)
                                  └─ kc->nsleep(...)
                                      └─ hrtimer_nanosleep(...)
                                          └─ do_nanosleep(...)
                                              ├─ set_current_state(TASK_INTERRUPTIBLE | TASK_FREEZABLE)
                                              ├─ hrtimer_sleeper_start_expires(...)
                                              ├─ schedule()
                                              └─ __set_current_state(TASK_RUNNING)
```

对应源码位置如下：

- `lerobot/src/lerobot/utils/robot_utils.py`：Linux 分支中 `busy_wait()` 调用 `time.sleep(seconds)`；
- `cpython/Modules/timemodule.c`：`time_sleep()` 解析 Python 参数后调用 `pysleep()`；
- `cpython/Modules/timemodule.c`：`pysleep()` 在 `HAVE_CLOCK_NANOSLEEP` 分支调用 `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, ...)`；
- `glibc-2.42/sysdeps/unix/sysv/linux/clock_nanosleep.c`：`__clock_nanosleep_time64()` 使用 `INTERNAL_SYSCALL_CANCEL` 进入 Linux syscall；
- `glibc-2.42/sysdeps/unix/sysv/linux/aarch64/arch-syscall.h`：AArch64 上 `__NR_clock_nanosleep` 的 syscall 号为 115；
- `linux/kernel/time/posix-timers.c`：`SYSCALL_DEFINE4(clock_nanosleep, ...)` 是内核系统调用入口；
- `linux/kernel/time/hrtimer.c`：`hrtimer_nanosleep()` / `do_nanosleep()` 设置高精度定时器并调用 `schedule()` 让出 CPU。

这里有两个细节值得注意。第一，CPython 在睡眠前后用了 `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`，
说明 `time.sleep()` 期间会释放 GIL；这让相机后台线程、图像处理线程或其他 Python 线程有机会运行。
第二，CPython 在有 `clock_nanosleep` 的平台上使用的是 `CLOCK_MONOTONIC + TIMER_ABSTIME`，
也就是按单调时钟的绝对截止时间睡眠，而不是简单地每次相对睡一段时间。这样做可以减少被信号打断后反复重睡带来的累计漂移。

如果平台没有 `clock_nanosleep`，CPython 才会退到 `nanosleep()`，再不行才用 `select()`。
对树莓派 Linux 来说，重点链路应按 `clock_nanosleep` 理解，而不是 `wait4`，也不是普通的进程等待。

## sleep、线程等待和硬件阻塞的区别

wait 进内核，后面 `async_read()` 等相机线程、`sync_read()` 等串口硬件，也可能进内核，
但它们等待的原因不同，优化含义也不同。

`busy_wait()` 在 Linux 上走 `time.sleep()`，这是主动睡眠。主线程已经完成本帧工作，
只是为了对齐 `1/fps` 的控制周期，把剩余时间交给内核定时器。它等的是“时间到期”，
不是等某个资源释放。睡眠期间主线程不会占 CPU，内核可以调度相机后台线程、串口驱动处理、PyTorch 相关线程或其他系统任务。

`async_read()` 里的等待主要不是等互斥锁，而是等 `new_frame_event`。后台读线程持续调用 `read()` 采集图像，
采到新帧后会在 `frame_lock` 保护下更新 `latest_frame`，然后调用 `new_frame_event.set()` 通知主线程。
主线程在 `async_read()` 里执行：

```python
self.new_frame_event.wait(timeout=timeout_ms / 1000.0)

with self.frame_lock:
    frame = self.latest_frame
    self.new_frame_event.clear()
```

这里的 `Event.wait()` 如果事件还没到，会让线程阻塞；在 Linux 上，这类 Python 线程同步原语底层通常会走
pthread/futex 之类的内核等待机制，因此也可能发生上下文切换。它和 `time.sleep()` 的区别是：
`time.sleep()` 等固定时长，`Event.wait()` 等另一个线程发出“新帧到了”的通知。如果相机线程很快 set event，
主线程就很快醒；如果相机没有新帧，主线程最多等到 timeout。

`frame_lock` 的等待通常更短。后台线程只在写 `latest_frame` 的一小段时间持有锁，
主线程也只在复制引用、清除 event 时短暂持有锁。锁没有竞争时，获取锁可能只走用户态快速路径；
只有锁正好被另一个线程持有时，才可能进入内核等待。也就是说，`async_read()` 中真正可能产生毫秒级等待的，
一般是 `new_frame_event.wait()` 等新图像，而不是 `frame_lock` 本身。

舵机 `sync_read()` 又是另一类等待。它要通过串口总线发包并等待舵机回包，底层会涉及串口设备文件读写和驱动等待。
这里等的不是 Python 线程通知，也不是固定时间到期，而是硬件 I/O 完成。如果总线忙、舵机响应慢或出现重试，
主线程会被硬件响应时间拖住。这个等待即使删掉 record_loop 末尾的 wait 也不会消失。

可以把三类等待区分为：

| 等待位置 | 等待对象 | 是否可能进内核 | 主要含义 |
|---|---|---|---|
| `time.sleep()` | 定时器到期 | 会 | 主动让出 CPU，维持控制节拍 |
| `Event.wait()` | 相机后台线程通知新帧 | 会 | 等生产者线程产生数据 |
| `frame_lock` | `latest_frame` 临界区释放 | 竞争时会 | 保护共享帧，通常很短 |
| `sync_read()` | 串口/舵机回包 | 会 | 等硬件 I/O 完成 |

所以，虽然它们都可能表现为“线程不在跑”并发生上下文切换，但语义不同。
record_loop 末尾的 wait 是可控的节拍等待；`async_read()` 的等待说明主线程跑到了相机生产速度前面；
`sync_read()` 的等待说明控制循环受硬件 I/O 限制。删除 wait 以后，如果主线程更早进入下一帧，
很可能只是把原本主动的 sleep 时间，变成 `Event.wait()` 或串口读写中的被动等待。

从损失大小看，通常主动 wait 的损失更小，也更容易分析。`time.sleep()` 的主要代价是一次系统调用、
一次睡眠和一次唤醒调度；它发生在本帧工作已经完成之后，等待时长由 `1/fps - dt_s` 明确决定。
相机和舵机等待虽然也可能让线程睡眠，但它们的结束条件取决于外部资源：相机后台线程什么时候产出新帧、
串口总线什么时候收到舵机回包。这类等待不只包含调度成本，还包含设备响应时间、线程竞争和 I/O 抖动。

因此，更准确的判断是：wait 是可控的节拍等待，`async_read()` / `sync_read()` 是资源或硬件没准备好造成的被动等待。
前者通常更干净，后者更容易把抖动压进观测阶段。删除 wait 不一定减少总损失，很多时候只是把可控等待换成不可控等待。

## wait 留着有什么用

wait 的第一层作用是稳定控制频率。机器人控制不是单纯追求“循环越快越好”，而是希望动作以相对稳定的时间间隔下发。
训练数据是按固定 fps 采集的，策略在训练时看到的是一帧一帧等间隔的观测和动作。如果在线运行时忽快忽慢，
同样的动作序列在真实时间里的执行速度就会变化，机械臂运动会更难预测。

第二层作用是稳定数据集时间戳。`LeRobotDataset.add_frame()` 在没有显式传入 timestamp 时，
会使用：

```python
timestamp = frame_index / self.fps
```

也就是说，数据集文件里的时间戳默认不是按真实墙钟时间写入的，而是按帧号和数据集 fps 推出来的。
如果主循环实际运行节奏和 `dataset.fps` 差得很远，数据文件仍然会表现为“理想的 30 fps”，
但真实采集过程并不是这个节奏。训练时模型看到的时间尺度就会和真实硬件执行过程出现偏差。

第三层作用是避免主循环无节制地抢占 CPU 和硬件 I/O。`record_loop` 是一个串行 Python 循环，
一帧结束后如果马上进入下一帧，就会立刻再次调用 `get_observation()`、`select_action()` 和 `send_action()`。
在树莓派 5 这种纯 ARM CPU 推理环境里，推理、图像处理、串口读写和相机后台线程都在争 CPU。
保留 wait 可以给系统一个明确的控制节奏，避免主线程把所有空闲时间都压到下一轮硬件读取或模型推理上。

## 去掉 wait 会不会只是把等待挪到下一次观测

这个质疑是成立的，而且很关键。wait 不一定是“可以直接删掉的浪费时间”，因为删掉之后，系统未必真的按同等幅度变快。

SO101 的 `get_observation()` 里包含两类读取。舵机位置通过 `sync_read("Present_Position")` 读取，这是阻塞式串口 I/O；
相机图像通过 `cam.async_read()` 获取，它不直接读摄像头硬件，而是从后台读线程维护的 `latest_frame` 中取最新帧。
OpenCV 相机的 `async_read()` 内部会等待 `new_frame_event`，随后在 `frame_lock` 保护下读取 `latest_frame` 并清除事件。

如果删掉 wait，下一帧会更早进入 `get_observation()`。这时可能出现几种情况：

1. 相机后台线程还没采到新帧，`async_read()` 会在观测阶段等待 `new_frame_event`。
2. 串口总线还没准备好响应下一次舵机读取，`sync_read()` 仍然会阻塞。
3. 主线程更频繁地进入推理和下发动作，CPU 占用上升，反过来可能影响相机后台线程和系统调度。

也就是说，删除 wait 后，原本计入 `wait` 的一部分时间可能会转移到下一帧的 `obs` 中，
尤其是转移到 `async_read()` 等新图像、舵机 `sync_read()` 等硬件响应这些地方。
从统计表面看，`wait_pct` 会下降，但 `obs_pct` 或总循环抖动可能上升，任务效果未必改善。

如果某次实验里 wait 占比很高，它首先说明当前循环的实际工作时间短于目标帧周期，系统在主动等 30 fps 的节拍。
它不自动等价于性能瓶颈。真正需要判断的是：去掉或缩短 wait 后，总 fps 是否稳定提升，观测阻塞是否增加，
CPU 占用是否明显上升，动作是否仍然平滑，任务成功率是否保持。

## fps 会不会存到模型里

严格说，`fps` 不作为一个普通输入张量进入 ACT 模型，也不是模型前向计算里的显式参数。
但是它会进入数据集元信息，并通过数据集和训练配置间接影响模型学到的时间尺度。

在 LeRobot 数据集中，`fps` 存在 `dataset.meta.info["fps"]`。它至少参与以下几件事：

- `LeRobotDataset.add_frame()` 默认用 `frame_index / self.fps` 生成每帧 timestamp；
- 视频帧编码和读取时会使用数据集 fps；
- 数据加载时会根据 `dataset.meta.fps` 把策略配置里的 delta indices 换算成 delta timestamps；
- `record_loop` 会检查 `dataset.fps != fps`，不一致时直接报错；
- `sanity_check_dataset_robot_compatibility()` 也会把 fps 作为数据集和当前采集配置是否兼容的一部分。

以 ACT 为例，`ACTConfig.action_delta_indices` 返回 `list(range(chunk_size))`。
这些 index 本身是“第 0 帧、第 1 帧、……第 chunk_size-1 帧”的离散步数。
训练数据加载时，LeRobot 会用数据集 fps 把这些离散步数解释成真实时间上的间隔。
所以模型权重文件里不一定有一个叫 fps 的数，但模型是在某个 fps 对应的动作时间尺度上训练出来的。

## 推理时换一个 fps 是否可行

从接口上看，推理时换一个 fps 并不一定马上报错，特别是不往同一个数据集追加数据、只做在线控制时。
但是从控制含义上看，它会改变动作序列对应的真实时间。

假设训练和采集都是 30 fps，那么相邻两帧动作间隔约为 33.3 ms。
如果在线运行时改成 20 fps，相邻两次动作下发变成约 50 ms。
模型输出的动作数值本身没有变，但动作执行节奏变慢了，原本 1 秒内执行 30 个控制步，现在只执行 20 个控制步。
这会让轨迹在真实时间上被拉长，可能表现为动作滞后、响应慢、接触时机不准，严重时会影响抓取或推倒这类任务的成功率。

反过来，如果运行 fps 高于训练 fps，也不一定更好。动作下发更频繁，可能让相邻动作变化过密，
也可能因为相机和舵机读取跟不上而产生重复图像、观测延迟或更高的 CPU 占用。
对于闭环控制来说，fps 改变的是系统的时间尺度，而不只是一个显示参数。

因此，“采集 30 fps、运行时用另一个 fps”可以作为实验变量，但不应默认认为没有影响。
小幅偏差可能还能工作，较大偏差需要通过任务成功率、动作平滑性和观测延迟来验证。

## 保留必要性判断

在当前代码结构下，wait 默认应该保留。它的价值不在于让单帧计算更快，而在于把控制循环固定在数据集 fps 对应的时间节拍上。
对于离线模仿学习，训练数据的时间间隔、动作序列的执行节奏和在线推理的控制频率应该尽量一致。

如果目标是研究性能上限，可以临时去掉 wait 或把 fps 调高，但这个实验回答的是“系统最快能跑多快”，
不是“实际部署时应该删除 wait”。如果目标是提升任务效果，更稳妥的方向通常是降低推理耗时、减少图像分辨率、
减少相机路数、优化模型，或者把控制循环改成明确的自适应节拍，而不是简单删除等待。

一个更准确的说法是：wait 不是推理瓶颈，但它是时间语义的一部分。只看 `wait_pct` 容易误判；
需要结合真实 fps、各阶段耗时转移、CPU 占用和任务结果一起判断。

## 建议实验

可以做一个 A/B 对比来验证 wait 是否只是转移到了观测阶段。

实验 A 保留当前逻辑：

```python
busy_wait(1 / fps - dt_s)
```

实验 B 临时跳过等待：

```python
# busy_wait(1 / fps - dt_s)
```

两组实验都记录以下指标：

- `obs`、`inference`、`action`、`wait`、`total`、实际 fps；
- `async_read()` 的实际等待时间，尤其是是否接近相机帧间隔；
- 舵机 `sync_read("Present_Position")` 的耗时；
- CPU 占用和系统负载；
- 任务成功率、动作是否抖动、接触时机是否变差。

如果实验 B 中 `wait` 下降，但 `obs` 明显上升，说明等待主要被挪到了下一次观测；
如果实验 B 的 CPU 占用升高、动作更不稳定，即使平均 fps 变高，也不适合作为默认部署方式。
只有在总延迟下降、观测阻塞没有明显增加、任务效果不变或更好时，才值得进一步考虑修改控制节拍。

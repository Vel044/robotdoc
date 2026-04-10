# PyTorch CPU 推理线程池管理详解（Native 后端）

> ⚠️ **注意**：此文档描述的是 **Native 后端**（c10::ThreadPool），你的环境（lerobot）实际使用的是 **OpenMP 后端**。
>
> 你的环境对应的文档：[PyTorch_OpenMP_调用链详解.md](PyTorch_OpenMP_调用链详解.md)
>
> ---
>
> 基于 ARM64 树莓派5 CPU + PyTorch Native 后端，分析从矩阵运算到 glibc futex 的完整链路
>
> 源码路径：`/Users/vel/Work/RobotOS/Lerobot/`

---

## 目录

- [1. 整体架构](#1-整体架构)
- [2. 并行后端选择：Native vs OpenMP](#2-并行后端选择native-vs-openmp)
- [3. parallel_for 核心流程](#3-parallel_for-核心流程)
- [4. c10::ThreadPool 线程池实现](#4-c10threadpool-线程池实现)
- [5. std::mutex → glibc futex → Linux 内核 完整调用链](#5-stdmutex--glibc-futex--linux-内核-完整调用链)
- [6. futex 快路径与慢路径详解](#6-futex-快路径与慢路径详解)
- [7. 矩阵运算中的线程分配](#7-矩阵运算中的线程分配)
- [8. 线程数配置与调优](#8-线程数配置与调优)
- [附录：源码文件索引](#附录源码文件索引)

---

## 1. 整体架构

```
Python 用户代码: output = model(input)
        │
        ▼
torch.nn.Linear.forward()
        │
        ▼
at::addmm_out()  ──→  native/Blas.cpp
        │
        ▼
cpublas_gemm()  ──→  native/cpu/BlasKernel.cpp
        │
        ▼
gemm_core_() → gemm_transa_() → parallel_for(0, m, 1, lambda)
        │
        ├─ AT_PARALLEL_OPENMP=1  ──→ OpenMP 后端（libgomp）──→ gcc-13.2.0/libgomp/
        │
        └─ AT_PARALLEL_NATIVE=1  ──→ Native 后端 ──────────────→ c10::ThreadPool
                                   【树莓派5默认走这条线】
                                          │
                                          ▼
                            ┌────────────────────────────┐
                            │  4 个工作线程 (pthread)      │
                            │                              │
                            │  工作线程等待任务：            │
                            │    std::mutex::lock()         │
                            │      ↓                        │
                            │    pthread_mutex_lock()        │
                            │      ↓                        │
                            │    lll_lock() / futex_wait   │
                            │      ↓                        │
                            │    Linux sys_futex()          │
                            │      ↓                        │
                            │    schedule() 挂起线程        │
                            │    【零 CPU 占用】            │
                            └────────────────────────────┘
```

---

## 2. 并行后端选择：Native vs OpenMP

**源码位置**：`aten/src/ATen/Parallel.h:88-92`

```cpp
// 编译时按 AT_PARALLEL_* 宏选择后端
#if AT_PARALLEL_OPENMP
    #include <ATen/ParallelOpenMP.h>   // OpenMP 后端（libgomp）
#elif AT_PARALLEL_NATIVE
    #include <ATen/ParallelNative.h>   // Native 后端（c10::ThreadPool）
#endif
```

**两种后端对比**：

| 特性 | OpenMP 后端 | **Native 后端（树莓派5）** |
|------|-------------|--------------------------|
| 实现方式 | `#pragma omp parallel for` | `c10::ThreadPool`（自实现） |
| 线程库 | libgomp（GCC）或 libiomp（Intel） | **libstdc++（GCC）+ glibc NPTL** |
| 任务队列 | OpenMP 运行时管理 | `std::queue<std::function>` |
| 同步原语 | libgomp 内部管理 | **std::mutex + std::condition_variable** |
| 适用场景 | 有 OpenMP 支持 | **嵌入式/无 OpenMP（树莓派5默认）** |

**树莓派5上的实际情况**：
- PyTorch 官方 ARM build **不启用 OpenMP**（AT_PARALLEL_OPENMP=0）
- 启用 `AT_PARALLEL_NATIVE=1`，走 **c10::ThreadPool**
- 所以同步机制走的是 **`std::mutex` → `libstdc++` → `glibc NPTL` → `futex`**，不是 libgomp

---

## 3. parallel_for 核心流程

**源码位置**：`aten/src/ATen/Parallel-inl.h` + `aten/src/ATen/ParallelNative.cpp`

### 3.1 parallel_for 模板

```cpp
template <class F>
inline void parallel_for(
    const int64_t begin,       // 起始索引
    const int64_t end,         // 结束索引
    const int64_t grain_size,  // 最小任务粒度（小于则串行）
    const F& f) {              // 任务函数 f(begin, end)

    if (begin >= end) return;

    // 判断是否并行：迭代数够大 + 不在并行区域内 + 线程数>1
    const bool use_parallel =
        (end - begin > grain_size &&
         end - begin > 1 &&
         !at::in_parallel_region() &&
         at::get_num_threads() > 1);

    if (!use_parallel) {
        // 串行执行
        f(begin, end);
        return;
    }

    // 并行执行：分发到线程池
    internal::invoke_parallel(begin, end, grain_size, f);
}
```

### 3.2 invoke_parallel 实现

**源码位置**：`aten/src/ATen/ParallelNative.cpp:130-195`

```cpp
void invoke_parallel(..., const std::function<void(int64_t, int64_t)>& f) {
    // Step 1: 计算任务数和块大小
    size_t num_tasks, chunk_size;
    std::tie(num_tasks, chunk_size) =
        internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

    // Step 2: 构造同步状态（用于等待所有任务完成）
    struct {
        std::atomic_flag err_flag;
        std::exception_ptr eptr;
        std::mutex mutex;              // ★ 关键：libstdc++ std::mutex
        std::atomic_size_t remaining;
        std::condition_variable cv;    // ★ 关键：libstdc++ condition_variable
    } state;
    state.remaining = num_tasks;

    // Step 3: 定义任务函数
    auto task = [&](size_t task_id) {
        int64_t local_start = begin + task_id * chunk_size;
        int64_t local_end = std::min(end, local_start + chunk_size);
        try {
            f(local_start, local_end);  // 执行矩阵块计算
        } catch (...) {
            state.eptr = std::current_exception();
        }
        // 通知主线程任务完成
        {
            std::lock_guard<std::mutex> lk(state.mutex);
            if (--state.remaining == 0)
                state.cv.notify_one();
        }
    };

    // Step 4: 分发到线程池（task_id=0 在主线程执行）
    _run_with_pool(task, num_tasks);

    // Step 5: 阻塞等待所有任务完成
    {
        std::unique_lock<std::mutex> lk(state.mutex);
        if (state.remaining != 0)
            state.cv.wait(lk);  // ★ 等待时通过 condition_variable 阻塞
    }
}
```

---

## 4. c10::ThreadPool 线程池实现

### 4.1 ThreadPool 类定义

**源码位置**：`c10/core/thread_pool.h:28-80`

```cpp
class C10_API ThreadPool : public c10::TaskThreadPoolBase {
    // 任务队列
    std::queue<task_element_t> tasks_;      // 任务队列（FIFO）

    // 同步原语 ──────────────────────────────────────────
    std::mutex mutex_;                       // ★ libstdc++ std::mutex
    std::condition_variable condition_;      // ★ libstdc++ condition_variable
    std::condition_variable completed_;      // ★ libstdc++ condition_variable
    // ─────────────────────────────────────────────────

    std::vector<std::thread> threads_;       // 工作线程数组
    std::atomic_bool running_;               // 线程池运行标志
    std::size_t available_;                  // 可用线程数
    std::size_t total_;                      // 总线程数

    void main_loop(std::size_t index);      // 工作线程主循环
};
```

### 4.2 工作线程主循环（核心阻塞点）

**源码位置**：`c10/core/thread_pool.cpp:85-130`

```cpp
void ThreadPool::main_loop(std::size_t index) {
    std::unique_lock<std::mutex> lock(mutex_);  // ★ 加锁

    while (running_) {
        // ★★★ 关键阻塞点 ★★★
        // condition_.wait() 让线程挂起，直到有新任务
        // 底层：libstdc++ → glibc futex → Linux 线程调度
        condition_.wait(lock, [&]() {
            return !tasks_.empty() || !running_;
        });

        if (!running_) break;

        // 取出任务
        task_element_t task = std::move(tasks_.front());
        tasks_.pop();
        --available_;

        lock.unlock();  // 释放锁，允许其他线程继续添加任务

        // 执行任务
        try {
            if (task.run_with_id)
                task.with_id(index);
            else
                task.no_id();
        } catch (...) { ... }

        lock.lock();   // 重新加锁
        ++available_;

        // 全部完成，通知等待线程
        if (tasks_.empty() && available_ == total_)
            completed_.notify_one();
    }
}
```

### 4.3 任务提交

**源码位置**：`c10/core/thread_pool.cpp:75-83`

```cpp
void ThreadPool::run(std::function<void()> func) {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));   // 入队
    condition_.notify_one();           // 唤醒一个等待中的工作线程
}
```

---

## 5. std::mutex → glibc futex → Linux 内核 完整调用链

> **这是树莓派5上 PyTorch Native 后端使用的完整同步链路。**
> 涉及：libstdc++（gcc-13.2.0）→ glibc-2.42 NPTL → Linux kernel

### 5.1 第一层：C++ 标准库 libstdc++

**源码**：`gcc-13.2.0/libstdc++-v3/include/bits/std_mutex.h:114-125`

```cpp
class mutex : private __mutex_base {
    void lock() {
        // 直接调用 __gthread_mutex_lock，传入原生互斥锁指针
        int __e = __gthread_mutex_lock(&_M_mutex);
        if (__e) __throw_system_error(__e);
    }
    // _M_mutex 类型是 __gthread_mutex_t，在 Linux 上 = pthread_mutex_t
};
```

### 5.2 第二层：GThread 抽象层（gcc）

**源码**：`gcc-13.2.0/libgcc/gthr-posix.h:746-752`

```cpp
static inline int
__gthread_mutex_lock(__gthread_mutex_t *__mutex)
{
    if (__gthread_active_p())
        return __gthrw_(pthread_mutex_lock)(__mutex);  // 直接调用 glibc
    else
        return 0;
}
```

### 5.3 第三层：glibc NPTL - pthread_mutex_lock

**源码**：`glibc-2.42/nptl/pthread_mutex_lock.c:75-95`

```cpp
int __pthread_mutex_lock(pthread_mutex_t *mutex)
{
    unsigned int type = PTHREAD_MUTEX_TYPE_ELISION(mutex);

    // 最常见路径：PTHREAD_MUTEX_TIMED_NP（普通定时互斥锁）
    if (__glibc_likely(type == PTHREAD_MUTEX_TIMED_NP)) {
        FORCE_ELISION(mutex, goto elision);
    simple:
        // ★ 走 LLL_MUTEX_LOCK_OPTIMIZED = 低层锁（CAS + futex）
        LLL_MUTEX_LOCK_OPTIMIZED(mutex);
    }
    // ...
}
```

### 5.4 第四层：glibc LLL（Low-Level Lock）

**源码**：`glibc-2.42/sysdeps/nptl/lowlevellock.h:94-108`

```cpp
// lll_lock = 低层锁（Low-Level Lock），是 futex 的上一层封装
#define __lll_lock(futex, private)                          \
  ({                                                        \
    int *__futex = (futex);                                 \
    // 第1步（快路径）：CAS 原子地把 futex 从 0 改成 1
    if (__glibc_unlikely(                                   \
          atomic_compare_and_exchange_bool_acq(__futex, 1, 0))) \
      // 第2步（慢路径）：CAS 失败，进入等待
      __lll_lock_wait_private(__futex);                     \
    else                                                    \
      __lll_lock_wait(__futex, private);                    \
  })
```

### 5.5 第五层：glibc futex_wait

**源码**：`glibc-2.42/nptl/lowlevellock.c:39-51`

```cpp
void __lll_lock_wait(int *futex, int private)
{
  // 如果值已经是 2（已有等待者），直接进 futex
  if (atomic_load_relaxed(futex) == 2)
    goto futex;

  // 否则，尝试原子交换把 futex 设为 2
  while (atomic_exchange_acquire(futex, 2) != 0) {
  futex:
    LIBC_PROBE(lll_lock_wait, 1, futex);
    // ★★★ 终于进入 futex_wait：发起系统调用让线程挂起 ★★★
    futex_wait((unsigned int *)futex, 2, private);
  }
}
```

### 5.6 第六层：glibc → Linux 系统调用

**源码**：`glibc-2.42/sysdeps/nptl/lowlevellock-futex.h:56-78`

```cpp
// INTERNAL_SYSCALL：将参数路由到 ARM64 的 svc #0（系统调用指令）
// ARM64 系统调用号：__NR_futex = __NR_futex（在 arch/arm64/include/uapi/asm/unistd.h 定义）
//
// glibc-2.42/sysdeps/unix/sysv/linux/arm/bits/syscall.h 中的定义：
// #define INTERNAL_SYSCALL(futex, nr, args...) \
//     __internal_syscall_error_t __err;        \
//     __asm__ __volatile__ ("svc #0"           \
//         : "=r"(__err) : "i"(__NR_futex), ## args)
//
// 最终在 glibc-2.42/sysdeps/unix/sysv/linux/arm/syscall.h 中：
// #define syscall(...)  syscall(__NR_futex, ...)
```

### 5.7 第七层：Linux Kernel sys_futex

**源码**：`linux/kernel/futex/syscalls.c`（ARM64 arch: `linux/arch/arm64/kernel/sys.c`）

```cpp
// 内核处理流程（ARM64）
//
// 1. 验证用户空间 futex_word 地址合法
//
// 2. 比较 *futex_word 与 expected 值
//    ├─ 相等 → 把线程加入 futex_word 的等待队列
//    │         → 设置状态为 TASK_INTERRUPTIBLE
//    │         → 调用 schedule() 让出 CPU
//    │         → 【线程彻底挂起，零 CPU 占用】
//    └─ 不等 → 立即返回 -EAGAIN（不阻塞）
//
// 3. 被 wake 时：
//    → 从等待队列移出线程
//    → 设置状态为 TASK_RUNNING
//    → 返回用户空间，线程继续执行
```

### 5.8 完整链路图

```
c10::ThreadPool::main_loop()
    condition_.wait(lock, predicate)
         │
         ├─ [libstdc++: gcc-13.2.0/libstdc++-v3/include/bits/std_mutex.h]
         │     std::mutex::lock() → __gthread_mutex_lock()
         │
         ├─ [GThread: gcc-13.2.0/libgcc/gthr-posix.h:749]
         │     pthread_mutex_lock()  ─────────────────────┐
         │                                                   │
         ├─ [glibc NPTL: glibc-2.42/nptl/pthread_mutex_lock.c:93]
         │     LLL_MUTEX_LOCK_OPTIMIZED(mutex) → lll_lock(mutex->__data.__lock, private)
         │
         ├─ [glibc LLL: glibc-2.42/sysdeps/nptl/lowlevellock.h:94]
         │     __lll_lock(&futex, private)
         │     ├─ 快路径: atomic_compare_and_exchange(0→1) → 立即返回 ★零syscall★
         │     └─ 慢路径: atomic_exchange(0→2) → __lll_lock_wait()
         │
         ├─ [glibc futex: glibc-2.42/nptl/lowlevellock.c:45]
         │     futex_wait(futex, 2, private)
         │
         ├─ [glibc syscall: glibc-2.42/sysdeps/nptl/lowlevellock-futex.h:76]
         │     INTERNAL_SYSCALL(futex, 4, futexp, FUTEX_WAIT, val, NULL)
         │
         ├─ [Linux Kernel: linux/kernel/futex/syscalls.c]
         │     sys_futex(futex_word, FUTEX_WAIT, 2, NULL)
         │     ├─ *futex_word == 2 ? → 加入等待队列 → schedule()【挂起，零CPU】
         │     └─ *futex_word != 2 ? → 返回 -EAGAIN（立即返回）
         │
         └─ 被唤醒时：
                futex_wake(futex, 1) → sys_futex(WAKE) → 线程移回就绪队列
```

---

## 6. futex 快路径与慢路径详解

### 6.1 futex 的 3 个状态值

| 值（mutex->__data.__lock） | 含义 | 触发条件 |
|---------------------------|------|---------|
| `0` | 未锁定（Unlocked） | 锁空闲 |
| `1` | 已被我锁定，无等待者（Locked, no waiters） | 首次 CAS(0→1) 成功 |
| `2` | 已被他人锁定，有等待者（Locked, waiters） | 有线程在 futex 队列 |

### 6.2 快路径（零系统调用）

发生在锁竞争**不激烈**时，完全在用户空间：

```
Thread A: lll_lock() → CAS(0→1) 成功 → 立即返回
                              ↕
Thread B: lll_lock() → CAS(0→1) 失败
                              → atomic_exchange(0→2) 成功 → 立即返回 ★零syscall★
```

Thread B 只用了 1 次 CAS + 1 次原子交换，**没有触发任何系统调用**。

### 6.3 慢路径（内核介入）

锁被长期持有时，线程调用 `futex(FUTEX_WAIT)` 进入内核：

```
Thread B: futex_wait(addr, 2, private)
         │
         ├─ [glibc] INTERNAL_SYSCALL(futex, ...)  --→ ARM64: svc #0
         │
         └─ [Linux Kernel: sys_futex()]
                ├─ 读取 *addr，比较 == 2 ?
                │     ├─ 是 → 加入等待队列
                │     │       → set_current_state(TASK_INTERRUPTIBLE)
                │     │       → schedule() 【彻底挂起，零 CPU 占用】
                │     │
                │     └─ 否 → 返回 -EAGAIN（不等了，用户空间重试）
                │
                └─ 被 wake 时：
                        → remove_wait_queue()
                        → set_current_state(TASK_RUNNING)
                        → 返回用户空间
```

### 6.4 解锁：futex_wake

**源码**：`glibc-2.42/sysdeps/nptl/lowlevellock.h:145-159`

```cpp
#define __lll_unlock(futex, private)                \
  {                                                  \
    int *__futex = (futex);                         \
    // 原子地把 futex 改回 0，读取旧值
    int __oldval = atomic_exchange_release(futex, 0);
    // 如果原来有等待者，唤醒一个
    if (__glibc_unlikely(__oldval != 1))
      __lll_lock_wake_private(__futex);  // → futex(FUTEX_WAKE, 1)
  }
```

- `__oldval == 1`（无等待者）→ 直接返回，**零 syscall**
- `__oldval != 1`（有等待者）→ `futex_wake` 唤醒一个线程，**一次 syscall**

### 6.5 高效的根源

| 场景 | 机制 | syscall 次数 | CPU 占用 |
|------|------|------------|---------|
| 无竞争 | CAS(0→1) | **零** | ~数十周期 |
| 轻度竞争 | CAS(0→2) | **零** | ~数百周期 |
| 激烈竞争 | futex_wait 挂起 | 1次 | **零**（彻底挂起） |
| 唤醒 | futex_wake | 1次 | - |

---

## 7. 矩阵运算中的线程分配

### 7.1 GEMM 并行示意

矩阵乘法 `C = A @ B`，m×k 乘 k×n，按**行分块**给 4 个工作线程：

```
矩阵 A (m×k)               结果 C (m×n)
┌──────────────────┐      ┌──────────────────┐
│  Thread 0        │      │  Thread 0        │
│  处理行 [0, 250)  │      │  计算 C[0:250, :]│
├──────────────────┤      ├──────────────────┤
│  Thread 1        │      │  Thread 1        │
│  处理行 [250, 500)│      │  计算 C[250:500,:]│
├──────────────────┤      ├──────────────────┤
│  Thread 2        │      │  Thread 2        │
│  处理行 [500, 750)│      │  计算 C[500:750,:]│
├──────────────────┤      ├──────────────────┤
│  Thread 3        │      │  Thread 3        │
│  处理行[750,1000) │      │  计算 C[750:1000,:]│
└──────────────────┘      └──────────────────┘
      ▲
      │ 读取 A 的不同行（无数据竞争）
      │
      ▼ 读取 B 的相同列（只读，天然线程安全）
```

**每个线程写不同的行，无数据竞争，无需额外加锁。**

### 7.2 任务分发流程

```
parallel_for(0, m=1000, grain_size=1, lambda)
         │
         ▼
calc_num_tasks_and_chunk_size(0, 1000, 1)
    chunk_size = ceil(1000 / 4) = 250
    num_tasks  = ceil(1000 / 250) = 4
         │
         ▼
_run_with_pool(task, 4)
    ├─ 主线程: task(0) → lambda(0, 250)
    ├─ Worker 0: task(1) → lambda(250, 500)
    ├─ Worker 1: task(2) → lambda(500, 750)
    └─ Worker 2: task(3) → lambda(750, 1000)

完成后 condition_.notify_one() 唤醒主线程的 cv.wait()
```

---

## 8. 线程数配置与调优

### 8.1 默认线程数（树莓派5）

**源码**：`c10/core/thread_pool.cpp:11-24`

```cpp
size_t defaultNumThreads() {
    if (cpuinfo_initialize()) {
        size_t num_cores = cpuinfo_get_cores_count();      // 物理核心数
        size_t num_threads = cpuinfo_get_processors_count(); // 逻辑处理器

        if (num_cores > 0 && num_cores < num_threads)
            return num_cores;  // 返回物理核心数（避免超线程）
    }
    return std::thread::hardware_concurrency();  // 回退
}
```

| 平台 | CPU | 物理核心 | 默认线程数 |
|------|-----|---------|-----------|
| Raspberry Pi 4 | Cortex-A72 | 4 | 4 |
| **Raspberry Pi 5** | **Cortex-A76** | **4** | **4** |

### 8.2 设置方法

```python
import torch

# 全局设置（必须在第一次并行操作前调用）
torch.set_num_threads(4)

# 查看当前值
print(torch.get_num_threads())

# 环境变量（次选）
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
```

### 8.3 线程数选择建议

| 场景 | 建议线程数 | 原因 |
|------|-----------|------|
| 单任务推理 | 4（物理核心数） | 避免超线程竞争，充分利用物理核心 |
| 多任务并行 | max(1, 核心数/任务数) | 避免过载 |
| 低延迟优先 | 核心数 - 1 | 保留主线程资源 |
| 吞吐量优先 | 核心数 | 压满 CPU |

---

## 附录：源码文件索引

| 文件 | 相对路径 | 作用 |
|------|---------|------|
| `aten/src/ATen/Parallel.h` | Parallel.h | 并行 API 定义和后端选择 |
| `aten/src/ATen/Parallel-inl.h` | Parallel-inl.h | parallel_for 模板实现 |
| `aten/src/ATen/ParallelNative.cpp` | ParallelNative.cpp | Native 后端实现（invoke_parallel） |
| `c10/core/thread_pool.h` | thread_pool.h | ThreadPool 类定义 |
| `c10/core/thread_pool.cpp` | thread_pool.cpp | ThreadPool 实现（main_loop） |
| `gcc-13.2.0/libstdc++-v3/include/bits/std_mutex.h` | libstdc++ | std::mutex → __gthread_mutex_lock |
| `gcc-13.2.0/libgcc/gthr-posix.h` | GThread | __gthread_mutex_lock → pthread_mutex_lock |
| `glibc-2.42/nptl/pthread_mutex_lock.c` | glibc NPTL | pthread_mutex_lock → LLL_MUTEX_LOCK_OPTIMIZED |
| `glibc-2.42/sysdeps/nptl/lowlevellock.h` | glibc LLL | lll_lock → __lll_lock_wait + CAS |
| `glibc-2.42/sysdeps/nptl/lowlevellock-futex.h` | glibc futex | futex_wait → INTERNAL_SYSCALL(futex) |
| `linux/kernel/futex/syscalls.c` | Linux Kernel | sys_futex → do_futex → schedule() |

---

### 附录：std::mutex vs pthread_mutex_t vs futex 对应关系

| 层次 | 类型/函数 | 所在文件 |
|------|-----------|---------|
| C++ 标准库 | `std::mutex` | `gcc-13.2.0/libstdc++-v3/include/std/mutex` |
| C++ 基类 | `__mutex_base::_M_mutex` | `gcc-13.2.0/libstdc++-v3/include/bits/std_mutex.h` |
| GThread 抽象 | `__gthread_mutex_t` | `gcc-13.2.0/libgcc/gthr-posix.h` |
| POSIX 线程 | `pthread_mutex_t` / `pthread_mutex_lock()` | `glibc-2.42/nptl/pthread_mutex_lock.c` |
| 低级锁 | `lll_lock()` / `__lll_lock_wait()` | `glibc-2.42/sysdeps/nptl/lowlevellock.h` |
| futex 封装 | `futex_wait()` / `lll_futex_syscall()` | `glibc-2.42/sysdeps/nptl/lowlevellock-futex.h` |
| 内核 | `sys_futex()` / `do_futex()` | `linux/kernel/futex/syscalls.c` |

### 附录：树莓派 ARM64 上的 futex

树莓派5跑的是 64-bit ARM Linux（Debian/Raspbian），内核支持 futex syscall（`__NR_futex`），但**没有 futex2**（futex2 是 x86_64 特有的更高效 futex 实现）。

glibc 通过 `INTERNAL_SYSCALL(futex, ...)` 路由到内核的 `sys_futex`，ARM64 上对应 `linux/arch/arm64/kernel/sys.c` 中的 `sys_futex` 实现，最终调用 `do_futex()`。

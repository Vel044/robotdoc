# 外核、微内核与宏内核的性能评估论文综述

## 1. 外核及外核思想相关（4 篇）

### 1.1 Engler, Kaashoek, O’Toole，1995：Exokernel: An Operating System Architecture for Application-Level Resource Management

**论文介绍**：这是 MIT 最早、也是外核架构的奠基论文之一。需要区分名称：论文中的原型是 **Aegis 外核 + ExOS**，后来的 **Xok/ExOS** 是后续工作。

**评估方法**：作者用 Aegis/ExOS 与传统 Ultrix 做 **内核原语** 对比，逐项测量异常处理、受保护控制传输、系统调用、虚拟内存和磁盘资源操作的延迟/周期，观察把资源管理交给应用后，保护机制是否仍然足够低成本。

**评估结果**：多数 **内核原语** 比 Ultrix 快一个数量级，部分操作达到约 10–100 倍；论文证明了“内核只负责保护，应用负责资源管理”在性能上是可行的。不过它主要测原语，还没有后续 Xok/ExOS 论文那样完整的应用级 **端到端性能** 和并发负载。

**论文**：[ACM DOI](https://doi.org/10.1145/224057.224076)；[开放 PDF](https://www.cs.utexas.edu/~dahlin/Classes/UGOS/reading/engler95exokernel.pdf)。

### 1.2 Kaashoek 等，1997：Application Performance and Flexibility on Exokernel Systems

**论文介绍**：这是 Xok/ExOS 对 1995 年 Aegis/ExOS 原型的后续完整评估，重点回答外核能否既支持专门化应用，又不牺牲未修改 UNIX 程序的性能。

**评估方法**：在同一台 200 MHz Pentium Pro 上，把 Xok/ExOS 与 FreeBSD、OpenBSD 对照。作者测了 **内核保护开销**（受保护管道、系统调用和共享状态检查），再运行 I/O 密集的软件开发流程、Modified Andrew Benchmark、Cheetah HTTP 服务器，并让 CPU 密集和 I/O 密集程序并发执行；每项运行 10 次，报告最小运行时间和并发任务的总时间、最长时间、最短时间。

**评估结果**：未修改 UNIX 应用在 Xok/ExOS 上总体与 BSD 相当，部分应用最高快约 4 倍；Cheetah HTTP 服务器最高快约 8 倍；混合并发负载下，Xok/ExOS 的总吞吐和最坏任务延迟仍具有竞争力。

**论文**：[MIT 原文与实验数据](https://pdos.csail.mit.edu/papers/exo-sosp97/exo-sosp97.html)；[本地译文](./exokernel-engler95-译文.md)。

### 1.3 Ganger 等，2002：Fast and Flexible Application-Level Networking on Exokernel Systems

**论文介绍**：论文评估 Xok/ExOS 的应用级**网络服务性能**，研究 UDP/TCP、包过滤、包环和协议栈放到应用级 libOS 后，能否安全共享网卡并提升网络服务性能。

**评估方法**：作者先对 **包发送、过滤器、通知、内存映射和拷贝路径** 做 profiling，再用 Cheetah HTTP、webswamp 和应用级 TCP forwarder 做 **端到端吞吐** 测试，并与 BSD socket 实现比较；每组测量至少执行 5 次并检查变异系数。

**评估结果**：普通 Xok/ExOS socket 实现已经达到或超过当时 BSD 的性能；经过应用专门化的 Cheetah 吞吐最高提升约 3–8 倍，webswamp 可施加 2–8 倍更重的负载，TCP forwarder 吞吐提升约 50–300%。论文还把收益分解到了具体优化点，而不是只报一个总数。

**论文**：[MIT 论文页面](https://pdos.csail.mit.edu/publications/)，[开放 PDF](https://pdos.csail.mit.edu/papers/exo%3Atocs.pdf)。

## 2. 微内核（5 篇）

### 2.1 Härtig 等，1997：The Performance of µ-Kernel-Based Systems

**论文介绍**：本文首先介绍了将 Linux 移植为 **L4Linux** 的方法：把 Linux 作为 L4 用户态的操作系统服务器运行；然后以 L4Linux 为主要实验平台，与原生 Linux、MkLinux 做系统级比较。论文的核心问题不是单纯介绍 L4Linux，也不是只测 IPC，而是验证高性能微内核能否承载完整且性能可接受的 UNIX 系统。

**评估方法**：作者用 Pentium **cycle** counter 测量 **getpid、IPC、地址空间切换、页故障、管道和同步 RPC**，再运行 AIM 多用户基准和普通 Linux 应用；对快速操作使用多个独立基准交叉验证，避免把系统性测量误差当成随机误差。

**评估结果**：L4Linux 的整体性能通常只比原生 Linux 慢约 5–10%，明显优于基于 Mach 的 MkLinux；经过专门化的 L4 pipe/RPC 在延迟和带宽上还能超过传统 Linux pipe。论文说明 **微内核原语成本** 会直接决定上层 UNIX 系统的性能。

**论文**：[TU Dresden 论文与实验章节](https://os.inf.tu-dresden.de/pubs/sosp97/)；[ACM DOI](https://doi.org/10.1145/268998.266660)。

### 2.2 Hohmuth, Härtig，2001：Pragmatic Nonblocking Synchronization for Real-Time Systems

**论文介绍**：这篇论文一方面介绍了 **Fiasco L4 实时微内核**，另一方面提出了一套适合实时内核的非阻塞同步方法。作者关注的问题是：如果内核大量使用普通锁或长时间关闭中断，其他高优先级任务可能被阻塞，导致 **IRQ 延迟不可预测**。因此，论文把简单数据更新交给 lock-free 的 CAS，把复杂操作交给带 helping 机制的 wait-free lock，并与早期的 single-server 同步方案进行比较。

**评估方法**：论文分成两组实验。第一组是同步原语微基准：在不同 x86 处理器上，分别测量未保护计数器、CAS、Fiasco 的 wait-free lock、旧 single-server lock，以及一次完整的单向 IPC，单位是 CPU cycles。这样可以判断新同步机制本身的成本，并把它和一次 IPC 的成本放在一起比较。第二组是实时性实验：在 200 MHz Pentium Pro 上使用本地 APIC 每 250 µs 产生一次硬件 IRQ，由独立的高优先级用户态线程接收中断并记录相邻中断的实际间隔，最后计算 **最大迟到**。测试期间同时运行缓存冲刷程序和 L4Linux 多用户基准，制造高负载；对照系统包括 Fiasco/L4Linux、L4/x86/L4Linux 和内核态处理 IRQ 的 RTLinux。

**评估结果**：实时性实验中，Fiasco 的最大 IRQ 迟到为约 65 µs，RTLinux 约 58 µs，而 L4/x86 达到约 541 µs；三者平均迟到都小于 1 µs，但最大值明显区分了内核临界区设计的差异。同步实验中，Fiasco wait-free lock 约 245 cycles，旧 single-server 方案约 607–627 cycles，新方案成本降低超过一半；Fiasco 的 IPC 约 653–810 cycles，L4/x86 约 398–438 cycles。论文因此证明：只要内核同步操作具有有界执行时间、避免长时间关闭中断，**用户态 IRQ 处理** 也可以接近内核态实时系统，而不必然牺牲实时性。需要注意的是，实验主要针对单处理器，作者也承认最坏延迟分析仍不完整，65 µs 不能直接当作所有硬件上的通用保证。

**论文**：[USENIX 页面](https://www.usenix.org/conference/2001-usenix-annual-technical-conference/pragmatic-nonblocking-synchronization-real-time-systems)；[完整评估章节](https://www.usenix.org/publications/library/proceedings/usenix01/full_papers/hohmuth/hohmuth_html/index.html)。

### 2.3 Klein 等，2014：Comprehensive Formal Verification of an OS Microkernel

**论文介绍**：这篇论文主要总结 seL4 的完整形式化验证，但同时给出了微内核性能和时间可预测性的评估。

**评估方法**：作者用 cycle 计数器测量 **IPC fast path/slow path**、不同 GCC 优化级别和消息方向，并用性能计数器确认没有缓存冲突和 TLB miss；另外对内核二进制进行带流水线、缓存模型的静态 WCET 分析，计算 **最坏中断延迟**。

**评估结果**：seL4 的已验证 IPC fast path 接近当时最快的 L4 实现；静态分析给出安全的最坏时间上界，而实际测得的中断延迟更低。论文的关键贡献是把 **实测性能** 和 **可证明的时间上界** 分开报告。

**论文**：[ACM/NICTA 开放 PDF](https://sel4.org/Research/pdfs/comprehensive-formal-verification-os-microkernel.pdf)。

### 2.4 Steinberg, Kauer，2010：NOVA: A Microhypervisor-Based Secure Virtualization Architecture

**论文介绍**：NOVA 是微内核式 microhypervisor，论文关注虚拟化隔离机制会给 guest 带来多少性能损失。

**评估方法**：作者分别改变 **nested paging、shadow page table、VPID、宿主页大小和虚拟化退出路径**，测量这些机制的额外成本；再用冷 buffer cache、4 个并行任务编译 Linux 2.6.32，以墙钟时间比较裸机、NOVA、Xen 和 L4Linux。每组取数十次试验的中位数，并关闭 Hyper-Threading、Turbo Boost。

**评估结果**：启用 nested paging、VPID 和合适的宿主页大小后，NOVA 的虚拟化开销很小；没有这些硬件支持时，TLB 和 shadow page table 成本会明显增加。论文说明虚拟化性能不能只报一个总数，必须把硬件辅助路径拆开比较。

**论文**：[EuroSys 2010 PDF](https://www.hypervisor.org/eurosys2010.pdf)；[ACM DOI](https://doi.org/10.1145/1755913.1755935)。

### 2.5 Miemietz 等，2025：MettEagle: Costs and Benefits of Implementing Containers on Microkernels

**论文介绍**：MettEagle 在 L4Re 微内核上实现容器/compartment，研究微内核是否能用更简单的隔离机制提供接近 Linux 的容器性能。

**评估方法**：作者测冷启动延迟 CDF、空闲容器数量的影响、10 Gbit Ethernet UDP ping-pong 和并发网络吞吐；再用 SeBS 测试 Python 空函数、HTML、压缩和图算法，分别执行单个实例和 16 个实例，并与 Linux process、runC、Kata/Firecracker 对照，报告中位数和 5/95 分位。

**评估结果**：L4Re 在许多端到端工作负载上接近 runC，部分 HTML/启动场景更快；文件系统操作、Python 初始化和并发 capability 管理是主要瓶颈。论文还表明，微内核系统的性能问题可以通过具体 profiling 定位，而不是简单归因于“IPC 慢”。

**论文**：[USENIX OSDI 2025 页面](https://www.usenix.org/conference/osdi25/presentation/miemietz)；[开放 PDF](https://www.usenix.org/system/files/osdi25-miemietz.pdf)。

## 3. 宏内核/单体内核（5 篇）

### 3.1 Boyd-Wickizer 等，2010：An Analysis of Linux Scalability to Many Cores

**论文介绍**：论文研究传统 Linux 单体内核在多核机器上是否还能扩展，而不是直接假定必须换成新内核架构。

**评估方法**：作者用 MOSBENCH 的 Exim、memcached、Apache、PostgreSQL、gmake、Psearchy 和 MapReduce，在 48 核机器上比较不同核心数下的吞吐和每核工作量；使用 tmpfs 排除磁盘瓶颈，通过 system time、锁、原子操作、目录项、路由表、DMA 缓冲区和页分配定位 **内核扩展性瓶颈**，然后分别修改应用和内核并复测。

**评估结果**：除 gmake 外，多数应用在 stock Linux 上都出现扩展性瓶颈；经过约 3002 行局部修改后，大多数瓶颈可以消除或显著缓解。论文认为传统单体内核并没有立即失去多核扩展能力。

**论文**：[USENIX 页面](https://www.usenix.org/conference/osdi10/analysis-linux-scalability-many-cores)；[开放 PDF](https://www.usenix.org/events/osdi10/tech/full_papers/Boyd-Wickizer.pdf)。

### 3.2 Ren 等，2019：An Analysis of Performance Evolution of Linux’s Core Operations

**论文介绍**：论文不是比较不同内核架构，而是追踪 Linux 多个版本中 **核心内核操作** 的性能变化，寻找回退原因。

**评估方法**：作者开发 LEBench，测量系统调用、上下文切换、读写、mmap/munmap、fork、poll/select/epoll、send/recv 和 page fault 等 13 类操作，覆盖 Linux 3.0–4.20 共 36 个版本；在内核操作前后直接取时间戳，能绕过 libc 就绕过，并在单台 Intel Xeon 上控制硬件变量。找到回退后，再把相关配置或补丁放回 Redis、Apache、Nginx 验证端到端影响。

**评估结果**：许多 Linux 核心操作在新版本中变慢，主要原因是安全增强、功能增加和配置变化；关闭或优化相关改动后，Redis、Apache、Nginx 的性能分别最高提升约 56%、33% 和 34%。这说明 **内核原语回归** 必须用真实应用再次验证。

**论文**：[论文 PDF](https://www.eecg.toronto.edu/~stumm/Papers/Ren-sosp-19.pdf)；[出版信息](https://www.eecg.toronto.edu/~stumm/Ren-sosp-19.html)。

### 3.3 Banga, Mogul，1998：Scalable Kernel Performance for Internet Servers Under Realistic Loads

**论文介绍**：论文研究 UNIX 网络服务器在真实广域网负载下的扩展性，重点关注 select 和文件描述符分配造成的瓶颈。

**评估方法**：作者分别改变连接到达率和同时存在的连接数，测量 select/ufalloc 的 CPU 负载、profile 和数据缓存影响；然后在 Web proxy 和 Web server 上比较修改前后的 **吞吐量和扩展性**。

**评估结果**：实验室短连接基准会低估真实负载下的开销；改进 select 和文件描述符分配后，Web proxy/server 吞吐最高提升约 58%，并显著改善连接数增加时的扩展性。

**论文**：[USENIX 页面](https://www.usenix.org/conference/1998-usenix-annual-technical-conference/scalable-kernel-performance-internet-servers)；[HTML 全文](https://www.usenix.org/legacy/publications/library/proceedings/lisa97/failsafe/usenix98/full_papers/banga/banga_html/banga.html)。

### 3.4 Molloy，2000：Accept() Scalability in Linux

**论文介绍**：论文专门分析 Linux accept 系统调用在多线程/多进程服务器中的 thundering-herd 问题。

**评估方法**：作者比较多个 accept 修复方案，观察并发等待者被唤醒时的 **系统调用开销、锁竞争和无效唤醒**；再用服务器式连接到达负载测量连接处理吞吐和扩展性。

**评估结果**：论文证明 accept 的唤醒策略会直接限制 Linux 服务器的扩展性，经过针对性修改后可以改善并发连接处理性能。它展示了一个单独的 **系统调用路径问题** 如何在真实服务器负载中放大。

**论文**：[USENIX 页面](https://www.usenix.org/conference/2000-usenix-annual-technical-conference/accept-scalability-linux)；[论文链接](https://www.usenix.org/publications/library/proceedings/usenix2000/freenix/full_papers/molloy/molloy_html/index.html)。

### 3.5 Hwang 等，2021：Rearchitecting Linux Storage Stack for µs Latency and High Throughput

**论文介绍**：论文重新设计 Linux 存储栈，目标是在高吞吐的同时把延迟压到微秒级。

**评估方法**：作者把存储请求按多个队列处理，分别观察排队、优先级、负载均衡和调度对单个 I/O 请求服务时间的影响；然后让几十个 **延迟敏感应用** 与接近硬件极限吞吐的读写应用并发运行，报告平均延迟、p99、p99.9 和硬件利用率。实验不修改应用、网络硬件、CPU 调度器或网络栈。

**评估结果**：新存储栈在竞争负载下仍能保持微秒级平均和尾延迟，同时接近满载使用硬件；论文说明只看平均吞吐会掩盖 **尾延迟** 和实时任务退化。

**论文**：[USENIX OSDI 2021 页面](https://www.usenix.org/conference/osdi21/presentation/hwang)。

## 4. 三类论文的共同方法和差异

| 维度           | 外核                                        | 微内核                                         | 宏内核                                     |
| -------------- | ------------------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| 最重要的微指标 | 保护原语、资源绑定、libOS 直达硬件的成本    | IPC、通知、上下文切换、IRQ/WCET                | syscall、锁竞争、缓存/TLB、调度和 I/O 队列 |
| 最重要的宏指标 | 专门化 libOS 的端到端收益、混合负载全局性能 | 服务/容器/VM 的端到端延迟、隔离代价和尾延迟    | 吞吐、每核扩展性、真实并发负载、p99/p99.9  |
| 常见对照       | BSD/Ultrix、普通 socket、普通 VM            | native Linux、MkLinux、L4/x86、RTOS、runC/Kata | stock kernel 与补丁/新栈、不同版本         |
| 容易犯的错误   | 只测原语，不测 libOS 应用                   | 只报最佳 IPC，不测缓存和真实服务               | 只报平均吞吐，不报并发和尾延迟             |

## 5. 对 RobotOS/exokernel 的落地建议

建议将性能测试固定成三层，而不是为每个应用再建一套路径：

1. **Kernel primitive 层（微观）**
   - SVC/异常进入和返回；
   - IPC、notification、事件环；
   - 线程创建、调度、上下文切换；
   - frame alloc、map/unmap、缺页；
   - GIC IRQ 到用户线程的延迟；
   - DMA buffer 分配、缓存维护和设备可见性。

2. **Subsystem 层（半宏观）**
   - UART：事件率、往返延迟、突发输入下的丢包/积压；
   - xHCI：TRB 提交到 Event Ring、批量传输吞吐、并发端点；
   - 文件/日志：顺序吞吐、随机小 I/O、后台 I/O 对实时任务的干扰；
   - 内存：并发页框分配、回收和 DMA 连续内存压力。

3. **Robot end-to-end 层（宏观）**
   - 传感器 IRQ → 用户态驱动 → 控制任务 → 执行器提交的端到端延迟；
   - p50/p95/p99/p99.9、最大延迟和 deadline miss rate；
   - 控制任务与日志、网络、文件系统、其他设备并发时的最坏退化；
   - 每次控制周期的 CPU cycles、内存页框、IRQ 次数和 DMA 拷贝字节数。

每个结果至少记录：硬件/板卡或 QEMU 版本、内核提交号、编译器和优化级别、feature 配置、CPU 亲和性、缓存热/冷状态、样本数、均值/中位数/分位数/最大值。真实 USB 暂时不可用时，可以先用 QEMU xHCI 做可重复的机制和端到端测试；真实设备测试只作为后续硬件验证，不要把两者的数据混在同一张图里。

# 系统级 Profiling 技术 — 文献调研

## 1. torch.profiler（PyTorch 推理性能分析）

### PyTorch Profiler — 官方文档与教程
- **作者**：PyTorch 官方团队
- **年份**：2021–2024（持续维护）
- **出版**：PyTorch 官方文档
- **链接**：https://docs.pytorch.org/docs/stable/profiler.html | 教程：https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **摘要**：支持 CPU/GPU 事件追踪、内存分配追踪、FLOPs 估计；输出 Perfetto 格式供 Chrome DevTools / ui.perfetto.dev 可视化；内置 `key_averages()` 自动识别热点算子并给出优化建议。对于无 GPU 的 ARM 推理场景，CPU 追踪和内存快照是核心功能。

---

## 2. perf（Linux 性能计数器与采样）

### Choosing a Linux Tracer（综合选型指南）
- **作者**：Brendan Gregg
- **年份**：2015–2024（持续更新）
- **出版**：brendangregg.com（行业标准参考）
- **链接**：https://www.brendangregg.com/blog/2015-07-08/choosing-a-linux-tracer.html | perf 示例集：https://www.brendangregg.com/perf.html
- **摘要**：系统介绍 perf 与 ftrace 在内核追踪、性能计数器、函数级 profiling 中的应用，涵盖采样（sampling）、追踪点（tracepoints）、火焰图（flame graph）生成等技术，是 Linux 性能分析的入门核心参考。

### Linux Tracing in 15 Minutes
- **作者**：Brendan Gregg
- **年份**：2016
- **出版**：brendangregg.com
- **链接**：https://www.brendangregg.com/blog/2016-12-27/linux-tracing-in-15-minutes.html
- **摘要**：快速入门 perf/ftrace/BPF 工具链，适合在树莓派等嵌入式 Linux 系统上快速定位推理延迟瓶颈。

---

## 3. ftrace（Linux 内核函数追踪）

### Ftrace - Function Tracer — 官方内核文档
- **作者**：Linux 内核官方文档
- **年份**：持续更新
- **出版**：The Linux Kernel Documentation
- **链接**：https://docs.kernel.org/trace/ftrace.html
- **摘要**：Linux 内核内置追踪工具，支持函数追踪、事件追踪、动态 Kprobes；适用于调试调度延迟、分析实时任务执行时间、定位内核-用户态切换开销，在树莓派（ARM Linux）上无需额外安装即可使用。

### Real-Time Performance in Linux: Harnessing PREEMPT_RT for Embedded Systems
- **作者**：Runtime Research 团队
- **年份**：2024
- **出版**：Runtime Resources
- **链接**：https://runtimerec.com/wp-content/uploads/2024/10/real-time-performance-in-linux-harnessing-preempt-rt-for-embedded-systems_67219ae1.pdf
- **摘要**：针对机器人、汽车控制等实时系统，介绍 PREEMPT_RT 补丁与 perf/ftrace 结合的实时性能分析方法，与树莓派实时推理场景直接相关。

---

## 4. 可视化与综合追踪框架

### Perfetto: Production-grade Tracing and Analysis
- **作者**：Google Perfetto 开源项目
- **年份**：2018–2024
- **出版**：GitHub 开源项目
- **链接**：https://github.com/google/perfetto | UI：https://ui.perfetto.dev
- **摘要**：生产级追踪框架，可可视化 ftrace/perf 原始数据、CPU 调度分析、内存追踪和自定义事件；torch.profiler 默认也以 Perfetto 格式输出，适合同时分析 Python 层与 kernel 层的端到端延迟。

---

## 5. strace（系统调用追踪）

strace 目前无专门发表的学术论文，以官方手册页和工具文档为准：
- **man 页**：`man strace`
- **官方文档**：https://strace.io/
- **用途说明**：追踪进程的系统调用序列，适合分析推理过程中的 I/O 等待、内存映射（mmap）、共享内存操作等系统级行为；但 overhead 较高，不适用于精确计时，通常与 perf/ftrace 配合使用。

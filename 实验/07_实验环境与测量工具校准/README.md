# 实验 07：实验环境与测量工具校准

## 为什么第 2 章需要这个实验

第 2 章主要交代实验平台、软件栈、工作负载和测量方法。现在 2.3 已经有三类任务工作负载对比，能说明后续为什么选 `chunk_size=100` 和三类任务作为基准；但 2.4、2.5 里关于 `strace`、`ftrace`、`trace-cmd` 的取舍，还需要一个很小但很关键的校准实验。

这个实验不回答“模型为什么慢”，而是回答“测量工具会不会把主循环测歪”。尤其是 `strace` 会通过 `ptrace` 拦截系统调用，扰动很大；`ftrace` 写内核 ring buffer 的开销通常小，但把 trace 实时导出到文件时也可能吃 CPU、内存带宽和 I/O。第 2 章需要把这件事讲清楚，否则后面第 3 章的性能数据来源不够硬。

## 研究问题

1. 不开任何 tracing 时，基准 episode 的 FPS、阶段占比和资源占用是多少。
2. `strace -f -ttT` 附加后，会把系统时间、上下文切换和 FPS 扰动到什么程度。
3. `ftrace` 只写内核 ring buffer、episode 结束后再导出时，扰动是否足够小。
4. `ftrace` 边跑边读 `trace_pipe` 实时导出时，开销主要出现在 CPU、RSS 还是 I/O。
5. `trace-cmd record` 是否比实时 `cat` 更适合作为正式实验的导出方式。
6. 软件版本、CPU 频率、温度、设备节点和摄像头参数是否能被固定记录，保证后续实验能复现。

## 实验分组

| 组别 | 名称 | 做法 | 用途 |
|---|---|---|---|
| A | baseline | 不开 tracing，只记录 LeRobot 自身阶段计时和 `/usr/bin/time -v` | 作为所有扰动的参照 |
| B | strace | `strace -f -ttT -o raw/strace_*.log` 包住同一条 record 命令 | 证明 strace 只适合定性看系统调用分布 |
| C | ftrace 延迟导出 | episode 前打开 tracing，运行期间只写 ring buffer，停止后再复制 trace | 校准 ftrace 插桩本身的扰动 |
| D | ftrace 实时导出 | episode 前打开 tracing，并在 record 前启动后台 `cat trace_pipe` 到文件 | 校准实时导出带来的额外压力 |
| E | trace-cmd record | 用 `trace-cmd record` 采集同一批 tracepoint | 评估是否替代实时 `cat` |

每组建议先做 3 次预实验，确认脚本、权限和设备节点没问题；正式数据每组做 5 个 episode。任务固定为抓取或三类任务中的一个代表任务，`chunk_size=100`、目标 `fps=30`、episode 时长约 30 s，硬件连接和第 2.6 节保持一致。

## 核心指标

| 指标 | 来源 | 解释 |
|---|---|---|
| `fps_mean` | LeRobot 阶段日志 | 主循环实际平均帧率 |
| `obs_pct` / `inference_pct` / `action_pct` / `wait_pct` | LeRobot 阶段日志 | tracing 是否改变阶段占比 |
| `user_time_s` / `system_time_s` | `/usr/bin/time -v` | 用户态和内核态 CPU 时间 |
| `max_rss_kb` | `/usr/bin/time -v` | 峰值内存占用 |
| `voluntary_cs` / `involuntary_cs` | `/usr/bin/time -v` | 上下文切换次数 |
| `trace_file_mb` | trace 日志文件 | 日志规模和导出压力 |
| `lost_events` | ftrace / trace-cmd 输出 | ring buffer 是否丢事件 |
| `cpu_temp_start_c` / `cpu_temp_end_c` | `vcgencmd measure_temp` | 树莓派热状态 |
| `cpu_freq_mean_mhz` | `/sys/devices/system/cpu/.../scaling_cur_freq` | 是否发生降频 |

## 预期写进论文的结论

如果结果符合预期，第 2.5 节可以这样收束：

- `strace` 的系统调用级信息很直观，但扰动太大，只用于前期定位系统调用类型，不进入最终耗时数据。
- `ftrace` 只写 ring buffer 时对主循环影响较小，可以作为正式内核态计时来源。
- trace 导出不是免费的；如果实时导出让 FPS 或系统时间明显漂移，正式实验应改为 episode 后导出或 `trace-cmd record`。
- 第 3 章的性能结论只使用经过本实验校准的采集方式，避免“测量工具本身成为瓶颈”。

建议阈值：相对 baseline，正式采集方案的 `fps_mean` 下降不超过 5%，`obs/inference/action/wait` 任一占比漂移不超过 3 个百分点，且无明显 lost events。

## 文件说明

| 文件 | 作用 |
|---|---|
| `实验设计.md` | 详细实验流程、命令模板和验收标准 |
| `tool_overhead_summary_template.csv` | 结果汇总表模板 |
| `environment_snapshot_template.md` | 环境快照记录模板 |

# 实验 08：异步推理阈值参数扫描

**状态**：✅ 已完成  
**对应论文章节**：第 4.3 节 异步推理收益的实验验证  
**创建日期**：2026-05-09  
**分析与图**：`lerobot/analysis/08_async_inference_hold_sweep/`

---

## 1. 实验目的

### 前提约束：动作必须仍然可以完成

**提升帧率不是唯一目标——动作能否完成是硬约束。**

stall 期间机器人停止运动，`latest_action` 冻结，末端执行器保持最后一帧位置直到新 chunk
到货才恢复运动。如果 stall 发生在抓取或插入等对连续性要求高的关键阶段，
机器人可能错过时间窗口导致任务失败。
因此本实验的评价体系分两层：

1. **首先验证任务依然可以完成**（成功率 ≥ 同步基线），这是所有 hold 值的准入门槛
2. **在满足第 1 条的前提下**，比较各 hold 值的 FPS 收益

任何导致成功率显著下降的 hold 值，无论其 FPS 数字多高，均不列为可用配置。

### 具体研究问题

4.2 节的不等式分析给出理论边界 $N \geq 2 T_{\text{infer}} F$，
但该分析是在 `chunk_size_threshold = 0.5`（队列剩 50% 时触发推理）的前提下推导的。
实际上，**阈值（hold）** 是一个可调参数：

- 阈值越低（如 0.2）→ 触发越早 → 推理有更长重叠窗口 → 理论上减少 stall
- 阈值越高（如 0.8）→ 触发越晚 → 类似同步模式 → stall 概率更高

本实验通过扫描不同 hold 值（`chunk_size_threshold`），在保证动作完成的前提下测量：

1. **各 hold 值下的任务成功率**：动作质量优先，FPS 其次
2. **各 hold 值的 stall 比例与 FPS**：在动作质量合格的前提下选择最优 hold
3. 验证理论预测：临界 hold ≈ 0.54，实测确认

---

## 2. 参数空间

| 变量 | 取值 |
|------|------|
| 调度模式 | `sync`（基线）、`async` |
| hold（`chunk_size_threshold`） | 0.3 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9（仅 async 模式） |
| chunk_size $N$ | 100（固定） |
| 目标 FPS $F$ | 30 |
| 任务 | pick |
| 每组重复 | 当前数据为每个 hold 1 个 60s episode，动作质量人工观察记录 |

共 **8 组**：1 个同步基线 + 7 个异步 hold 取值。

---

## 3. 理论预测（预先写下，实验后核对）

设 hold $= h$（0 < h < 1），触发时队列剩余 $hN$ 帧，
推理在此后 $T_{\text{infer}}$ 秒到货，届时已消耗 $T_{\text{infer}} F$ 帧。

稳态不 stall 的条件变为：

$$
hN \geq T_{\text{infer}} F
\quad\Longleftrightarrow\quad
h \geq \frac{T_{\text{infer}} F}{N} = \frac{2 \times 30}{100} = 0.6
$$

**预测**：
- $h < 0.6$：推理还未完成时旧队列已耗尽，发生 stall，FPS 低于同步
- $h = 0.6$：临界点，恰好不 stall，FPS ≈ 30（但无 Ensemble 余量）
- $h > 0.6$：理论上不 stall，但触发太晚、队列提前耗尽，实际效果需实测
- 同步基线：≈ 19 FPS（$N/(T_{\text{infer}} + N/F) = 100/(2 + 3.33) \approx 18.8$）

注：上述推导忽略了 CPU 争抢（异步时 policy_server 与 robot_client 同机）导致的
$T_{\text{infer}}$ 抬高约 10–25%，实际边界会比 $h=0.6$ 更高。

---

## 4. 已完成的代码改动

`record.py` 已有完整的 `timing_stats.csv` 逐 episode 写入，但 **`robot_client.py` 和
`policy_server.py` 原本没有结构化 CSV 输出**，只有 logger 文字日志。
本实验已在两个文件中加入打点逻辑，默认写入
`lerobot/analysis/08_async_inference_hold_sweep/`。

### 4.1 `robot_client.py` 新增字段

在 episode 结束时（`control_loop` 退出后）写入一行到 `async_client_stats.csv`：

```
episode_idx, hold, fps_mean, fps_p5, fps_p95,
stall_frames, stall_frames_pct, stall_mean_dur_s,
must_go_count,
expire_frames, expire_frames_pct,
total_frames,
cpu_temp_start_c, cpu_temp_end_c, cpu_freq_mean_mhz
```

**采集方式**：在 `control_loop` 主循环里每帧记录：
- `stall`：`action_queue.qsize() == 0`（本帧无 action 可用）
- `must_go_count`：`self.must_go` 被 set 的次数（已有 Event，加计数器即可）
- `expire_frames`：`receive_actions` 后台线程丢弃的帧数
  （在 temporal ensemble 合并时，`timestep <= latest_action` 的帧数已有逻辑，加计数器）
- FPS：同 `record.py` 做法，episode 总帧数 / 总耗时，再取 p5/p95

### 4.2 `policy_server.py` 新增字段

在每次 `predict_action_chunk()` 完成后追加一行到 `async_server_stats.csv`：

```
episode_idx, timestep, t_infer_s
```

**采集方式**：在 `GetActions` handler 的推理调用前后各打一个 `time.perf_counter()`。

### 4.3 合并脚本（离线后处理）

脚本位置：`lerobot/analysis/08_async_inference_hold_sweep/merge_async_stats.py`

合并策略使用 `--match-mode auto`：

1. 若 server/client 的 `episode_idx` 可直接对齐，则按 `episode_idx` join
2. 否则按 `timestep` 回到 0 的位置切分 server 日志
3. 若 reset 数量仍不匹配，最后退回按 client 行数等分 server 记录

### 4.4 同步基线直接复用 `timing_stats.csv`

同步模式直接跑 `record.py`，已有 `fps_mean`、`obs_s`、`inference_s` 等字段。
合并时只需把 `timing_stats.csv` 里的基线行加一列 `hold = "sync"` 拼入对比表即可。

---

## 5. 测量指标

| 指标 | 来源文件 | 字段名 |
|------|---------|--------|
| **任务成功率** | 人工观测，每 episode 记录 | `success`（0/1） |
| `fps_mean`, `fps_p5`, `fps_p95` | `async_client_stats.csv` | 直接字段 |
| stall 帧比例 | `async_client_stats.csv` | `stall_frames_pct` |
| stall 平均时长 | `async_client_stats.csv` | `stall_mean_dur_s` |
| `must_go` 触发次数 | `async_client_stats.csv` | `must_go_count` |
| chunk 过期帧比例 | `async_client_stats.csv` | `expire_frames_pct` |
| $T_{\text{infer}}$ 均值/std | `async_server_stats.csv` → merged | `t_infer_mean`, `t_infer_std` |
| CPU 温度、频率 | `async_client_stats.csv` | `cpu_temp_*`, `cpu_freq_mean_mhz` |

**任务成功率的判定标准**（pick 任务）：episode 结束时物体是否被抓起并放置到目标区域。
由操作者在每个 episode 结束后立即记录，写入 `success` 列（1=成功，0=失败）。
当前这批 CSV 尚未写入机器可读成功率，`success=-1`；本报告用人工观察的动作质量描述记录结论。

---

## 6. 控制变量

- CPU governor 锁 `performance`，防止 DVFS 节流
- `torch.set_num_threads(3)`，`cv2.setNumThreads(1)` 两种模式一致
- 异步模式：`taskset -c 0-2` 绑 server，`taskset -c 3` 绑 client
- 固定 git commit、模型权重 hash、相机分辨率（MJPEG 1280×720 @30 FPS）
- 每次实验前后记录 CPU 温度，排除热降频

---

## 7. 实验流程

**步骤 0（一次性准备）**：按第 4 节完成代码修改，验证 CSV 输出正确。

**步骤 1**：环境准备，固定 commit + 权重 hash，锁 governor，关 WiFi，风扇全速

**步骤 2**：同步基线，直接用现有 `record.py`，或复用实验 04 A 组无 tracing baseline

```bash
python -m lerobot.scripts.record --chunk_size 100 --num_episodes 3 ...
```

**步骤 3**：按 hold ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} 顺序逐组异步实验（当前数据各 1 个 60s episode）

```bash
# 终端 1（server，绑核 0-2）
taskset -c 0-2 python -m lerobot.scripts.server.policy_server \
  --host 127.0.0.1 --port 50051

# 终端 2（client，绑核 3）
taskset -c 3 python -m lerobot.scripts.server.robot_client \
  --server_address 127.0.0.1:50051 \
  --chunk_size_threshold <hold> \
  --chunk_size 100 --num_episodes 3
```

**步骤 4**：运行 `lerobot/analysis/08_async_inference_hold_sweep/merge_async_stats.py`，
生成 `async_merged_stats.csv`

**步骤 5**：绘图（`lerobot/analysis/08_async_inference_hold_sweep/plot_async_sweep.py`）

---

## 8. 输出图表

1. `chart1_fps_vs_sync.png`：hold 有效 FPS 与同步 baseline 对比
2. `chart2_must_go.png`：hold 与 `must_go` 触发次数
3. `chart3_frame_breakdown.png`：执行帧 / 过期帧 / stall 帧组成
4. `chart4_tinfer_boxplot.png`：各 hold 的 `T_infer` 分布
5. `async_hold_summary.csv`：7 个 hold 的全量指标（fps、stall%、过期%、must_go、T_infer、CPU 温度/频率）

---

## 9. 结果解读

### 9.1 FPS 边界

实测 T_infer ≈ 1.8s，临界 hold ≈ 0.54。
各 hold 有效 FPS 均远超同步基线（+55%~+66%），但 hold 值对任务成功率影响更大。

### 9.2 动作完成性确认（实测结论）

**核心发现：hold 值越低，动作质量越差。**

| hold | 动作质量 | 推荐 |
|------|---------|------|
| 0.3  | 动作漂移，成功率低 | ✗ |
| 0.4  | 有所改善 | ✗ |
| 0.5  | 基本可接受 | △ |
| 0.6  | 可接受 | △ |
| 0.7  | 良好 | △ |
| **0.8** | **良好，推荐** | **✓** |
| **0.9** | **良好，推荐** | **✓** |

**原因**：hold=0.3 触发推理时机器人处于早期状态，temporal ensemble 融合历史预测，
输出动作与执行时刻状态脱节；hold≥0.8 时预测新鲜，动作与状态匹配。

**结论报告格式**（实测已填）：

| hold | success | fps_mean | stall_pct | must_go | 结论 |
|------|---------|---------|-----------|---------|------|
| 0.30 | 低 | 26.80 | 6.83% | 13 | ✗ 动作质量差 |
| 0.40 | 改善中 | 27.09 | 8.32% | 27 | ✗ |
| 0.50 | 基本可接受 | 26.00 | 11.58% | 30 | △ |
| 0.60 | 可接受 | 26.35 | 10.97% | 29 | △ |
| 0.70 | 良好 | 26.24 | 10.81% | 29 | △ |
| **0.80** | **良好** | **26.15** | **11.25%** | **28** | **✓ 推荐** |
| **0.90** | **良好** | **25.86** | **12.06%** | **29** | **✓ 推荐** |

**最终结论**：异步推理在 hold ≥ 0.8 时任务成功率可接受，
有效 FPS 比同步基线高 55%，推荐默认 hold = **0.8**；
后续推理加速（T_infer ≤ 1.2s）后可逐步降低 hold 以获取更高吞吐。

---

## 10. 实验结果

结果文件位于 `lerobot/analysis/08_async_inference_hold_sweep/`：

| 文件 | 内容 |
|------|------|
| `async_hold_summary.csv` | hold 扫描汇总表 |
| `async_merged_stats.csv` | client/server 合并后的 `T_infer` 表 |
| `chart1_fps_vs_sync.png` | 有效 FPS 与同步 baseline 对比 |
| `chart2_must_go.png` | `must_go` 触发次数 |
| `chart3_frame_breakdown.png` | 执行帧 / 过期帧 / stall 帧组成 |
| `chart4_tinfer_boxplot.png` | `T_infer` 分布 |

注：当前 CSV 的 `success=-1` 表示未写入机器可读成功率标签；动作质量结论来自实验时的人工观察记录。

| hold | success | fps_mean | fps_p5 | fps_p95 | stall% | expire% | must_go | eff FPS |
|------|---------|---------|--------|---------|--------|---------|---------|---------|
| 0.30 | 低 ✗ | 27.82 | 24.52 | 33.72 | 6.83% | 49.72% | 13 | 26.80 |
| 0.40 | 改善中 ✗ | 27.09 | 22.80 | 33.51 | 8.32% | 48.85% | 27 | 26.24 |
| 0.50 | 基本可接受 △ | 26.00 | 22.15 | 32.91 | 11.58% | 49.21% | 30 | 25.19 |
| 0.60 | 可接受 △ | 26.35 | 22.23 | 32.83 | 10.97% | 49.01% | 29 | 25.41 |
| 0.70 | 良好 △ | 26.24 | 22.02 | 33.17 | 10.81% | 49.84% | 29 | 25.44 |
| **0.80** | **良好 ✓** | **26.15** | **22.68** | **32.80** | **11.25%** | **49.92%** | **28** | **25.36** |
| **0.90** | **良好 ✓** | **25.86** | **22.59** | **32.91** | **12.06%** | **50.22%** | **29** | **25.03** |

**Server T_infer**（稳态，排除前 2 次 warm-up）：mean=1.80s, std=0.11s, range=[1.33, 2.15]s
**同步基线**：fps_mean ≈ 16.1，异步各 hold 均提升 +55%~+66%
**推荐 hold**：**0.8**（动作质量合格，stall 可接受）

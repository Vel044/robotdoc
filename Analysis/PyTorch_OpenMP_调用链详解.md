# PyTorch OpenMP 并行调用链详解（你的环境）

> 基于 `lerobot` 环境 + PyTorch 2.7.1+cpu + ARM64 平台
>
> 你的环境：**OpenMP 后端**，不是 c10::ThreadPool

---

## 目录

- [1. 环境确认](#1-环境确认)
- [2. 核心问题：为什么要并行？](#2-核心问题为什么要并行)
- [3. 完整调用链总览](#3-完整调用链总览)
- [4. 第1跳：机器人控制循环层（Python）](#4-第1跳机器人控制循环层python)
- [5. 第2跳：Python/ATen 绑定层](#5-第2跳pythonaten-绑定层)
- [6. 第3跳：ATen BLAS 分派层](#6-第3跳aten-blas-分派层)
- [7. 第4跳：GEMM 计算层](#7-第4跳gemm-计算层)
- [8. 第5跳：并行层](#8-第5跳并行层)
- [附录：关键源码索引](#附录关键源码索引)

---

## 1. 环境确认

你的 PyTorch 配置：

| 配置项 | 值 | 含义 |
|-------|-----|-------|
| `AT_PARALLEL_OPENMP` | **1** | OpenMP 已启用 |
| `AT_PARALLEL_NATIVE` | 0 | Native 后端已禁用 |
| `USE_OPENMP` | ON | OpenMP 编译开启 |
| `libgomp.so.1` | 已加载 | OpenMP 运行时库存在 |
| `num_threads` | 4 | 默认 4 线程 |

**结论：你的环境走 OpenMP 后端 (`#pragma omp parallel`)，不是 c10::ThreadPool。**

---

## 2. 核心问题：为什么要并行？

**矩阵乘法（GEMM）是瓶颈**。一个典型的 ACT forward 包含多次矩阵乘法：
- 图像通过 CNN backbone：无数小矩阵乘法
- Transformer 的 Attention：Query-Key-Value 投影、输出投影
- 输出层的线性变换

这些矩阵乘法的计算量远大于内存访问开销，**单个线程吃不满 CPU 的算力单元**。所以必须把矩阵按行切片，分给多个线程同时计算。

---

## 3. 完整调用链总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第1跳：机器人控制循环层（Python）                                              │
│ lerobot/src/lerobot/record.py::record_loop()                                 │
│     ↓                                                                     │
│ lerobot/src/lerobot/utils/control_utils.py::predict_action()              │
│     ↓                                                                     │
│ lerobot/src/lerobot/policies/act/modeling_act.py::select_action()          │
│     ↓                                                                     │
│ lerobot/src/lerobot/policies/act/modeling_act.py::predict_action_chunk()   │
│     ↓                                                                     │
│ lerobot/src/lerobot/policies/act/modeling_act.py::ACT.forward()            │
└──────────────┼───────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第2跳：Python/ATen 绑定层                                                    │
│ pytorch/torch/nn/modules/linear.py::nn.Linear.forward()                    │
│     ↓                                                                     │
│ pytorch/torch/nn/functional.py::F.linear  ──→  torch._C._nn.linear (C++绑定)
│     ↓                                                                     │
│ pytorch/aten/src/ATen/native/Linear.cpp::at::native::linear()             │
│     ↓                                                                     │
│ pytorch/aten/src/ATen/native/Linear.cpp::at::addmm()  ← 融合bias+matmul   │
└──────────────┼───────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第3跳：ATen BLAS 分派层                                                     │
│ pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp::cpublas_gemm_impl()      │
│     ↓                                                                     │
│ pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp::gemm_core_()             │
│     ↓ 分派至                                                                │
│ gemm_notrans_() 或 gemm_transa_()                                          │
└──────────────┼───────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第4跳：GEMM 计算层（真正计算发生地）                                          │
│ 三重循环：k累加 × n列展开 × m行SIMD                                         │
│ float/double: 4路SIMD向量乘法（第105行）                                    │
│ Half/BFloat16: sum()函数高精度累加（第142行）                               │
└──────────────┼───────────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 第5跳：并行层                                                               │
│ pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp::parallel_for()  ← 第391行│
│     ↓                                                                     │
│ pytorch/aten/src/ATen/ParallelOpenMP.h::invoke_parallel()  ← 第17行       │
│     ↓                                                                     │
│ #pragma omp parallel  ← 第25行，GCC生成 libgomp::pthread_create()          │
│     ↓                                                                     │
│ pthread_create() → Linux Kernel                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 第1跳：机器人控制循环层（Python）

### 4.1 分段调用链

```
lerobot/src/lerobot/record.py::record_loop()
    ↓
lerobot/src/lerobot/utils/control_utils.py::predict_action()
    ↓
lerobot/src/lerobot/policies/act/modeling_act.py::ACTPolicy.select_action()
    ↓
lerobot/src/lerobot/policies/act/modeling_act.py::ACTPolicy.predict_action_chunk()
    ↓
lerobot/src/lerobot/policies/act/modeling_act.py::ACT.forward()
```

### 4.2 record_loop — 机器人控制循环入口

**源码路径**：`lerobot/src/lerobot/record.py`

**意图**：机器人在每个 episode 内反复循环：采集观察 → 推理决策 → 执行动作 → 存储数据。这是整个系统的最外层驱动。

```python
# lerobot/src/lerobot/record.py（相关行：第367行、第372-381行）
inference_start_t = time.perf_counter()

if policy is not None and preprocessor is not None and postprocessor is not None:
    # policy 路径：观察 → predict_action → 动作字典
    action_values = predict_action(
        observation=observation_frame,
        policy=policy,                       # ACTPolicy 实例，内部有 self.model(batch)
        device=get_safe_torch_device(policy.config.device),  # policy 指定的设备
        preprocessor=preprocessor,           # 预处理流水线（归一化等）
        postprocessor=postprocessor,        # 后处理流水线（动作裁剪等）
        use_amp=policy.config.use_amp,       # 是否用 AMP（CPU一般不用）
        task=single_task,
        robot_type=robot.robot_type,
    )

    # action_values 是 (action_dim,) 的 tensor，转换为 {action_name: float} 字典
    action_names = dataset.features["action"]["names"]
    act_processed_policy: RobotAction = {
        f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
    }

inference_end_t = time.perf_counter()
# inference 计时结束，后续记录到 episode 存储
```

### 4.3 predict_action — 格式胶水层

**源码路径**：`lerobot/src/lerobot/utils/control_utils.py:125-189`

**意图**：numpy数组 → PyTorch tensor（float32, CHW, batch-first）、预处理、调用模型推理、后处理、结果移回CPU。

```python
# lerobot/src/lerobot/utils/control_utils.py:125-189
def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: Callable,
    postprocessor: Callable,
    use_amp: bool = False,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    observation = copy(observation)
    with torch.inference_mode():
        # step 1: numpy → tensor，BGR/HWC → CHW，归一化到 [0,1]
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])           # numpy转tensor
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255.0  # [0,255]→[0,1]
                observation[name] = observation[name].permute(2, 0, 1).contiguous()   # HWC→CHW
            observation[name] = observation[name].unsqueeze(0).to(device)     # 加batch维，迁入设备

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)     # 预处理（数据增强等）
        action = policy.select_action(observation) # ← 触发C++推理（核心）
        action = postprocessor(action)              # 后处理（动作裁剪等）
        action = action.squeeze(0).to("cpu")       # 去batch维，结果回CPU
    return action
```

### 4.4 ACTPolicy.select_action — 动作队列管理

**源码路径**：`lerobot/src/lerobot/policies/act/modeling_act.py:99-121`

**意图**：一次推理预测 `n_action_steps` 步动作，但机器人控制循环每帧只执行一步，所以用队列缓冲。

```python
# lerobot/src/lerobot/policies/act/modeling_act.py:99-121
@torch.no_grad()
def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    self.eval()

    if self.config.temporal_ensemble_coeff is not None:
        # 时间集成：多个历史预测做加权平均来平滑动作
        actions = self.predict_action_chunk(batch)
        return self.temporal_ensembler.update(actions)

    # 动作队列为空时，才触发一次模型推理（不是每帧都推理！）
    if len(self._action_queue) == 0:
        # predict_action_chunk 返回 (batch_size, chunk_size, action_dim)
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
        # (B, S, D) → (S, B, D)，队列里存成"时间优先"形式
        self._action_queue.extend(actions.transpose(0, 1))

    return self._action_queue.popleft()  # 每次只弹出一帧给机器人
```

**`predict_action_chunk` 方法**（第124行）：调用 `ACT.forward`，返回完整动作序列。

### 4.5 ACT.forward — 神经网络前向传播

**源码路径**：`lerobot/src/lerobot/policies/act/modeling_act.py:377`

**意图**：CNN提图像特征 → Transformer Encoder(VAE)算隐变量 → Transformer Decoder解码动作序列。内部全部是 `linear()` 即矩阵乘法。

#### 4.5.1 ACT.forward 核心流程

```python
# lerobot/src/lerobot/policies/act/modeling_act.py:377
def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
    # 1. 图像 → CNN特征 (batch_size, n_cameras, C, H, W) → (batch_size, feat_dim)
    image_features = self.cnn(batch["observation.images"])

    # 2. Transformer Encoder (VAE): 图像+动作序列 → 隐变量分布
    cls_token_out = self.vae_encoder(vae_encoder_input)[0]   # 取[CLS] token
    latent_params = self.vae_encoder_latent_output_proj(cls_token_out)
    mu, log_sigma_x2 = latent_params.split([self.config.latent_dim, self.config.latent_dim], dim=-1)

    # 3. Transformer Decoder: 隐变量 + 当前状态 → 未来动作序列
    decoder_output = self.decoder(decoder_input)            # self.decoder 是 nn.Transformer
    actions = self.action_head(decoder_output)            # nn.Linear，触发 linear() 调用
    return actions, (mu, log_sigma_x2)
```

#### 4.5.2 推理时 vs 训练时的区别

| | 推理 (`training=False`) | 训练 (`training=True`) |
|---|---|---|
| VAE encoder | **不走**，latent = 全零 | **走**，用 action 序列算 mu/logsigma |
| VAE loss | 不计算 | KL + reconstruction loss |
| 其他模块 | 相同 | 相同 |

#### 4.5.3 配置参数 → 各层 tensor 形状

| 配置参数 | 值 | 对应 linear 形状 |
|---------|------|----------------|
| `dim_model=512`, `n_heads=8` | 每头 64 维 | MultiheadAttention 内部 Q/K/V = 512→512 |
| `dim_feedforward=3200` | FFN 中间维度 | `linear1`: 512→3200, `linear2`: 3200→512 |
| `n_encoder_layers=4` | Encoder 4 层 | 上述模块 × 4 |
| `n_decoder_layers=1` | Decoder 1 层（ACT 原始 bug） | 上述模块 × 1 |
| `chunk_size=100` | Decoder 输入序列长度 | Decoder Q 序列长度 = 100 |
| `latent_dim=32` | VAE 隐变量维度 | `encoder_latent_input_proj`: 32→512 |
| `robot_state` 维度 | 假设 14 | `encoder_robot_state_input_proj`: 14→512 |
| ResNet18 layer4 输出 | 假设 feature_map=3×3=9 tokens/cam | `encoder_img_feat_input_proj`: 512→512 |
| 假设 2 个相机 | encoder 共 20 tokens | 1(latent) + 1(robot_state) + 9×2(image) = 20 |

#### 4.5.4 推理时的 tensor 形状流经各层

```
输入: images (B, 2, 3, 96, 96), state (B, 14)
         │
         ▼
backbone(img)  ──→  feature_map (B, 512, 3, 3)  ──→  reshape to (9, B, 512)  ─┐
                                                                               │
encoder_latent_input_proj(torch.zeros(B,32)) ───────────────────────────────→  │
     (32 → 512)                                                                  │
                                                                                  │
encoder_robot_state_input_proj(state) ──────────────────────────────────────→  │ concat
     (14 → 512)                                                                  │  along
                                                                                  │  seq dim
                                                                                  ▼
                                                    encoder_in_tokens: (20, B, 512)
                                                    pos_embed: (20, 1, 512)
                                                         │
                                                         ▼
self.encoder(encoder_in_tokens)  ──────────────────────────────────────────────────→  encoder_out: (20, B, 512)
  └── ACTEncoderLayer × 4  (每层，modeling_act.py 第539行创建 self.self_attn):
        ├── self.self_attn(q,k,v=x)  ─────────────────────────────────────────────────┐
        │     Q/K/V 投影: (20,B,512) @ (512,512) = (20,B,512)   [4层 × 3个投影 = 12次 linear] │
        │     out_proj: (20,B,512) @ (512,512) = (20,B,512)     [4层 × 1个投影 = 4次 linear]  │
        │                                                                                  │
        │     MACs/层 self_attn: 20×512×512×3(QKV) + 20×512×512(out_proj) ≈ 10.5M           │
        │                                                                                  │
        └── self.linear1(x) → self.activation → self.linear2(x)                             │
              512→3200 (linear1) + 3200→512 (linear2) = 512×3200 + 3200×512 = 6.55M/层     │
              FFN MACs/层 ≈ 6.6M × 4层 = 26.2M                                            │
                                                                                          │
decoder_in: (chunk_size=100, B, 512)  ←── 初始为零向量                                      │
     │                                                                                    │
     ▼                                                                                    │
self.decoder(decoder_in, encoder_out)                                                      │
  └── ACTDecoderLayer × 1 (modeling_act.py 第601-602行创建 self.self_attn 和 self.multihead_attn)
        ├── self.self_attn(q,k,v=decoder_in) ─────────────────────────────────────────────┐
        │     Q/K/V: (100,B,512) @ (512,512) × 3 ──→ (100,B,512)                          │
        │     out_proj: (100,B,512) @ (512,512) ──→ (100,B,512)                           │
        │     self_attn MACs ≈ 100×512×512×4 ≈ 104.9M                                      │
        │                                                                                  │
        ├── self.multihead_attn(q=x, k=encoder_out, v=encoder_out) ──────────────────────┤
        │     Q: (100,B,512) @ (512,512) ──→ (100,B,512)                                   │
        │     K: (20,B,512)  @ (512,512) ──→ (20,B,512)   ← 注意 K/V 序列是 encoder 的 20  │
        │     V: (20,B,512)  @ (512,512) ──→ (20,B,512)                                   │
        │     Attention Score: (100,8,64) @ (20,8,64)^T = (100,8,20)                       │
        │     Attention Output: (100,8,20) @ (20,8,64) = (100,8,64) → reshape (100,B,512)  │
        │     cross_attn MACs ≈ 100×512×512 + 100×20×512×2 + 100×512×512 ≈ 41.0M           │
        │                                                                                  │
        └── self.linear1 → activation → self.linear2 ──────────────────────────────────────┤
              FFN MACs ≈ 6.6M                                                           │
                                                                                          │
action_head(decoder_out) ──────────────────────────────────────────────────────────────→  (B, 100, action_dim)
     (100,B,512) @ (512, action_dim) ──→ MACs ≈ 100×512×action_dim ≈ negligible
```

#### 4.5.5 MACs 汇总（推理，单次 forward）

| 模块 | MACs 估算 | 占比 | 备注 |
|------|-----------|------|------|
| Encoder self-attn（4层） | ~42M | 28% | QKV×3 + out_proj，每层~10.5M |
| **Encoder FFN（4层）** | **~26M** | **17%** | 512→3200→512，每层~6.6M |
| Decoder self-attn（1层） | ~105M | **70%** | chunk_size=100 很大，QKV×3 + out_proj |
| Decoder cross-attn（1层） | ~41M | ~27% | Q@K + Q@V，K/V 序列长 20 |
| Decoder FFN（1层） | ~6.6M | 4% | |
| CNN backbone | ~11M | 7% | ResNet18, 96×96 输入 |
| 各类投影层 | ~1M | 1% | latent/state/image 投影 |
| **总计** | **~232M** | 100% | |

> Decoder self-attn 占比最高（70%），因为 `chunk_size=100`，Decoder 的 Q 序列长度是 100，而 Encoder 只有 20。`chunk_size` 越大，Decoder self-attn 代价增长是 **O(N²·d)**（N=chunk_size, d=dim_model），是主要瓶颈。

---

## 5. 第2跳：Python/ATen 绑定层

### 5.1 分段调用链

```
pytorch/torch/nn/modules/linear.py::nn.Linear.forward()
    ↓
pytorch/torch/nn/functional.py::F.linear  ──→  torch._C._nn.linear (C++绑定)
    ↓
pytorch/aten/src/ATen/native/Linear.cpp::at::native::linear()
    ↓
pytorch/aten/src/ATen/native/Linear.cpp::at::addmm()  ← 融合bias+matmul
```

### 5.2 命名含义

| 层级 | 命名 | 含义 |
|------|------|------|
| `nn.Linear` | Linear | 线性变换层，`y = xAᵀ + b` |
| `F.linear` | functional.linear | Python 侧函数接口，实际调用 C++ |
| `at::addmm` | **add** (bias) + **mm** (matrix multiplication) | 融合操作：bias + matmul 一次完成，避免两次 kernel 启动 |
| `at::native::linear` | native = C++ 实现 | ATen 框架的 C++ 原生实现（非 CUDA/MKL 等加速库） |

### 5.3 nn.Linear.forward

**源码路径**：`pytorch/torch/nn/modules/linear.py:130`

```python
class Linear(Module):
    def forward(self, input: Tensor) -> Tensor:
        # self.weight: (out_features, in_features)
        # self.bias:   (out_features,)
        return F.linear(input, self.weight, self.bias)
```

### 5.4 F.linear（C++ 绑定）

**源码路径**：`pytorch/torch/nn/functional.py:2328-2349`

```python
# linear 是通过 _add_docstr 绑定到 torch._C._nn.linear（C++ 函数）
linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: y = xA^T + b.
...
"""
)
```

`F.linear` 内部直接跳转到 ATen C++ 函数，无 Python 中间层。

### 5.5 at::native::linear（ATen C++ 实现）

**源码路径**：`pytorch/aten/src/ATen/native/Linear.cpp:85-143`

```cpp
// pytorch/aten/src/ATen/native/Linear.cpp:85-143
Tensor linear(const Tensor& input, const Tensor& weight,
              const std::optional<Tensor>& bias_opt) {
    const auto input_dim = input.dim();
    const auto weight_dim = weight.dim();
    TORCH_CHECK(input_dim != 0 && weight_dim != 0,
                "both arguments to linear need to be at least 1D");

    // c10::MaybeOwned：偏置是否存在，不存在就造一个"空壳"避免后续判空
    auto bias = bias_opt.has_value()
        ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
        : c10::MaybeOwned<Tensor>::owned(std::in_place);

    if (input.is_mkldnn()) return at::mkldnn_linear(input, weight, *bias);  // MKLDNN 路径（CPU专用）
#if defined(C10_MOBILE)
    if (xnnpack::use_linear(...)) return xnnpack::linear(...);              // 移动端路径
#endif
    // ★ 最常见路径：input 是 2D 且有 bias → 融合 addmm
    if (input_dim == 2 && bias->defined()) {
        // at::addmm(out, input, weight) = input @ weight + out
        // 把 bias 当作 out 传进去，一次完成 bias + matmul，省一次 kernel 启动
        return at::addmm(*bias, input, weight.t());  // ← 第108行：融合计算入口
    }
    // ...
}
```

---

## 6. 第3跳：ATen BLAS 分派层

### 6.1 分段调用链

```
pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp::cpublas_gemm_impl()
    ↓
pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp::gemm_core_()
    ↓ 分派至
┌─────────────────────────────────┐
│ gemm_notrans_()  ← 第105行     │  A、B 均不转置（最常见路径）
│ gemm_transa_()   ← 第171行     │  A 转置、B 不转置
│ gemm_transb_()   ← 第233行     │  A 不转置、B 转置
│ gemm_transab_()  ← 第311行     │  A、B 均转置
└─────────────────────────────────┘
```

### 6.2 命名含义

| 命名 | 含义 |
|------|------|
| `cpublas_gemm_impl` | CPU BLAS 层的 GEMM 实现入口（分派到各模板） |
| `gemm` | **G**eneral **M**atrix **M**atrix multiplication，来自 BLAS 标准，意为"通用矩阵乘" |
| `gemm_core_` | GEMM 核心分派函数，根据 A/B 转置状态选择具体计算路径 |
| `gemm_notrans_` | A、B 均**不转置** |
| `gemm_transa_` | A **转置**（A.T），B 不转置 |
| `gemm_transb_` | A 不转置，B **转置**（B.T） |
| `gemm_transab_` | A、B **均转置** |

> `trans` = transpose（转置），`transa`/`transb` 表示对哪个矩阵转置

### 6.3 cpublas_gemm_impl — 类型分派入口

**源码路径**：`pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:478-497`

**意图**：根据 dtype（float32/float16/bfloat16）做类型分派，调用对应的 GEMM 模板实例。

```cpp
// pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:478-497
void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  _AT_DISPATCH_GEMM_TYPES(type, "cpublas_gemm_impl", [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<opmath_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<opmath_t>(),
            static_cast<scalar_t *>(c), ldc);
      });
}
```

> `_AT_DISPATCH_GEMM_TYPES` 宏展开后根据 dtype 匹配 lambda 参数类型：
> - `kFloat` → `scalar_t=float/double`, `opmath_t=double`, `out_t=float*/double*`
> - `kHalf` → `scalar_t=Half`, `opmath_t=float`（float32累加保证精度）, `out_t=float*`
> - `kBFloat16` → `scalar_t=BFloat16`, `opmath_t=float`, `out_t=float*`

### 6.4 gemm_core_ — 根据 A/B 转置状态选路径

**源码路径**：`pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:442-465`

**意图**：GEMM 计算 C = α·A·B + β·C，A/B 各有两种状态，排列组合成 4 条路径。

```cpp
// pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:442-465
template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_core_(TransposeType transa, TransposeType transb,
                int64_t m, int64_t n, int64_t k,
                opmath_t alpha, const scalar_t* a, int64_t lda,
                const scalar_t* b, int64_t ldb,
                opmath_t beta, out_t* c, int64_t ldc) {
    // ★ 最常见情况：A/B 都不转置 → 直接走 gemm_notrans_（效率最高）
    if (transa == TransposeType::NoTranspose &&
        transb == TransposeType::NoTranspose) {
        return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if (transa != TransposeType::NoTranspose &&
               transb == TransposeType::NoTranspose) {
        // A 需要转置：C = α·A.T·B + β·C
        gemm_transa_(transa, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if (transa == TransposeType::NoTranspose &&
               transb != TransposeType::NoTranspose) {
        // B 需要转置：C = α·A·B.T + β·C
        gemm_transb_(transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        // A/B 都要转置
        gemm_transab_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}
```

---

## 7. 第4跳：GEMM 计算层（矩阵乘法真正执行的地方）

### 7.1 先搞清楚：矩阵乘法在算什么

GEMM 计算的是 **C = α·A·B + β·C**，用大白话说：

- A 是输入，m行 × k列
- B 是权重，k行 × n列
- C 是输出，m行 × n列
- α、β 是缩放系数（nn.Linear 里 α=1，β=1）

**C 里每个格子的计算方式**：
```
C[行i, 列j] = α × (A第i行 · B第j列的内积) + β × C[行i, 列j]原值

              = α × Σ(A[i,l] × B[l,j])  + β × C[i,j]
                    l=0..k-1
```

用图示来理解：
```
A (m×k)    B (k×n)         C (m×n)
┌─────┐   ┌─────┐         ┌─────┐
│行 0 │ × │列 0 │ ─→ C[0,0]│     │
│行 1 │   │列 1 │ ─→ C[1,0]│     │
│ ... │   │ ... │    ...   │ ... │
│行 m │   │列 n │ ─→ C[m,n]│     │
└─────┘   └─────┘         └─────┘

C[i,j] = A第i行 与 B第j列 逐元素相乘后求和
```

**在 nn.Linear 里**：A=input（batch_size × in_features），B=weight（out_features × in_features），做 `input @ weight.T + bias`。

### 7.2 lda/ldb/ldc 是什么（内存步长）

矩阵在内存里是一维数组，`lda`（leading dimension）告诉你从第0列跳到第1列需要跨过几个元素。BLAS 用的是**列优先**存储（竖着放）：

```
3×3矩阵，列优先存储（lda=3）：
        行0  行1  行2
列0 →  [A00, A10, A20,   ← 列0放在内存最前面
列1 →   A01, A11, A21,   ← 列1紧随其后
列2 →   A02, A12, A22]   ← 列2在最后

访问 A[行i, 列j] = 内存中的 a[j*lda + i]
         ↑ lda=3，即每列有3个元素
```

简单记忆：**lda = 矩阵的行数**（正方形 m×m 时 lda=m，矩形 m×k 时 lda=m）。

### 7.3 四条路径：根据 A/B 是否需要转置来选

`gemm_core_` 做的事只有一件：**判断 A、B 要不要转置，然后跳到对应函数**。

| 路径 | A | B | 何时走 |
|------|---|---|--------|
| `gemm_notrans_`（第105行） | 不转置 | 不转置 | **nn.Linear 正向传播，最常见** |
| `gemm_transa_`（第171行） | **转置** | 不转置 | Attention 的某些反向传播 |
| `gemm_transb_`（第233行） | 不转置 | **转置** | 反向传播梯度计算 |
| `gemm_transab_`（第311行） | **转置** | **转置** | 极少见 |

> `gemm_notrans_` 是 float32 的最优路径，支持 SIMD 4路并行。`gemm_transa_` 因为 A 转置后内存不连续，无法 SIMD，但 Half 类型有专用的多线程并行版本（第8节详述）。

### 7.4 gemm_notrans_ — 三重循环 + SIMD（nn.Linear 走的路径）

**源码**：`pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:105`

**三重循环的顺序是 k → n → m**。为什么这个顺序？因为这样可以**把 B[l,j]×α 提到内层循环外面**，避免重复读 B，同时让内层对 A、C 的访问在内存上是连续的（cache 友好）：

```
第1步：C = β × C（先把 C 缩放）

第2步：三层循环累加 C += α × A × B

  for l in 0..k:          ← 遍历 A/B 的共享维度（k维）
    for j in 0..n:        ← 遍历输出的列（n维）
      val = B[l,j] × α   ← 把这个值提出来，下面 m 行都要用它

      // 内层：每次同时处理 4 行（SIMD 向量化）
      for i_block in 0..m/4:
        C[j, i*4+0] += A[i*4+0, l] × val   ← 4行同时算，
        C[j, i*4+1] += A[i*4+1, l] × val      利用 NEON/ARM
        C[j, i*4+2] += A[i*4+2, l] × val      向量寄存器
        C[j, i*4+3] += A[i*4+3, l] × val

      // 尾部：m 不整除 4 的剩余行，逐个处理
      for i in (m/4)*4 .. m:
        C[j, i] += A[i, l] × val
```

```cpp
// pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:105-137
// std::enable_if_t<std::is_same_v<scalar_t, opmath_t>> 表示：
// 只有 float/double 类型走这个重载（计算精度和存储精度相同）
// Half/BFloat16 走另一个重载（第142行），累加时升精度到 float32
template <typename scalar_t, typename opmath_t, typename out_t>
std::enable_if_t<std::is_same_v<scalar_t, opmath_t>, void>
gemm_notrans_(int64_t m, int64_t n, int64_t k, opmath_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    opmath_t beta, out_t* c, int64_t ldc) {

    scale_(m, n, beta, c, ldc);  // C = β × C

    const uint64_t i_m = (uint64_t)m / 4;  // m 行里能整除 4 的部分

    for (const uint64_t l : c10::irange(k)) {
        for (const uint64_t j : c10::irange(n)) {
            opmath_t val = b[l + j * ldb] * alpha;  // 固定 B[l,j]×α，提到内层外
            for (const auto i_i : c10::irange(i_m)) {
                // 4路展开：编译器会把这4行映射到 NEON 向量指令（ARM）
                c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
                c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
                c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
                c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
            }
            uint64_t i = i_m * 4;
            for (; i < (uint64_t)m; i++)
                c[j * ldc + i] += a[i + l * lda] * val;  // 尾部逐个处理
        }
    }
}
```

---

## 8. 第5跳：并行层（把矩阵行分给多个 CPU 核同时算）

### 8.1 为什么要并行，怎么切

C 矩阵的**每一行可以独立计算**（行与行之间没有依赖），所以可以把 m 行平均分给 4 个核：

```
原来（单线程，m=1000行）：
  核0：算第 0~999 行（全部）

改为 4 线程后：
  核0：算第 0~249 行   ──┐
  核1：算第 250~499 行   ├── 四个核同时进行
  核2：算第 500~749 行   │
  核3：算第 750~999 行 ──┘
                          └→ 都算完后，C 矩阵拼在一起
```

### 8.2 调用链（由外到内）

并行层的入口在 **Half 类型专用的 `gemm_transa_`** 里（float32 的 `gemm_notrans_` 靠编译器自动向量化，不用显式并行）：

```
gemm_transa_<Half>()            BlasKernel.cpp:378
  ↓ 调用
parallel_for(0, m, 1, lambda)   Parallel-inl.h:10   ← 判断：要不要并行？
  ↓ 满足条件时调用
invoke_parallel(0, m, 1, f)     ParallelOpenMP.h:17  ← 真正创建线程
  ↓ 通过 GCC 编译指令
#pragma omp parallel             libgomp 在运行时创建线程
  ↓
每个线程：lambda(begin_i, end_i)  ← 各自算自己那段行
```

### 8.3 gemm_transa_（Half 类型专用版）— 触发并行的地方

**源码**：`pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:378`

这个函数是并行层的"触发点"。它先判断是否能走快速路径，否则调用 `parallel_for` 把行分给多个线程：

```cpp
// pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:378-407
template <>
void gemm_transa_<at::Half, float, at::Half>(
    TransposeType transa, int64_t m, int64_t n, int64_t k,
    float alpha, const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta, at::Half *c, int64_t ldc) {

    // 快速路径：n=1 时 GEMM 退化为"矩阵×向量"（GEMV），有专用优化函数
    if (n == 1 && alpha == 1.0) {
        at::native::blas_impl::fp16_gemv_trans(k, m, 1.0f, a, lda, b, 1, beta, c, 1);
        return;
    }

    // 通用路径：把 m 行分给多线程，grain_size=1 允许最细粒度分配
    parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
        // 每个线程只看自己负责的行范围 [begin, end)
        const auto *a_ = a + begin * lda;   // 本线程从 A 的第 begin 行开始
        for (const auto i : c10::irange(begin, end)) {
            const auto *b_ = b;             // 每算新的一行，B 的列指针都从头开始
            for (const auto j : c10::irange(n)) {
                // compute_dot：fp16 乘法 + float32 累加（防止 fp16 精度溢出）
                // fp16 最大值只有 65504，累加时容易溢出，所以用 float32 中间结果
                const auto dot = compute_dot(a_, b_, k);
                b_ += ldb;  // 移到 B 的下一列
                if (beta == 0)
                    c[j*ldc+i] = alpha * dot;
                else
                    c[j*ldc+i] = beta * c[j*ldc+i] + alpha * dot;
            }
            a_ += lda;  // 移到 A 的下一行（内存连续，cache 友好）
        }
    });
}
```

### 8.4 parallel_for — 决策：要不要真的开线程？

**源码**：`pytorch/aten/src/ATen/Parallel-inl.h:10`

`parallel_for` **不一定真的并行**——它先做一个三步判断，任何一步不满足就直接串行执行，省去线程切换开销：

```
判断流程：

① 任务量够吗？     end - begin > grain_size（grain_size=1 几乎总满足）
② 至少2行吗？      end - begin > 1（1行不值得分线程）
③ 不在嵌套并行里？  !in_parallel_region()（防止"4线程×4线程=16线程"爆炸）
④ 机器多核吗？     get_num_threads() > 1（单核没必要并行）

全部满足 → 调用 invoke_parallel 开多线程
有任意一条不满足 → 直接串行 f(begin, end)
```

树莓派5实际情况：4核，m 通常是几百到几千，全部满足条件 → 走并行路径。

```cpp
// pytorch/aten/src/ATen/Parallel-inl.h:10-43
template <class F>
inline void parallel_for(
    const int64_t begin, const int64_t end,
    const int64_t grain_size, const F& f) {
  if (begin >= end) return;

#ifdef INTRA_OP_PARALLEL
  at::internal::lazy_init_num_threads();  // 第一次调用时探测 CPU 核数（只探测一次）

  const auto numiter = end - begin;
  const bool use_parallel =
      (numiter > grain_size &&
       numiter > 1 &&
       !at::in_parallel_region() &&  // ParallelGuard 在并行区里会返回 true，阻止嵌套
       at::get_num_threads() > 1);

  if (!use_parallel) {
    f(begin, end);  // 串行直接跑
    return;
  }

  internal::invoke_parallel(begin, end, grain_size,
      [&](int64_t begin, int64_t end) {
        c10::ParallelGuard guard(true);  // 标记"已在并行区"：lambda 里若再调 parallel_for 会串行
        f(begin, end);
      });
#else
  f(begin, end);  // 未开启并行编译时：无条件串行
#endif
}
```

### 8.5 invoke_parallel — 创建线程，分配行范围，等待汇合

**源码**：`pytorch/aten/src/ATen/ParallelOpenMP.h:17`

`#pragma omp parallel` 是 GCC 编译指令：编译器看到它，会在这里插入线程池代码（libgomp 的 `pthread_create()`）。**大括号里的代码被所有线程同时执行**，但每个线程有自己的编号 `tid`，所以每个线程能算出自己负责的行范围，互不重叠：

```
m=1000，4核，grain_size=1

每线程行数 = ceil(1000 / 4) = 250

线程0 (tid=0)：begin=0,   end=250   ─┐
线程1 (tid=1)：begin=250, end=500    ├── 同时执行，各自算各自那段
线程2 (tid=2)：begin=500, end=750    │
线程3 (tid=3)：begin=750, end=1000  ─┘
                                      ↓
                              #pragma omp parallel 结束处有隐式屏障
                              所有线程到达屏障后才继续，保证 C 写完
```

小任务自动缩减线程数：如果 m=2，`min(4线程, ceil(2/1)=2)=2`，只开 2 个线程，不浪费。

```cpp
// pytorch/aten/src/ATen/ParallelOpenMP.h:17-52
template <typename F>
inline void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

#pragma omp parallel  // GCC 编译指令：在此 fork 出线程池，{ } 里的代码被所有线程执行
  {
    int64_t num_threads = omp_get_num_threads();  // 进入并行区域后才能查到实际线程数
    if (grain_size > 0) {
      // 小任务优化：如果行数 < 线程数，就减少线程数，避免浪费
      num_threads = std::min(num_threads, divup(end - begin, grain_size));
    }

    int64_t tid        = omp_get_thread_num();              // 本线程编号：0/1/2/3
    int64_t chunk      = divup(end - begin, num_threads);   // 每线程分到的行数（向上取整）
    int64_t begin_tid  = begin + tid * chunk;               // 本线程起始行

    if (begin_tid < end) {  // 越界保护（最后一个线程可能行数不足 chunk）
      try {
        f(begin_tid, std::min(end, begin_tid + chunk));  // 执行 GEMM lambda
      } catch (...) {
        // 多线程同时抛异常时只记录第一个（test_and_set 是原子操作）
        if (!err_flag.test_and_set())
          eptr = std::current_exception();
      }
    }
  }
  // ↑ 隐式屏障：所有线程算完后才继续往下走，保证 C 矩阵数据完整

  if (eptr) std::rethrow_exception(eptr);  // 统一抛出异常给调用方
}
```

---

## 附录：关键源码索引

### A.1 lerobot Python 层

| 函数/类 | 文件路径 | 行号 |
|---------|---------|------|
| `record_loop` | `lerobot/src/lerobot/record.py` | 第367行（inference_start_t）、第372行（predict_action调用） |
| `predict_action` | `lerobot/src/lerobot/utils/control_utils.py` | 第125行定义 |
| `ACTPolicy.select_action` | `lerobot/src/lerobot/policies/act/modeling_act.py` | 第99行定义 |
| `ACTPolicy.predict_action_chunk` | `lerobot/src/lerobot/policies/act/modeling_act.py` | 第124行定义 |
| `ACT.forward` | `lerobot/src/lerobot/policies/act/modeling_act.py` | 第377行定义 |
| `nn.MultiheadAttention`（Encoder层） | `lerobot/src/lerobot/policies/act/modeling_act.py` | 第539行 |
| `nn.MultiheadAttention`（Decoder层） | `lerobot/src/lerobot/policies/act/modeling_act.py` | 第601-602行 |

### A.2 PyTorch Python 层

| 函数/类 | 文件路径 | 行号 |
|---------|---------|------|
| `nn.Linear.forward` | `pytorch/torch/nn/modules/linear.py` | 第130行 |
| `F.linear`（C++绑定） | `pytorch/torch/nn/functional.py` | 第2328行（通过 `_add_docstr(torch._C._nn.linear, ...)` 绑定） |

### A.3 ATen C++ 层

| 函数 | 文件路径 | 行号 |
|------|---------|------|
| `at::native::linear` | `pytorch/aten/src/ATen/native/Linear.cpp` | 第85行定义、第108行（融合addmm路径） |
| `at::addmm`（融合调用） | `pytorch/aten/src/ATen/native/Linear.cpp` | 第68行（_flatten_nd_linear内）、第108行（linear内） |
| `cpublas_gemm_impl` | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第478行 |
| `gemm_core_` | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第443行 |
| `gemm_notrans_`（float/double） | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第105行 |
| `gemm_notrans_`（Half/BF16） | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第142行 |
| `gemm_transa_`（通用模板） | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第171行 |
| `gemm_transa_`（Half专用，含parallel_for） | `pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp` | 第378行 |

### A.4 ATen 并行后端

| 函数/宏 | 文件路径 | 行号 |
|---------|---------|------|
| ATen 并行后端选择 | `pytorch/aten/src/ATen/Parallel.h` | 第152-156行 |
| `parallel_for` | `pytorch/aten/src/ATen/Parallel-inl.h` | 第10行 |
| `invoke_parallel`（OpenMP） | `pytorch/aten/src/ATen/ParallelOpenMP.h` | 第17行（第25行 `#pragma omp parallel`） |

# PyTorch Tensor → GEMM → OpenMP 多线程 核心调用链

> 基于 PyTorch 2.7.1+CPU + ARM64 环境

---

## 一、整体架构：四层调用

```
┌─────────────────────────────────────────────────────────┐
│ Python 层                                               │
│ record.py → predict_action() → ACTPolicy.select_action()│
│            → ACT.forward() → nn.Linear.forward()        │
└────────────────────────┬────────────────────────────────┘
                         │ F.linear() = torch._C._nn.linear
                         ▼
┌─────────────────────────────────────────────────────────┐
│ ATen C++ 层                                             │
│ at::native::linear() → at::addmm()                      │
│     → cpublas_gemm_impl() → gemm_core_()                │
│         → gemm_notrans_() / gemm_transa_()             │
└────────────────────────┬────────────────────────────────┘
                         │ parallel_for(0, m, 1, lambda)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OpenMP 层                                               │
│ invoke_parallel() → #pragma omp parallel                │
│     → libgomp (pthread_create) → 多线程并行              │
└─────────────────────────────────────────────────────────┘
```

**核心瓶颈**：Decoder self-attn 占比 70%（chunk_size=100 导致 Q 序列长 100）

---

## 二、Python 层：ACT 推理流程

### 2.1 入口：`record_loop()` — [lerobot/src/lerobot/record.py:363](lerobot/src/lerobot/record.py#L363)

```python
action_values = predict_action(
    observation=observation_frame,
    policy=policy,
    device=get_safe_torch_device(policy.config.device),
    preprocessor=preprocessor,
    postprocessor=postprocessor,
)
```

### 2.2 推理触发：`ACTPolicy.select_action()` — [lerobot/src/lerobot/policies/act/modeling_act.py:99](lerobot/src/lerobot/policies/act/modeling_act.py#L99)

```python
@torch.no_grad()
def select_action(self, batch):
    # 动作队列为空时触发推理（不是每帧都推理！）
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
        self._action_queue.extend(actions.transpose(0, 1))  # (B,S,D) → (S,B,D)
    return self._action_queue.popleft()  # 每帧弹出一帧
```

### 2.3 ACT forward 核心：`modeling_act.py:377`

```python
def forward(self, batch):
    # 1. 图像 → CNN特征 → (B, 512, 3, 3) → reshape(9, B, 512)
    image_features = self.cnn(batch["observation.images"])

    # 2. Transformer Encoder (推理时不走)
    #    训练时: VAE隐变量 + robot_state + image_features → encoder_out
    #    推理时: encoder_in 全零，直接跳过

    # 3. Transformer Decoder: 隐变量 + 当前状态 → 动作序列
    decoder_output = self.decoder(decoder_in)   # 触发多次 linear()
    actions = self.action_head(decoder_output)  # 最后一个 linear()
    return actions
```

---

## 三、ATen C++ 层：线性层到 GEMM

### 3.1 `nn.Linear.forward` — [torch/nn/modules/linear.py:124](torch/nn/modules/linear.py#L124)

```python
class Linear(Module):
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
        # weight: (out_features, in_features)
        # bias:   (out_features,)
```

### 3.2 `F.linear` — [torch/nn/functional.py:2309](torch/nn/functional.py#L2309)

```python
linear(input, weight, bias=None) -> Tensor
# 实际调用: torch._C._nn.linear (C++ 绑定入口)
```

### 3.3 `at::native::linear()` — [pytorch/aten/src/ATen/native/Linear.cpp:85](pytorch/aten/src/ATen/native/Linear.cpp#L85)

```cpp
// input: (*, in_features), weight: (out_features, in_features), bias: (out_features,)
Tensor linear(const Tensor& input, const Tensor& weight,
              const std::optional<Tensor>& bias_opt) {
    // ★ 最常见路径：input 2D + 有 bias → 融合 addmm（省一次 kernel 启动）
    if (input_dim == 2 && bias->defined()) {
        return at::addmm(*bias, input, weight.t());  // y = input @ weight.T + bias
    }
}
```

### 3.4 `at::addmm()` — [pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp:478](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L478)

```cpp
// addmm(out, input, mat2) = input @ mat2 + out
// 把 bias 当作 out 传进去，一次完成 bias + matmul
void cpublas_gemm_impl(...) {
    _AT_DISPATCH_GEMM_TYPES(type, ..., [&] {
        gemm_core_(transa, transb, m, n, k, ...);  // 根据 dtype 派发
    });
}
```

---

## 四、GEMM 并行调用链（按实际调用顺序，从 cpublas_gemm_impl 开始一步步走）

### 调用总览

```
cpublas_gemm_impl()          ← ① 入口：dtype 派发
    │
    ▼
gemm_core_()                 ← ② 路由：A/B 是否转置？走哪条路径？
    │
    ├── transa=No, transb=No → gemm_notrans_()   ← ③a 最常见路径（串行三重循环）
    ├── transa=Yes,transb=No → gemm_transa_()    ← ③b A 转置路径
    ├── transa=No, transb=Yes→ gemm_transb_()    ← ③c B 转置路径
    └── transa=Yes,transb=Yes→ gemm_transab_()   ← ③d 都转置
```

**ACT 推理走哪条？** `at::addmm(bias, input, weight.t())` 中 `weight.t()` 已经在外部转置好了，
所以传给 GEMM 时 A 和 B 都不需要转置，**走 ③a gemm_notrans_ 路径**。

> 但 Half/BFloat16 类型的 `gemm_transa_` 内部用了 `parallel_for` 做并行，
> float32 类型的 `gemm_notrans_` 是纯串行三重循环。下面逐层展开。

---

### ① 入口：cpublas_gemm_impl — [BlasKernel.cpp:478-497](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L478)

```cpp
// 传入参数：
//   type = dtype（kFloat / kHalf / kBFloat16）
//   transa, transb = A/B 是否需要转置
//   m, n, k = 矩阵维度（A: m×k, B: k×n, C: m×n）
//   alpha, beta = 系数（C = α·A·B + β·C）
//   a, b, c = 矩阵数据指针
//   lda, ldb, ldc = leading dimension（内存中每行的跨度）
void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  // _AT_DISPATCH_GEMM_TYPES：根据 type 派发到具体 C++ 类型
  //   kFloat    → scalar_t = float,    opmath_t = float
  //   kHalf     → scalar_t = Half,     opmath_t = float（float32累加保精度）
  //   kBFloat16 → scalar_t = BFloat16, opmath_t = float
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

**做了什么**：根据 dtype 选择 C++ 模板类型，然后调 `gemm_core_()`。

**ACT 中的典型 m/n/k：**

| 场景 | m | n | k | 含义 |
|-----|---|---|---|-----|
| Decoder Q 投影 | 100 | 512 | 512 | chunk_size=100 × dim_model |
| Encoder K/V 投影 | 20 | 512 | 512 | encoder 只有 20 tokens |
| FFN linear1 | 20或100 | 3200 | 512 | 中间层放大 |

---

### ② 路由：gemm_core_ — [BlasKernel.cpp:443-465](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L443)

```cpp
template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
  // ★ 第一个 if：A/B 都不转置 → 最常见，直接算
  if (transa == TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  // 第二个 if：A 需要转置，B 不转置
  else if (transa != TransposeType::NoTranspose &&
           transb == TransposeType::NoTranspose) {
    gemm_transa_(transa, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  // 第三个 if：A 不转置，B 需要转置
  else if (transa == TransposeType::NoTranspose &&
           transb != TransposeType::NoTranspose) {
    gemm_transb_(transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  // else：A/B 都需要转置
  else {
    gemm_transab_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
```

**做了什么**：看 A 和 B 需不需要转置，选一条计算路径。

**ACT 推理时**：`addmm(bias, input, weight.t())` 里 `weight.t()` 已在外面完成，
传进来时 transa=No, transb=No，**走第一个 if → gemm_notrans_**。

---

### ③a 路径一：gemm_notrans_（A/B 都不转置） — [BlasKernel.cpp:105-137](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L105)

**这是 ACT 推理最常走的路径。**

> **并行在哪里？** 这取决于编译时是否链接了外部 BLAS 库（如 OpenBLAS）：
> - **有外部 BLAS**（`AT_BUILD_WITH_BLAS()=ON`）：`cpublas::gemm` 直接调 `sgemm_()`，
>   外部 BLAS 库内部自带多线程，**不走** `gemm_notrans_`
> - **无外部 BLAS**（树莓派默认编译）：走 `gemm_stub` → `gemm_notrans_`，**纯串行三重循环**
>
> 详见 [CPUBlas.cpp:206-249](pytorch/aten/src/ATen/native/CPUBlas.cpp#L206)：
> ```cpp
> // cpublas::gemm (float32 版本)
> void gemm(..., float *c, int64_t ldc) {
> #if AT_BUILD_WITH_BLAS()
>   if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
>     sgemm_(...);   // ← 外部 BLAS（OpenBLAS 等），自带多线程
>     return;
>   }
> #endif
>   gemm_stub(...);  // ← 走 cpublas_gemm_impl → gemm_notrans_（串行）
> }
> ```


```cpp
// float32 版本（scalar_t == opmath_t，即 float → float）
// A: (m, k), B: (k, n), C: (m, n)
// 计算 C = β*C + α*A*B
template <typename scalar_t, typename opmath_t, typename out_t>
std::enable_if_t<std::is_same_v<scalar_t, opmath_t>, void>
gemm_notrans_(
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha, const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    opmath_t beta, out_t* c, int64_t ldc) {
  // step 1: C *= beta（β=0 时全部清零）
  scale_(m, n, beta, c, ldc);

  // step 2: C += alpha * (A @ B)，三重循环
  const uint64_t unsigned_m = m;
  const uint64_t i_m = unsigned_m / 4;  // m 按4整除，用于 SIMD 展开

  for (const uint64_t l : c10::irange(k)) {       // 外层：k 维（累加维度）
    for (const uint64_t j : c10::irange(n)) {     // 中层：n 维（输出列）
      opmath_t val = b[l + j * ldb] * alpha;      // 取 B[l,j] × α
      // 内层：m 维（输出行），4路 SIMD 展开
      for (const auto i_i : c10::irange(i_m)) {
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      // 尾部：m%4 不足4个的单独处理
      uint64_t i = i_m * 4;
      for (; i < unsigned_m; i++)
        c[j * ldc + i] += a[i + l * lda] * val;
    }
  }
}
```

**三重循环在算什么（以 Decoder Q 投影为例）：**

```
A = (100, 512)  输入（100个时间步，每个512维）
B = (512, 512)  权重
C = (100, 512)  输出

外层 l=0..511:  遍历累加维度 k
  中层 j=0..511:  遍历输出列 n
    内层 i=0..99:  遍历输出行 m（4路SIMD）

C[i,j] += A[i,l] × B[l,j] × α

整个循环跑完：100 × 512 × 512 = 26,214,400 次乘加
```

**float16/bfloat16 版本** — [BlasKernel.cpp:142-168](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L142)：

```cpp
// Half/BFloat16 版本（scalar_t ≠ opmath_t，需要类型转换保精度）
template <typename scalar_t, typename opmath_t, typename out_t>
std::enable_if_t<!std::is_same_v<scalar_t, opmath_t>, void>
gemm_notrans_(int64_t m, int64_t n, int64_t k,
    opmath_t alpha, const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb, opmath_t beta, out_t* c, int64_t ldc) {
  // 也是双层循环，用 opmath_t (float32) 累加保精度
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[j * ldb + l]);
      });
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}
```

> **为什么这两个版本都没有 parallel_for？**
> 因为 `gemm_notrans_` 是被 `gemm_core_` **直接调用**的，
> 并行不在这里做——外层的 `addmm` 在调用 `cpublas_gemm_impl` 之前/之后处理并行。
> 而 Half/BFloat16 类型的 `gemm_transa_` **内部自己调 parallel_for**，因为那种场景需要并行加速。

---

### ③b 路径二：gemm_transa_（A 需要转置） — [BlasKernel.cpp:170-196](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L170)

```cpp
// float32 版本：C = α·A.T·B + β·C
// 串行双层循环
template <typename scalar_t, typename opmath_t, typename out_t>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    out_t *c, int64_t ldc) {
  const scalar_t *a_ = a;                    // a_ 指向 A 的行起始
  for (const auto i : c10::irange(m)) {      // 遍历 C 的每一行
    const scalar_t *b_ = b;                  // b_ 指向 B 的列起始（每行重置）
    for (const auto j : c10::irange(n)) {    // 遍历 C 的每一列
      // dot = Σ_{l=0}^{k-1} A.T[i,l] × B[l,j]
      // A.T[i,l] = a_[l]（转置后直接按列读）
      // B[l,j]   = b_[l]（b_ 指向 B 第 j 列的第 l 个元素）
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        return static_cast<opmath_t>(transa == TransposeType::ConjTranspose
            ? conj_impl(a_[l]) : a_[l])
            * static_cast<opmath_t>(b_[l]);
      });
      b_ += ldb;  // B 移到下一列
      if (beta == opmath_t(0)) {
        c[j*ldc+i] = alpha*dot;
      } else {
        c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
      }
    }
    a_ += lda;  // A 移到下一行
  }
}
```

---

### ③b-special 路径二特例：Half/BFloat16 的 gemm_transa_（带 parallel_for）

**关键：这是唯一一个内部自己用 parallel_for 做并行的 GEMM 函数。**

Half 类型版本 — [BlasKernel.cpp:375-407](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L375)：

```cpp
// Half 类型专用版
template <>
void gemm_transa_<at::Half, float, at::Half>(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc) {
  // n==1 且 alpha==1.0 时走 fp16_gemv 优化路径（单列向量，矩阵-向量乘）
  if (n == 1 && alpha == 1.0) {
    at::native::blas_impl::fp16_gemv_trans(k, m, 1.0, a, lda, b, 1, beta, c, 1);
    return;
  }
  // ★ 通用路径：parallel_for 把 m 行分给多线程
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    // 每个线程收到 [begin, end) 范围内的行
    // 例如 m=100，4线程 → Thread0: [0,25), Thread1: [25,50), ...
    const auto *a_ = a + begin * lda;       // 本线程 A 的起始行
    for (const auto i : c10::irange(begin, end)) {
      const auto *b_ = b;                    // B 的列指针（每行重置）
      for (const auto j : c10::irange(n)) {
        // compute_dot：用 fp16_dot_with_fp32_arith 向量化累加
        const auto dot = compute_dot(a_, b_, k);
        b_ += ldb;
        if (beta == 0) {
          c[j*ldc+i] = alpha*dot;
        } else {
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
        }
      }
      a_ += lda;  // A 移到下一行
    }
  });
}
```

**parallel_for 做了什么**：把 m 行切成 N 段，每线程算一段。
上面 lambda 里的 `for i in [begin, end)` 就是每个线程要算的行范围。

BFloat16 类型版本完全一样的结构 — [BlasKernel.cpp:413-439](pytorch/aten/src/ATen/native/cpu/BlasKernel.cpp#L413)：

```cpp
template <>
void gemm_transa_<at::BFloat16, float, at::BFloat16>(...) {
  // 同样调 parallel_for(0, m, 1, lambda)
  // 同样内部用 compute_dot（bf16_dot_with_fp32_arith）
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    // 与 Half 版完全相同的逻辑
  });
}
```

---

### ④ parallel_for 的内部 — [Parallel-inl.h:10](pytorch/aten/src/ATen/Parallel-inl.h#L10)

```cpp
template <typename F>
inline void parallel_for(
    int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  if (begin >= end) return;

#ifdef INTRA_OP_PARALLEL
  at::internal::lazy_init_num_threads();
  const auto numiter = end - begin;
  // 判断是否值得并行：
  //   迭代数 > grain_size 且 > 1 且不在并行区域内且有多线程
  const bool use_parallel =
      (numiter > grain_size && numiter > 1 &&
       !at::in_parallel_region() && at::get_num_threads() > 1);
  if (!use_parallel) {
    // 任务太小，串行执行
    f(begin, end);
    return;
  }
  // ★ 值得并行：调 invoke_parallel → OpenMP 创建多线程
  internal::invoke_parallel(
      begin, end, grain_size, [&](int64_t begin, int64_t end) {
        f(begin, end);  // f 就是 BlasKernel.cpp 传进来的 lambda
      });
#else
  f(begin, end);  // 未启用并行，串行
#endif
}
```

**做了什么**：判断任务够不够大，够大就调 `invoke_parallel` 创建多线程。

---

### ⑤ invoke_parallel — OpenMP 线程创建 — [ParallelOpenMP.h:25](pytorch/aten/src/ATen/ParallelOpenMP.h#L25)

```cpp
#ifdef _OPENMP  // GCC 编译时看到 -fopenmp 就定义此宏
namespace at::internal {
template <typename F>
inline void invoke_parallel(
    int64_t begin, int64_t end, int64_t grain_size, const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

  #pragma omp parallel  // ← GCC 生成 libgomp::pthread_create()，fork 出 N 个线程
  {
    int64_t num_threads = omp_get_num_threads();  // N = CPU核数（树莓派5 = 4）
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();              // 本线程编号：0, 1, 2, 3
    int64_t chunk_size = divup((end - begin), num_threads);  // 每线程行数
    int64_t begin_tid = begin + tid * chunk_size;    // 本线程起始行
    int64_t end_tid = std::min(end, begin_tid + chunk_size); // 本线程结束行

    if (begin_tid < end) {
      f(begin_tid, end_tid);  // ★ 执行该线程被分到的行段
    }
  }  // ← 隐式屏障：所有线程在此等齐

  if (eptr) std::rethrow_exception(eptr);
}
} // namespace
#endif
```

**执行流程（m=100, 4线程）：**

```
主线程
  └── #pragma omp parallel
          ├── Thread 0: tid=0, f(0, 25)    ← lambda 内 for i in [0,25)
          ├── Thread 1: tid=1, f(25, 50)   ← lambda 内 for i in [25,50)
          ├── Thread 2: tid=2, f(50, 75)   ← lambda 内 for i in [50,75)
          └── Thread 3: tid=3, f(75, 100)  ← lambda 内 for i in [75,100)
  └── 隐式屏障同步 → 所有线程算完 → 主线程继续
```

---

### 调用链总结图

```
cpublas_gemm_impl(dtype, m, n, k, ...)        ← ① dtype 派发
    │
    ▼
gemm_core_(transa, transb, ...)                ← ② 看转置状态选路径
    │
    ├── transa=No, transb=No
    │   └── gemm_notrans_()                    ← ③a 串行三重循环（ACT 推理走这里）
    │       └── for l: for j: for i: C += A*B  （4路 SIMD 展开）
    │
    └── transa=Yes, transb=No  (Half/BF16 类型)
        └── gemm_transa_()                      ← ③b 带 parallel_for 并行
            └── parallel_for(0, m, 1, lambda)   ← ④ 并行入口
                └── invoke_parallel()            ← ⑤ OpenMP 线程创建
                    └── #pragma omp parallel
                        ├── Thread 0: lambda(0, m/4)
                        ├── Thread 1: lambda(m/4, m/2)
                        ├── Thread 2: lambda(m/2, 3m/4)
                        └── Thread 3: lambda(3m/4, m)
                              │
                              └── 每线程内: for i: for j: compute_dot()
```

---

## 五、关键数值汇总

| 配置 | 值 | 影响 |
|-----|-----|-----|
| `chunk_size=100` | Decoder Q 序列长度 | self_attn 瓶颈：O(N²·d) |
| `dim_model=512` | Q/K/V 每头维度 | 每头 64 维 × 8 heads |
| `dim_feedforward=3200` | FFN 中间维度 | linear1: 512→3200, linear2: 3200→512 |
| `num_threads=4` | OpenMP 线程数 | ARM 4核并行 |

**MACs 估算（推理单次 forward）**：
- Decoder self-attn: ~105M (70%)
- Encoder self-attn: ~42M (28%)
- 其他（CNN/投影）: ~10M (7%)
- **总计: ~232M**

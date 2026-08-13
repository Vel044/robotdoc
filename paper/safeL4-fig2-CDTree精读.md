# 精读笔记：safeL4 图 2 CDTree 派生/撤销流程详解

> 对应论文 §4.2.1、图 2（`figures/fig2-cdtree.png`）与清单 1。原文见 [safeL4-Rethinking-Safe-Language-Based-Single-Address-Space-Microkernels.md](safeL4-Rethinking-Safe-Language-Based-Single-Address-Space-Microkernels.md)。

![图 2：capability 系统的派生与撤销](figures/fig2-cdtree.png)

结合清单 1 的结构体定义，把图 2 的三个状态拆开看更清楚。先回顾涉及的三个结构体：

```rust
pub(crate) struct RawCap {
    pub(crate) object: Arc<RwLock<KObj>>,      // 真正的内核对象，强引用
}
pub(crate) struct CdtNode {
    pub(crate) cap: Arc<RawCap>,               // 本节点强引用 RawCap
    pub(crate) child: Vec<Arc<Mutex<CdtNode>>>, // 强引用所有子节点
}
pub struct Cap {
    pub(crate) raw_cap: Weak<RawCap>,          // 只弱引用 RawCap
    pub(crate) rights: Rights,
    pub(crate) cdt_node: Weak<Mutex<CdtNode>>,  // 只弱引用 CdtNode
}
```

关键设计：**树往下（父→子）是强引用（Arc），保护域手里的 Cap 对树和内核对象都只是弱引用（Weak）**。谁能强引用，谁就能决定对象的生死；域自己拿不到强引用，所以域没法阻止撤销。

## 状态 1（初始）

域 A 已经有一个根 CdtNode，它的 `cap` 字段强引用着某个 RawCap（进而强引用着 KObj）。域 A 手里拿着一个 Cap，其 `raw_cap`/`cdt_node` 分别弱引用这个 RawCap 和这个根 CdtNode。

## 状态 1 → 状态 2（域 A 给域 B 派生）

1. capability 系统 `clone` 一份 RawCap——注意这里 clone 的是 `Arc<RwLock<KObj>>`，clone 后底层 KObj 只有一份，只是强引用计数加一（新的 RawCap 结构体和原 RawCap 共享同一个 KObj）。
2. 用这份新克隆的 RawCap 建一个新的 CdtNode，把它 push 进父节点（域 A 的根 CdtNode）的 `child: Vec<Arc<Mutex<CdtNode>>>` 里——这一步让父节点对子节点持有强引用，是撤销级联的关键伏笔。
3. 用这个新 CdtNode 构造一个新 Cap（内部两个字段都是从新 CdtNode/新 RawCap 降级来的 Weak），返回给域 A。
4. 域 A 把这个新 Cap 转手交给域 B。此时域 B 手里的 Cap 跟树/内核对象之间，全程只有 Weak 关系。

## 状态 2 → 状态 3（域 A 撤销）

1. 域 A 通过自己保留的信息定位到这个派生出来的 CdtNode（在父节点的 `child` 列表里）。
2. 系统把这个 CdtNode 从父节点的 `child` Vec 中移除（`Vec::remove` 或类似操作），父节点对它的那个 `Arc<Mutex<CdtNode>>` 强引用就没了。
3. 如果没有别的地方还强引用这个 CdtNode，Rust 的所有权规则会自动 drop 它；drop CdtNode 时，它内部的 `cap: Arc<RawCap>` 这个强引用也随之消失，RawCap 的强引用计数减一。
4. 如果这也是 RawCap 最后一个强引用，RawCap（以及它内部 `Arc<RwLock<KObj>>` 的这一份强引用）也被释放。
5. 域 B 手里的 Cap 全程没有变化，但它的 `raw_cap: Weak<RawCap>` 现在指向的东西已经没有强引用存活了——域 B 再调用内核对象方法、触发 `upgrade()` 时会拿到 `None`，调用失败，达成撤销效果。

## 级联撤销是"白嫖"来的

假如域 B 之前又把自己的 Cap 派生给了域 C（即状态 2 的 CdtNode 下面还挂着一个孙子 CdtNode），那么撤销时 drop 父节点，Rust 会自动依次 drop 它 `child` Vec 里的所有子节点——整棵子树跟着被清空，域 C 的 Cap 也会同时失效。作者不需要像 seL4 那样手写"遍历子孙、逐个撤销"的逻辑，白拿 Rust 所有权系统的递归 drop 语义就实现了级联撤销。

这也是这套设计相比 seL4 原生双向链表 CDT 的一个巧妙之处：seL4 里撤销一个 cap 要靠遍历 CDT 链表、借助 badge/深度信息识别哪些节点算是"后代"再逐个删除；这里则是把"是否还有强引用"这件事直接交给编译器和运行时的引用计数机制去做。

## 两把锁分别管什么

- `CdtNode` 外面包了一层 `Mutex`：多个域可能并发对同一棵树做派生/撤销（增删 `child`），需要互斥防止树结构本身被并发破坏。
- `KObj` 外面包了一层 `RwLock`：内核对象本身允许多个读者并发访问、但写操作要独占。这跟树结构的并发保护是两件独立的事。

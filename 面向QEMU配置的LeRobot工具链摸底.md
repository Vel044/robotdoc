# 面向 QEMU / 新 kernel 的 LeRobot 链路摸底索引

这份文档已经拆分到目录：

```text
robotdoc/新内核_QEMU_设备树_LeRobot链路摸底/
```

拆分原因：这个问题本质是一条从 boot 到 LeRobot 的链。继续放在一个 Markdown 里会把启动链、设备树、RP1/USB、Linux ABI、QEMU、工具链、Python 包证据混在一起。现在主线按“新 kernel 要逐层打通什么”拆开。

## 阅读顺序

| 顺序 | 文件 | 回答的问题 |
| --- | --- | --- |
| 1 | `新内核_QEMU_设备树_LeRobot链路摸底/01_目标与总链路.md` | 我到底要做什么，最终怎么验收。 |
| 2 | `新内核_QEMU_设备树_LeRobot链路摸底/02_RaspberryPi5启动链.md` | Pi 5 怎么加载 kernel、dtb、cmdline、initramfs/rootfs。 |
| 3 | `新内核_QEMU_设备树_LeRobot链路摸底/03_BCM2712与设备树.md` | 新 kernel 怎么认识 BCM2712、设备树和关键硬件节点。 |
| 4 | `新内核_QEMU_设备树_LeRobot链路摸底/04_RP1_USB摄像头舵机链路.md` | LeRobot 的摄像头和舵机怎么从 RP1/USB 枚举成设备节点。 |
| 5 | `新内核_QEMU_设备树_LeRobot链路摸底/05_新Kernel必须提供的Linux接口.md` | LeRobot 用户态会压到哪些 Linux ABI/syscall/设备接口。 |
| 6 | `新内核_QEMU_设备树_LeRobot链路摸底/06_QEMU第一阶段验证方案.md` | Mac 上 QEMU 第一阶段先测什么、mock 什么。 |
| 7 | `新内核_QEMU_设备树_LeRobot链路摸底/07_工具链版本与构建命令.md` | 构建 kernel 需要哪些工具、版本和命令。 |
| 8 | `新内核_QEMU_设备树_LeRobot链路摸底/08_树莓派真机快照逐文件解析.md` | 从树莓派复制来的 boot、device-tree、hardware、modules 文件逐个是什么意思。 |

## 一句话主线

```text
新 kernel
  -> 启动链
  -> 设备树
  -> BCM2712 / RP1 / USB
  -> Linux 用户态 ABI
  -> LeRobot record + ACT 负载
  -> QEMU 第一阶段验证
```

## 完整证据附录

完整调用链、所有 import、runtime trace、Python 包版本、树莓派 SSH 核对结果保留在：

```text
robotdoc/面向QEMU配置的LeRobot工具链摸底_附录.md
```

当前树莓派 5 的最小真机文件快照保留在：

```text
robotdoc/新内核_QEMU_设备树_LeRobot链路摸底/树莓派5真机文件快照/2026-05-18/
```

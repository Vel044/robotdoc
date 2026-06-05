# 文献管理说明

本文论文使用 `biblatex + biber` 管理参考文献，主文献库文件为：

- `robotdoc/论文/references.bib`

## 当前已录入的核心条目

- `openvla`
- `act`
- `diffusionpolicy`
- `lerobot`
- `rt1`
- `rt2`
- `ros2`
- `ros2tracing`
- `onnxruntime`
- `tflite`

## 本地论文资料目录

引言相关论文 PDF 统一放在：

- `robotdoc/论文/1_引言/papers/03_ACT与VLA模型/`

当前已存在的对应文件：

- `02_OpenVLA_开源视觉语言动作模型.pdf`
- `04_ACT_基于低成本硬件学习精细双臂操作.pdf`
- `05_Diffusion_Policy_基于动作扩散的视觉运动策略学习.pdf`

## 使用规则

1. 在正文中引用文献时，统一使用 `\cite{key}`，其中 `key` 必须与 `references.bib` 中条目名一致。
2. 新增文献时，先把 Bib 条目写入 `references.bib`，再决定是否把 PDF 放入对应主题目录。
3. 经验上按“主题目录 + 编号 + 中文标题.pdf”保存 PDF，便于人工查找；Bib key 保持英文、简短、稳定。
4. 如果某篇文献只在正文中出现一次，也仍然优先加入 `references.bib`，不要直接手写到 LaTeX 正文。

## 本次 1.1 已使用的相关条目

- `openvla`
- `act`
- `diffusionpolicy`

对应位置：

- `robotdoc/论文/1_引言/01_研究背景.tex`

## 后续建议

- 如果继续补 `1.1` 的操作系统、ROS2、推理框架相关引用，优先直接复用 `references.bib` 现有条目。
- 若新增政策、行业报告或官网资料，建议单独分一组 key，并在文件名中标注来源与年份，避免和论文类文献混淆。

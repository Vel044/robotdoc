# ACT / VLA 模型原理 — 文献调研

## 1. ACT（Action Chunking with Transformers）

### Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（ACT 原始论文）
- **作者**：Zhao, T., Kumar, V., Levine, S., Finn, C.
- **年份**：2023
- **出版**：Robotics: Science and Systems (RSS) 2023
- **链接**：https://arxiv.org/abs/2304.13705 | 项目页：https://tonyzhaozh.github.io/aloha/
- **摘要**：提出 ACT 算法与 ALOHA 低成本硬件系统。核心贡献：**动作分块**（预测一段连续动作序列而非单步动作）和**时间集成**（temporal ensemble，对多个预测窗口取加权平均以降低抖动），在精细双臂操作任务上仅需 10 分钟演示数据即达 80–90% 成功率。

### Bi-ACT: Bilateral Control-Based Imitation Learning via Action Chunking with Transformer
- **作者**：Buamanee, T., Kobayashi, M., Uranishi, Y., Takemura, H.
- **年份**：2024
- **出版**：IEEE AIM 2024
- **链接**：https://arxiv.org/abs/2401.17698 | https://ieeexplore.ieee.org/iel8/10636941/10636942/10637173.pdf
- **摘要**：在 ACT 基础上融合双边控制（bilateral control）原理，同时预测关节位置、速度和扭矩，可动态适应物体硬度变化。

### InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation
- **作者**：Lee, A., Chuang, I., Chen, L.-Y., Soltani, I.
- **年份**：2024
- **出版**：CoRL 2024（PMLR）
- **链接**：https://arxiv.org/abs/2409.07914 | 项目页：https://soltanilara.github.io/interact/
- **摘要**：引入层级注意力机制捕捉双臂间相互依赖关系，并在预测块间共享跨臂信息，在模拟与真实双臂操作上均超越基础 ACT。

---

## 2. Diffusion Policy

### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **作者**：Chi, C., Xu, Z., Feng, S., Cousineau, E., et al.（Columbia、TRI、MIT）
- **年份**：2023
- **出版**：RSS 2023 / IJRR 2024
- **链接**：https://arxiv.org/abs/2303.04137 | 项目页：https://diffusion-policy.cs.columbia.edu/
- **摘要**：将 DDPM 扩散模型用于机器人策略学习，优雅处理多模态动作分布与高维动作空间，在 15 个操作任务上平均性能比基线提升 46.9%。

---

## 3. VLA（Vision-Language-Action Models）

### RT-1: Robotics Transformer for Real-World Control at Scale
- **作者**：Brohan, A., et al.（Google Robotics & Everyday Robots）
- **年份**：2022
- **出版**：arXiv 2212.06817 / RSS 2023
- **链接**：https://arxiv.org/abs/2212.06817 | 项目页：https://robotics-transformer1.github.io/
- **摘要**：首个大规模 Transformer 机器人策略模型，13 台机器人采集 130k+ 条轨迹覆盖 700+ 任务，新任务泛化能力分别比基线提升 25%、36%、18%。

### RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **作者**：Zitkovich, B., et al.（Google DeepMind）
- **年份**：2023
- **出版**：ICML 2023（PMLR）
- **链接**：https://arxiv.org/abs/2307.15818 | 项目页：https://robotics-transformer2.github.io/
- **摘要**：将预训练 VLM（PaLM-E）与机器人数据共同微调，直接输出机器人动作 token，在未见过场景上性能从 32% 提升至 62%，支持链式推理（CoT）。

### OpenVLA: An Open-Source Vision-Language-Action Model
- **作者**：Kim, M., Pertsch, K., et al.（Stanford ILIAD Lab）
- **年份**：2024
- **出版**：ICLR 2025（PMLR）
- **链接**：https://arxiv.org/abs/2406.09246 | 项目页：https://openvla.github.io/ | GitHub：https://github.com/openvla/openvla
- **摘要**：基于 Llama 2 的 7B 参数开源 VLA，在 970k 真实机器人演示上训练，性能超越 55B 参数的 RT-2-X，支持消费级 GPU 微调与 INT4/INT8 量化推理。

---

## 4. 通用基础模型

### Gato: A Generalist Agent
- **作者**：Reed, S., et al.（DeepMind）
- **年份**：2022
- **出版**：TMLR
- **链接**：https://arxiv.org/abs/2205.06175
- **摘要**：12 亿参数 Transformer 通用智能体，统一处理文本、图像、连续关节扭矩等多模态输入输出，执行 600+ 任务（含机器人控制、Atari 游戏、对话）。

### Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **作者**：多机构合作（DeepMind 主导，21 个机构参与）
- **年份**：2023
- **出版**：NeurIPS 2023
- **链接**：https://arxiv.org/abs/2310.08864 | 项目页：https://robotics-transformer-x.github.io/
- **摘要**：最大的开源真实机器人数据集，包含 1M+ 轨迹、22 种机器人机型、527 种技能，提供统一数据格式和 RT-X 跨机型训练框架。

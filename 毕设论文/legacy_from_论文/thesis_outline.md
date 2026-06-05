# LeRobot平台感知-推理-执行流程的系统级性能剖析研究

System-Level Performance Profiling of Perception-Inference-Execution Pipeline in the LeRobot Framework

学院：计算机学院

专业：软件工程

班级：08012204

学生姓名：俞乐楠

学号：1120221303

指导教师：王勇

日期：2026年4月20日

## 原创性声明

本人郑重声明：所呈交的毕业设计（论文），是本人在指导老师的指导下独立进行研究所取得的成果。除文中已经注明引用的内容外，本文不包含任何其他个人或集体已经发表或撰写过的研究成果。对本文的研究做出重要贡献的个人和集体，均已在文中以明确方式标明。

特此申明。

本人签名：________________

日期：______年______月______日

## 关于使用授权的声明

本人完全了解北京理工大学有关保管、使用毕业设计（论文）的规定，其中包括：①学校有权保管、并向有关部门送交本毕业设计（论文）的原件与复印件；②学校可以采用影印、缩印或其它复制手段复制并保存本毕业设计（论文）；③学校可允许本毕业设计（论文）被查阅或借阅；④学校可以学术交流为目的，复制赠送和交换本毕业设计（论文）；⑤学校可以公布本毕业设计（论文）的全部或部分内容。

本人签名：________________

日期：______年______月______日

指导教师签名：________________

日期：______年______月______日

## 摘要

本文面向LeRobot框架下ACT策略在SO-101六自由度机械臂上的在线控制任务，研究树莓派5这类无GPU ARM边缘平台中观测、推理与执行流程的系统级性能瓶颈。随着端到端机器人学习框架逐渐进入低成本硬件平台，系统能否稳定达到目标控制频率，不仅取决于模型规模和硬件配置，也受到Python运行时、深度学习框架线程行为、Linux内核调度以及设备I/O路径的共同影响。针对现有研究多关注模型压缩或通信优化、较少分析运行时和操作系统路径的问题，本文从机器人基础软件角度出发，对在线控制闭环进行跨层次剖析。

本文以LeRobot核心控制循环record.py为切入点，构建覆盖推理框架层、运行时层、内核层和硬件I/O层的四层性能分析模型。实验平台由树莓派5、SO-101机械臂、两路USB摄像头和ACT策略模型组成，围绕观测阶段的摄像头与舵机读取、推理阶段的ACT前向计算、执行阶段的舵机写入以及等待阶段的帧率控制记录时间开销。方法上，本文结合Python阶段计时、ftrace、strace和内核插桩，对系统端到端时延、模块级时间分布、系统调用阻塞和工具自身扰动进行量化分析，从而区分真实瓶颈与测量工具引入的额外开销。

实验结果表明，在30FPS目标配置下，系统同步执行时实测帧率约为18.7FPS，与目标仍存在明显差距。工具校准实验显示，延迟导出的ftrace使帧率下降13.9%，明显小于strace的40.9%，更适合后续定量分析。进一步分析发现，ACT前向推理是主要瓶颈，单次完整推理平均耗时约1925ms，其中ResNet18视觉编码与Transformer模块占主要比例；摄像头异步预取后主线程观测开销较低，舵机写入系统调用仅为微秒级，执行阶段主要受1Mbps波特率和半双工总线限制。针对推理瓶颈，本文设计并验证了异步推理方案。在当前平台的单轮扫描实验中，当异步推理触发阈值取0.30时，系统有效控制频率达到27.42FPS，相比同步基线提升70.0%。结果表明，系统级分层剖析能够有效定位边缘机器人在线控制的真实性能瓶颈，异步推理能够在不改变模型结构的条件下显著提升控制频率。

关键词：基础软件；Linux内核；系统级性能剖析；异步推理；机器人控制；ACT策略；LeRobot

## Abstract

In order to study the system-level performance bottlenecks of the online control loop of the ACT policy in the LeRobot framework on an SO-101 six-degree-of-freedom robotic arm, this work uses Raspberry Pi 5 as the edge platform and builds a four-layer profiling model covering the inference framework, runtime, kernel, and hardware I/O. Using ftrace, strace, and custom kernel instrumentation, the end-to-end latency and module-level time distribution are quantitatively analyzed.

The results show that, under a30FPS target configuration, the synchronous system reaches only about18.7FPS. Tool-calibration experiments show that deferred ftrace causes a13.9%FPS drop, much smaller than the40.9%drop caused by strace, and is therefore suitable for later quantitative analysis. ACT forward inference is identified as the dominant bottleneck, with an average latency of about1925ms per full pass, while the visual encoder and Transformer account for most of the cost. After asynchronous camera prefetching, observation overhead on the main thread becomes small, and the servo write syscall remains at the microsecond level; the execution stage is mainly constrained by the1Mbps baud rate and the half-duplex bus. To address the inference bottleneck, this work designs and validates an asynchronous inference scheme. In the current single-run sweep, setting the asynchronous inference trigger threshold to0.30 yields an effective control rate of27.42FPS, a70.0% improvement over the synchronous baseline. The results show that system-level layered profiling can effectively locate the real bottlenecks of edge robotic online control, and that asynchronous inference can significantly improve control frequency without changing the model structure.

Key Words: fundamental software;Linux kernel;system-level performance profiling;asynchronous inference;robotic control;ACT policy;LeRobot

## 目录

1. 引言
2. 实验环境构建
3. 性能分析
4. 优化
5. 研究局限性与未来工作
6. 结论

## 第一章引言

### 研究背景与意义

近年来，协作机器人与自主移动机器人在工业制造、仓储物流、医疗服务等领域加速落地，成为推动产业智能化转型的核心载体。视觉-语言-动作（Vision-Language-Action,VLA）大模型将视觉感知与动作决策统一到端到端神经网络中，显著拓展了机器人在开放场景下的泛化能力。在此背景下，如何在资源受限的边缘平台上高效、可靠地运行机器人智能软件，已成为制约机器人大规模部署的关键问题。

从技术栈角度看，机器人系统通常由三个层次组成。硬件层涵盖上位机（嵌入式计算板）、传感器（摄像头、力传感器）、执行器（电机、舵机）与机械结构，决定机器人的物理感知与运动能力；算法层包括视觉感知、策略规划与运动控制，以ACT和Diffusion Policy等端到端模型为代表；基础软件层则由操作系统、通信中间件和推理框架构成，向下管理硬件资源，向上支撑算法模型的运行。本文的研究聚焦于基础软件层：操作系统的内核调度策略、系统调用路径与设备驱动模型，以及推理框架的线程管理与算子调度，共同决定着机器人系统的硬件生态兼容性、推理性能、决策准确性、运行可靠性和系统安全性，是整个机器人软件栈性能与可信度的基础。

面向机器人基础软件的研究已有大量探索，但仍存在明显不足。广义上，机器人基础软件涵盖三个方向：操作系统内核、推理库、以及专用于机器人的通信工具。

在操作系统内核层面，Linux及其实时化改造方案仍是机器人系统最常见的承载平台，其通信栈、调度器行为与内核抢占机制会直接影响控制循环的时延和抖动；而更轻量的实时系统虽然在调度开销上更有优势，却往往难以直接承载完整的深度学习推理栈和复杂设备生态。

在推理库层面，PyTorch、TFLite等框架各有侧重，均以“外挂式”方式接入操作系统：其内置线程池（OpenMP）的调度与OS内核CFS（Completely Fair Scheduler，完全公平调度器）相互独立，在资源受限平台上容易引发调度冲突，造成额外的上下文切换与futex等待开销。

在机器人专用通信工具层面，ROS/ROS2显著降低了多节点系统的开发门槛，其组件化工作进一步改善了节点部署与通信组织方式，ros2_tracing等追踪框架也为系统分析提供了更好的观测手段，但节点通信、执行时序与消息传递延迟仍然受到底层操作系统调度与中间件实现的共同影响。

更根本的问题在于，现有性能优化与分析工作多聚焦于单一层次。主流优化路径往往集中在模型压缩、模型结构改进或硬件通信链路调优上，而对推理运行时、Python线程行为、OS内核调度、系统调用阻塞与设备I/O栈等基础软件因素关注不足。系统性能分析方法论强调，应从应用、运行时、操作系统和硬件路径联合定位瓶颈。近期AMS也表明，OS层原语会直接影响机器人推理管理的性能与可靠性。这使得许多工作虽然能够报告端到端延迟或单点优化收益，却缺乏从AI推理框架层到OS内核层的跨层次联合剖析，难以回答“瓶颈究竟在哪一层、由什么原因导致”等关键问题。

针对上述研究缺口，本文以HuggingFace LeRobot框架下ACT策略在树莓派5（4核ARM Cortex-A76处理器，4GB LPDDR4X，无GPU）上的在线推理为研究对象，系统开展机器人基础软件的跨层次性能剖析。本文实验表明，单次ACT完整推理耗时1924.8ms，其中ResNet18视觉编码器占66.6%（1281ms）为第一瓶颈；在chunk_size=100的同步推理配置下实测帧率仅18.7fps，距30fps目标仍有约38%缺口。通过推理异步化优化（异步推理触发阈值取0.30），有效帧率提升至27.42fps（相对提升70%），基本接近实时目标。本文提出覆盖AI推理框架层、Python运行时层、OS内核层和硬件I/O层的四层性能分析模型，建立面向“观测-推理-执行”紧耦合场景的多工具联合剖析方法，通过Python阶段计时、strace工具校准、ftrace追踪与内核自定义插桩，端到端量化各阶段时间开销，定位ResNet18编码器为第一瓶颈（占66.6%），揭示strace带来40.9%FPS下降，验证ftrace的低扰动必要性，为嵌入式机器人基础软件的性能优化提供明确的优先级依据。

### 国内外研究现状

机器人基础软件层面的研究可大致归为三个方向：操作系统、通信构件与开发SDK；本节按此三个方向梳理国内外现有工作，并指出研究空白。

在操作系统层面，研究核心是实时性保证。标准Linux内核基于CFS公平调度，不提供硬实时保证；PREEMPT-RT补丁通过把内核大段代码改为可抢占，将ARM嵌入式平台上的网络通信最坏延迟从vanilla Linux的13197µs压降至88–110µs，是目前机器人控制系统最主流的Linux实时化方案；但仅有PREEMPT-RT还不够，并发流量下仍需cpusets CPU绑核隔离才能维持有界延迟。ROS2实时性研究综述进一步指出，内核Executor与定制调度器是该方向密集投入的研究分支。近期AMS（Zheng et al.,2025）将action context、action exception、action replay三个OS原语融入VLA推理管理，在真实机械臂任务上将成功率提升7×–24×，执行步数减少29%–74%，表明OS层机制对机器人推理的性能与可靠性具有直接影响。

在通信构件层面，ROS2以DDS（Data Distribution Service）为底层通信层，是学术界与工业界的事实标准；但其节点调度仍依赖Linux CFS，无硬实时保证。针对ARM嵌入式平台的实测表明，节点组合方式（intra-process vs. multi-process）对CPU占用与消息延迟有显著影响；在树莓派3/4等平台上，Linux网络栈被发现是制约硬实时以太网通信的主要瓶颈。为支撑此类分析，ros2_tracing提供了基于LTTng的低开销追踪框架，可在不显著扰动主循环的条件下获取ROS2节点调度与消息传递的精细时序。

在SDK层面，深度学习推理框架是机器人基础软件最核心的组成之一。TFLite面向MCU级极端受限场景，支持INT8量化与ARM NEON SIMD加速；ONNX Runtime通过KleidiAI算子库在ARM64上实现28–51%的性能提升；PyTorch则提供TorchScript与CPU优化，是LeRobot等机器人开源框架的默认推理后端。在机器人生态软件方面，HuggingFace LeRobot等以Python为主的开源框架快速降低了ACT、Diffusion Policy等策略的训练与部署门槛，但围绕其性能特性的研究仍以模型精度与吞吐量为主要评估维度。更关键的是，上述推理框架与生态软件均以“外挂式”方式接入操作系统：其内置线程池（OpenMP或C10）的调度与OS内核CFS调度器相互独立，在资源受限平台上容易引发调度冲突，引入额外的futex阻塞与上下文切换开销，而这一类内核层耦合开销迄今缺乏系统性量化。

综上，已有工作多聚焦于上述三个层面中的单一层面：或针对内核做实时化改造、或评估通信中间件的延迟、或优化推理框架的算子性能；即便涉及多层（如ros2_tracing联合用户态-内核态追踪），其覆盖范围也仅限于ROS2通信链路。从Python运行时到AI推理框架再到Linux内核这条贯穿机器人推理主循环的端到端调用链，迄今仍缺乏系统性的跨层次联合剖析。本文的研究方向正是跨层次联合延迟分析：建立覆盖AI推理框架层、Python运行时层、OS内核层与硬件I/O层的四层性能模型，通过ftrace等工具联合剖析，端到端定位ARM嵌入式平台上机器人推理主循环的真实瓶颈所在。

### 研究目标

本文以LeRobot为代表的机器人软硬件栈为研究对象，构建覆盖AI推理框架层、Python运行时层、OS内核层与硬件I/O层的跨层次性能分析框架，弥补现有工作偏重模型或硬件通信单点优化、较少量化基础软件影响的不足，发现各阶段性能瓶颈，进行系统性量化测量，为后续性能优化提供实验依据。

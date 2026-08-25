# Embodied Task Agent

## 基于视觉感知、任务规划、执行验证与失败恢复的具身智能机械臂 Agent

本项目旨在构建一个能够根据自然语言指令，自主感知环境、规划任务、执行机器人技能，并根据视觉反馈判断执行结果和进行失败恢复的具身智能机器人 Agent。

项目使用 Franka Panda 机械臂作为机器人平台，围绕抓取、放置等桌面操作任务，逐步实现：

- 自然语言任务理解
- RGB-D 视觉感知
- 语言引导的目标识别与分割
- 三维目标定位
- 机器人运动学与轨迹执行
- 高层任务规划
- Pick / Place Skill
- 视觉执行验证
- Failure Recovery
- 自动化 Benchmark 与定量评测

项目最终目标并不是实现一个简单的：

```text
LLM 输出动作
      ↓
机器人执行
```

Demo，而是构建一个真正具有：

```text
Perception
    ↓
Planning
    ↓
Execution
    ↓
Verification
    ↓
Recovery
```

闭环能力的 **Vision-Grounded Embodied Agent**。

---

# 1. 项目目标

给定自然语言任务，例如：

```text
把绿色方块拿起来
```

或者：

```text
把绿色方块拿起来，然后放到右边
```

系统需要完成：

```text
Natural Language Instruction
            ↓
        LLM Planner
            ↓
      Visual Perception
            ↓
      Structured State
            ↓
       Skill Selection
            ↓
      Robot Execution
            ↓
     Visual Observation
            ↓
       Verification
            ↓
     Success / Failure
            ↓
   Recovery / Replanning
```

与直接读取仿真器真实物体坐标不同，本项目计划让机器人在运行时主要依赖相机获得的视觉信息完成：

- 目标识别
- 三维定位
- 抓取点估计
- Skill 执行
- 执行结果验证
- 失败恢复

Ground Truth 将与 Agent Runtime 隔离，仅用于实验评测。

---

# 2. 项目核心方向

项目正在从早期 Prototype：

```text
Language
   ↓
Qwen
   ↓
Pick / Place
   ↓
Robot
```

逐步升级为：

```text
Language
   ↓
Planner
   ↓
Vision
   ↓
Scene State
   ↓
Skill
   ↓
Robot Motion
   ↓
Vision
   ↓
Verifier
   ↓
Recovery
```

当前重点研究方向包括：

1. Vision-Grounded Manipulation
2. Structured Scene State
3. Skill-based Robot Control
4. Visual Execution Verification
5. Failure Diagnosis and Recovery
6. Closed-loop Task Execution
7. Quantitative Evaluation

其中项目后续最重要的目标之一是：

> Agent 在任务运行过程中不依赖仿真器 Ground Truth，而是依赖视觉感知得到的环境状态进行决策、控制和结果验证。

---

# 3. Franka Panda 基础控制

项目已经完成 Franka Panda 机械臂基础控制与状态理解实验，包括：

- Joint Position
- Joint Velocity
- TCP Pose
- Object Pose
- Goal Position
- Joint-space Control
- PD Controller 基础实验
- 机器人停止与速度衰减实验
- TCP 与目标物体之间的相对空间关系分析

通过这些实验已经建立对以下机器人状态的理解：

```text
qpos
qvel
TCP Pose
Object Pose
Goal Position
Action
```

并理解了：

```text
Observation ≠ Action
Joint State ≠ Cartesian State
qpos ≠ TCP Pose
```

当前 Robot Motion Layer 正在逐步模块化为：

```text
Target TCP Pose
      ↓
Inverse Kinematics
      ↓
Target Joint Position
      ↓
Trajectory Generation
      ↓
Robot Controller
      ↓
Franka Panda
```

---

# 4. Forward Kinematics 与 Inverse Kinematics

项目已经实现独立的 Panda Forward Kinematics / Inverse Kinematics 模块。

## 4.1 Forward Kinematics

Forward Kinematics：

```text
Joint Position
      ↓
      FK
      ↓
TCP Pose
```

输入：

```text
7-D Panda Joint Position
```

输出：

```text
TCP Position
+
TCP Orientation
```

即：

```text
Joint Space
    ↓
Cartesian Space
```

---

## 4.2 Inverse Kinematics

Inverse Kinematics：

```text
Target TCP Pose
      ↓
      IK
      ↓
Target Joint Configuration
```

输入：

```text
Target TCP Pose
```

输出：

```text
7-D Joint Target
```

当前 IK 模块使用成熟的机器人运动学工具完成求解，而不是自行重新实现完整 IK 数学算法。

在得到 IK 解之后，系统会再次进行：

```text
IK Result
    ↓
Forward Kinematics
    ↓
Calculated TCP Pose
    ↓
Compare With Target TCP Pose
```

从而验证 IK 求解结果。

---

# 5. Robot Motion Layer

Robot Motion Layer 后续主要由三个模块组成：

```text
robot/

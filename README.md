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
```

---

# 6. Results

V1 最终评测由 Main Benchmark 和 Complex Language Stress Test 组成：

- **Main Benchmark:** 60 episodes — **Task Success: 100%**
- **Complex Language Stress Test:** 30 episodes — **Task Success: 100%**
- **Total evaluated episodes:** 90
- **Mean 3D Localization Error:** about **6.21 mm**
- **Mean Placement Error:** about **3.57 mm**

其中，Failure Recovery benchmark contains deliberately injected invalid actions。相关结果用于验证 ActionGuard 与 Recovery 路径，不应解释为 Planner 的自然错误率。图中的 Run 1 是修复前的诊断基线，不计入上述 90 个 V1 最终评测 Episode。

## 6.1 Benchmark Comparison

![Benchmark comparison](results/figures/03_benchmark_comparison.png)

第一轮 Benchmark 的 Task Success 为 75%，暴露了 cross-episode agent belief state leakage。完成 episode-level reset 修复后，相同的 60-case Main Benchmark 达到 100%；随后进行的 30-case complex-language stress test 同样达到 100%。

## 6.2 Spatial Evaluation Errors

![Mean spatial errors](results/figures/02_error_metrics.png)

Main Benchmark 的 Mean 3D Localization Error 约为 6.21 mm，Mean Placement Error 约为 3.57 mm。MuJoCo ground truth 仅用于独立 Evaluation 和结果记录，不参与 Runtime 控制、任务规划、状态更新或恢复决策。

## 6.3 Main Benchmark Success Rates

![Main Benchmark success rates](results/figures/01_success_rates.png)

在 Main Benchmark 中，Task、Vision、Pick、Place 和 Recovery 的成功率均为 100%。这些结果仅适用于 tested single-object MuJoCo workspace configuration，不代表任意真实机器人环境中的普遍成功率，也不构成 real-world robot performance 声明。

---

# 7. Running the Project

从项目根目录安装直接依赖：

```bash
python -m pip install -r requirements.txt
```

通过环境变量或本地 `.env` 配置 Qwen API credential：

```dotenv
DASHSCOPE_API_KEY=your_api_key_here
base_url=your_compatible_api_base_url
```

不要将 API key 或本地 `.env` 提交到仓库。

从项目根目录运行交互式 Agent：

```bash
python panda_qwen_agent.py
```

运行正式 Benchmark：

```bash
python run_benchmark.py
```

根据已有结果生成可视化：

```bash
python evaluation/plot_results.py
```

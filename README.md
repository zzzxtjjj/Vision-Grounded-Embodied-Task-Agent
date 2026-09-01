# Vision-Grounded Embodied Task Agent

## 基于视觉定位、任务规划、机器人技能与失败恢复的具身智能机械臂 Agent

本项目旨在构建一个能够根据自然语言指令，自主感知环境、规划任务、执行机器人技能，并根据 Runtime 执行状态进行失败恢复的具身智能机器人 Agent。

项目使用 Franka Panda 机械臂作为机器人平台，围绕抓取、放置等桌面操作任务，逐步实现：

- 自然语言任务理解
- RGB-D 视觉感知
- 语言引导的目标识别与分割
- 三维目标定位
- 机器人运动学与轨迹执行
- 高层任务规划
- Pick / Place Skill
- 独立 Ground Truth Evaluation
- Failure Recovery
- 自动化 Benchmark 与定量评测

项目最终目标并不是实现一个简单的顺序控制脚本，而是保持 Runtime 控制与 Evaluation 的明确边界：

```text
Runtime:
Language → Planner → Vision（动作前定位） → ActionGuard
         → Skill → Robot Execution → StateManager

Evaluation:
MuJoCo Ground Truth → GTEvaluator → Metrics / Records only
```

Recovery / Replanning 只响应 Runtime 中的执行状态和错误；Ground Truth 不反馈给 Runtime。

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
 Visual Perception（动作前定位）
            ↓
       ActionGuard
            ↓
       Skill Selection
            ↓
      Robot Execution
            ↓
       StateManager
            ↓
   Recovery / Replanning
```

与直接读取仿真器真实物体坐标不同，本项目计划让机器人在运行时主要依赖相机获得的视觉信息完成：

- 目标识别
- 三维定位
- 抓取点估计
- Skill 执行
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
ActionGuard
   ↓
Skill
   ↓
Robot Motion
   ↓
StateManager
   ↓
Recovery
```

当前重点研究方向包括：

1. Vision-Grounded Manipulation
2. Structured Scene State
3. Skill-based Robot Control
4. Runtime / Evaluation Isolation
5. Failure Diagnosis and Recovery
6. Closed-loop Task Execution
7. Quantitative Evaluation

其中项目后续最重要的目标之一是：

> Agent 在任务运行过程中不依赖仿真器 Ground Truth，而是依赖动作前视觉定位、机器人执行结果和内部状态进行决策、控制与恢复。

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

当前 V1 使用 MuJoCo Forward Kinematics，并在 Agent 中实现基于 Jacobian 的阻尼最小二乘 Inverse Kinematics。

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

当前实现通过 `mujoco.mj_forward` 更新并读取末端执行器位姿。

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

当前 IK 使用 MuJoCo 提供的 site Jacobian，并通过阻尼最小二乘迭代计算 7-D joint target。

求解过程中每轮通过 MuJoCo Forward Kinematics 更新末端位姿，并与目标 TCP Pose 比较：

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

当前 V1 的 Robot Motion Layer 位于 `panda_qwen_agent.py`，主要流程为：

```text
Target TCP Position
        ↓
solve_ik
        ↓
JointTrajectory
        ↓
move_arm_to / move_fingers
        ↓
MuJoCo Robot Execution
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

## 6.4 Metric Definitions and Limitations

- **Vision Success** 表示 `PerceptionSystem` 成功返回可用的结构化观察；定位精度由独立的 3D localization error 报告，而不是由额外成功阈值判定。
- **Pick Success** 要求方块相对 episode 初始位置至少抬升 0.05 m，且方块与夹爪距离不超过 0.08 m。
- **Place / Place Task Success** 当前仅使用方块与目标区域之间不超过 0.06 m 的 XY distance；未评估 Z、orientation 或 stable placement，因此不应解释为完整 6-DoF placement evaluation。
- Home keyframe 中方块初始位置位于 center region，因此 center 的 task-level XY 判定单独看不能证明机器人实际完成了搬运；正式结果还应结合每个 episode 的 GT Pick / Place action records 解读。本仓库发布的 Main / Complex 成功 Place episodes 均包含成功的 Pick 和 Place action records。
- **Recovery Success** 表示 deliberately injected invalid action 被 ActionGuard 拒绝后，下一次 Runtime action 成功执行。Main Benchmark 的 12/12 recovery episodes 和 Complex Stress Test 的 6/6 recovery episodes 同时通过最终 GT task evaluation；该指标不代表任意物理故障下的通用恢复能力。

---

# 7. Running the Project

准备 Python 3 环境，并从项目根目录安装直接依赖。`requirements.txt` 包含 MuJoCo、PyTorch、Transformers / Florence-2、OpenAI-compatible Qwen client、NumPy、Pillow 和 Matplotlib：

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

运行 60-episode Main Benchmark：

```bash
python run_benchmark.py --suite main
```

运行 30-episode Complex Language Stress Test：

```bash
python run_benchmark.py --suite complex
```

新的运行结果分别写入 `results/benchmark_main_latest.json` 和 `results/benchmark_complex_latest.json`，不会覆盖三份正式发布结果。60-episode Main Benchmark 和 30-episode Complex Stress Test 的正式原始结果均保存在 `results/`。运行 Agent / Benchmark 需要 Qwen API 访问；Florence-2 首次加载通常还需要下载模型。`qwen-plus` 是远程服务模型名称，其具体服务版本不能由本仓库完全锁定。

根据已有结果生成可视化：

```bash
python evaluation/plot_results.py
```

---

# 8. License

本项目原创代码采用根目录 [MIT License](LICENSE)。Third-party assets notice：`franka_panda/` 中的第三方模型和资源仍遵循其目录内的原始许可证；根目录 MIT License 不替代或修改这些第三方许可条款。

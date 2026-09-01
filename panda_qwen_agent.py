import os
import json
import time
import numpy as np
import mujoco
import mujoco.viewer

from openai import OpenAI
from dotenv import load_dotenv
from vision.perception import PerceptionSystem

from place_regions import get_place_target
from core.state_manager import RobotStateManager
from core.recovery_policy import RecoveryPolicy
from core.action_guard import ActionGuard
from evaluation.gt_evaluator import GTEvaluator

# ==================== Benchmark / Failure Injection ====================

# 人为制造执行偏差，用于 Benchmark 故障注入。
DEBUG_FORCE_PICK_FAILURE = False
DEBUG_PICK_OFFSET = np.array(
    [0.10, 0.0, 0.0],
    dtype=float,
)
DEBUG_FORCE_INVALID_ACTION = False
DEBUG_FORCE_PLACE_FAILURE = False
DEBUG_PLACE_OFFSET = np.array(
    [0.10, 0.0, 0.0],
    dtype=float,
)
DEBUG_PLACE_FAILURE_USED = False

# ==================== Qwen ====================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("base_url")
)

QWEN_MODEL = "qwen-plus"


# ==================== MuJoCo ====================

model = mujoco.MjModel.from_xml_path("franka_panda/mjx_single_cube.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
mujoco.mj_forward(model, data)

gripper_id = model.site("gripper").id

GRIPPER_OPEN = float(data.qpos[7])
GRIPPER_CLOSED = 0.010

# 当前视觉目标
VISION_TARGET = "green cube"

# ==================== 视觉感知系统 ====================

# 整个程序只创建一个统一的视觉感知接口。
#
# Agent 后面不再直接调用 vision_system.py，
# 而是统一通过 PerceptionSystem 获取视觉结果。
perception = PerceptionSystem()

# 夹爪保持垂直向下
R_des = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


# ==================== 轨迹规划 ====================

def JointTrajectory(t, q_start, q_end, duration):
    if duration <= 0:
        return q_end.copy(), np.zeros(7), np.zeros(7)

    s = min(t / duration, 1.0)

    a = 10 * s**3 - 15 * s**4 + 6 * s**5
    v = 30 * s**2 - 60 * s**3 + 30 * s**4
    ac = 60 * s - 180 * s**2 + 120 * s**3

    q_des = q_start + (q_end - q_start) * a
    qvel_des = (q_end - q_start) * v / duration
    qacc_des = (q_end - q_start) * ac / duration**2

    return q_des, qvel_des, qacc_des


# ==================== IK ====================

def solve_ik(target_pos):
    saved_q = data.qpos[:7].copy()

    for _ in range(100):
        mujoco.mj_forward(model, data)

        # 位置误差
        pos_error = np.asarray(target_pos, float) - data.site_xpos[gripper_id]

        # 姿态误差
        R_cur = data.site_xmat[gripper_id].reshape(3, 3).copy()
        R_err = R_cur.T @ R_des

        ori_body = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

        ori_error = R_cur @ ori_body
        error = np.concatenate([pos_error, ori_error])

        if np.linalg.norm(error) < 0.001:
            break

        # Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, gripper_id)

        J = np.vstack([
            jacp[:, :7],
            jacr[:, :7]
        ])

        # 阻尼最小二乘
        A = J @ J.T + 0.01 * np.eye(6)
        dq = J.T @ np.linalg.solve(A, error)

        data.qpos[:7] += dq

    result = data.qpos[:7].copy()

    # IK计算完后恢复真实仿真状态
    data.qpos[:7] = saved_q
    mujoco.mj_forward(model, data)

    return result


# ==================== 机械臂运动 ====================

def move_arm_to(target_pos, duration, viewer=None):
    q_start = data.qpos[:7].copy()
    q_end = solve_ik(target_pos)

    steps = int(duration / model.opt.timestep)

    for i in range(steps):
        t = i * model.opt.timestep

        q_des, _, _ = JointTrajectory(t, q_start, q_end, duration)
        data.ctrl[:7] = q_des

        mujoco.mj_step(model, data)

        if viewer:
            viewer.sync()
            time.sleep(0.002)


# ==================== 夹爪控制 ====================

def move_fingers(target, steps=800, viewer=None):
    start = data.qpos[7]

    for i in range(steps):
        progress = (i + 1) / steps

        data.ctrl[7] = start + (target - start) * progress
        mujoco.mj_step(model, data)

        if viewer:
            viewer.sync()
            time.sleep(0.002)


def open_gripper(viewer=None):
    move_fingers(GRIPPER_OPEN, viewer=viewer)


def close_gripper(viewer=None):
    move_fingers(GRIPPER_CLOSED, 1000, viewer)

# ==================== Robot State ====================

state_manager = RobotStateManager()
recovery_policy = RecoveryPolicy(max_retries=2)
action_guard = ActionGuard()
gt_evaluator = GTEvaluator(
    min_lift_height=0.05,
    max_gripper_distance=0.08,
)

# ==================== Vision Pick Skill ====================

def pick_object_at(grasp_position, viewer=None):
    """
    根据视觉系统给出的世界坐标抓取物体。

    返回 True 只表示 Pick Skill 的机器人动作正常执行完成，
    不表示 GT 已证明抓取成功。

    grasp_position:
        [x, y, z]
        单位：米
        坐标系：MuJoCo World
    """

    grasp_position = np.asarray(grasp_position, dtype=float)

    print("\n[Pick] 视觉抓取位置:", grasp_position)

    # 根据视觉位置生成三个运动关键点
    pre_grasp = grasp_position + np.array([0.0, 0.0, 0.10])
    grasp_pos = grasp_position.copy()
    lift_pos = grasp_position + np.array([0.0, 0.0, 0.20])

    print("[Pick] pre_grasp:", pre_grasp)
    print("[Pick] grasp_pos:", grasp_pos)
    print("[Pick] lift_pos:", lift_pos)

    print("[Pick] 打开夹爪")
    open_gripper(viewer)

    print("[Pick] 移动到视觉目标上方")
    move_arm_to(pre_grasp, 4.0, viewer)

    print("[Pick] 下降到视觉抓取位置")
    move_arm_to(grasp_pos, 3.5, viewer)

    print("[Pick] 闭合夹爪")
    close_gripper(viewer)

    print("[Pick] 抬起")
    move_arm_to(lift_pos, 4.0, viewer)

    print("[Pick] 机械臂动作执行完成")
    return True


# ==================== Place Skill ====================

def place_object(target, viewer=None):
    """
    使用固定 Workspace 目标执行 Place。

    返回的 success 只表示机器人动作正常执行完成，
    不表示 GT 已证明物体放置成功。
    """

    # 1. 获取固定放置目标
    try:
        target_position = get_place_target(target)
    except ValueError as e:
        print("[Place] 非法放置目标:", e)
        return False, "invalid_target", False

    print(f"\n[Place] 目标区域: {target}")
    print("[Place] 固定目标:", target_position)

    global DEBUG_PLACE_FAILURE_USED
    execution_target = target_position.copy()
    if (
        DEBUG_FORCE_PLACE_FAILURE
        and not DEBUG_PLACE_FAILURE_USED
    ):
        execution_target = (
            execution_target
            + DEBUG_PLACE_OFFSET
        )

        DEBUG_PLACE_FAILURE_USED = True

        print("[Debug] 强制制造 Place 失败")
        print(
            "[Debug] 实际执行位置:",
            execution_target,
        )

    # 2. 生成运动关键点
    above = execution_target + np.array([0.0, 0.0, 0.20])
    place_position = execution_target.copy()
    retreat = execution_target + np.array([0.0, 0.0, 0.10])
    print("[Place] above:", above)
    print("[Place] place:", place_position)
    print("[Place] retreat:", retreat)

    # 3. 执行 Place
    print(f"[Place] 移动到 {target} 上方")
    move_arm_to(above, 4.0, viewer)

    print("[Place] 下降")
    move_arm_to(place_position, 3.5, viewer)

    print("[Place] 松开夹爪")
    open_gripper(viewer)

    released = True

    # 等待物体落稳
    for _ in range(300):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

    print("[Place] 撤离")
    move_arm_to(retreat, 3.0, viewer)

    print("[Place] 机械臂动作执行完成")
    return True, None, True


# ==================== Qwen Planner ====================

SYSTEM_PROMPT = """
你是 Franka Panda 机械臂的高层任务规划器。

你的任务：
根据用户自然语言任务和当前机器人状态，
每次只生成下一步动作。

====================
允许动作
====================

只能输出以下动作：

pick
place
finish


place 动作的 target 只能是：

left
right
center


====================
机器人状态说明
====================

当前状态包含：

holding:
表示机器人当前是否抓着物体。

可能值：

"box"
表示当前抓着方块。

None
表示当前没有抓着方块。


last_action:
表示上一次尝试执行的动作。

例如：

"pick"

"place"


last_result:
表示上一次动作执行结果。

True:
动作成功。

False:
动作失败。


last_error:
表示上一次动作失败的原因。

可能值：

null
vision_failed
motion_failed
pick_failed
place_failed
invalid_target
already_holding
not_holding_box
invalid_action


last_target:
表示上一次 place 的目标位置。

例如：

"left"

"center"

"right"


history:
表示当前任务执行历史。

包含之前执行过的动作和结果。

例如：

[
 {
  "action":"pick",
  "result":true
 },
 {
  "action":"place",
  "result":false
 }
]


====================
任务完成规则
====================


1.
如果用户只要求拿起方块：

例如：

"把方块拿起来"


当：

last_action == "pick"

并且：

last_result == true


说明抓取任务已经完成。


下一步必须输出：

{
 "action":"finish",
 "target":null
}



2.
如果用户要求：

"把方块拿起来放到右边"

或者类似任务。


当：

last_action == "place"

并且：

last_result == true

并且：

last_target == 用户要求的位置


说明任务已经完成。


下一步必须输出：

{
 "action":"finish",
 "target":null
}



====================
动作规划规则
====================

硬状态约束（最高优先级）：

- holding == None 时，禁止输出 place。
  如果任务仍需要放置方块，必须先输出 pick。

- holding == "box" 时，禁止输出 pick。
  如果任务需要放置，应该输出 place。

- 如果上一次 place 失败，并且 holding == None，
  说明物体已经释放。禁止直接再次 place，
  必须先重新 pick，再继续后续任务。


1.
如果任务需要操作方块：

并且：

holding == None


生成：

pick


例如：

用户：
"把方块拿起来"

输出：

{
 "action":"pick",
 "target":null
}



2.
如果：

holding == "box"

并且：

用户任务包含放置要求


生成：

place


例如：

用户：
"把方块拿起来放到center"


输出：

{
 "action":"place",
 "target":"center"
}



3.
如果：

上一步动作失败：

last_result == false


根据：
last_action

last_error

history


重新规划。


不要假设动作一定成功。


例如：

如果：

vision_failed

可以重新尝试观察。


如果：

pick_failed

可以重新执行pick。


如果：

place_failed

可以重新规划place。


====================
视觉系统规则
====================


物体位置由视觉系统负责。

你不能生成：

- XYZ坐标
- 关节角
- 速度
- 加速度
- 力矩


禁止根据物体位置规划。


你只负责：

任务理解

动作选择

高层规划



====================
输出格式
====================


必须只返回 JSON。


格式：

{
 "action":"pick/place/finish",
 "target":"left/right/center 或 null"
}


不要输出任何解释文字。
"""


def ask_qwen(goal):
    # 不再把真实 box_position 告诉 Qwen
    state = {
        **state_manager.get_state(),
        "gripper_position": np.round(
            data.site_xpos[gripper_id], 3
        ).tolist()
    }

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content":
                    f"最终任务：{goal}\n"
                    f"当前状态：{json.dumps(state, ensure_ascii=False)}\n"
                    "请用 JSON 生成下一步动作。"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


# ==================== Executor ====================

def execute_action(action, viewer=None, vision_records=None):
    error = None
    action_type = action.get("action")
    target = action.get("target")
    state_manager.update_action(action_type)

    # ---------- Pick ----------
    if action_type == "pick":

        if state_manager.get_state()["holding"] is not None:

            error = "already_holding"

            print("非法动作：已经抓着物体")

            state_manager.update_result(False)
            state_manager.update_error(error)

            state_manager.add_history(
                action_type,
                False,
                error
            )

            return False

        print("\n[Vision] 开始定位目标:", VISION_TARGET)

        before_observation = perception.observe_object(
            target_name=VISION_TARGET,
            mj_model=model,
            mj_data=data,
        )

        if not before_observation.success:
            if vision_records is not None:
                vision_records.append({
                    "success": False,
                    "localization_error": None,
                })

            print(
                "[Vision] 定位失败:",
                before_observation.error,
            )

            error = "vision_failed"
            state_manager.update_error(error)

            success = False

        else:
            # Evaluation 只读取原始 Vision 输出，不读取故障注入后的抓取点。
            try:
                vision_eval = gt_evaluator.evaluate_vision(
                    model,
                    data,
                    before_observation.grasp_position,
                )

                print("\n[GT Vision Evaluation]")
                print("success =", vision_eval["success"])
                print(
                    "localization_error =",
                    vision_eval["localization_error"],
                )

                if vision_records is not None:
                    vision_records.append({
                        "success": True,
                        "localization_error": (
                            vision_eval["localization_error"]
                        ),
                    })
            except Exception as e:
                print("\n[GT Vision Evaluation]")
                print("evaluation_error =", e)

            grasp_position = before_observation.grasp_position

            # Benchmark / Failure Injection：人为偏移实际执行抓取点。
            if DEBUG_FORCE_PICK_FAILURE:
                grasp_position = (
                    grasp_position
                    + DEBUG_PICK_OFFSET
                )

                print("[Debug] 强制制造 Pick 失败")
                print(
                    "[Debug] 偏移后的抓取位置:",
                    grasp_position,
                )

            print("[Vision] 定位完成")
            print("[Vision] 目标:", before_observation.target_name)
            print("[Vision] 世界坐标抓取点:", grasp_position)

            execution_completed = pick_object_at(
                grasp_position,
                viewer,
            )

            if not execution_completed:
                print("[Pick] 机械臂执行过程失败")

                error = "motion_failed"
                success = False
            else:
                success = True
                error = None

        state_manager.update_error(error)
        state_manager.update_holding("box" if success else None)

    # ---------- Place ----------
    elif action_type == "place":

        if state_manager.get_state()["holding"] != "box":

            error = "not_holding_box"

            print("非法动作：当前没有抓住 box")

            state_manager.update_result(False)
            state_manager.update_error(error)

            state_manager.add_history(
                action_type,
                False,
                error
            )

            return False

        success, error, released = place_object(
            target,
            viewer
        )

        state_manager.update_error(error)
        state_manager.update_target(target)

        # 只要已经执行松爪，
        # 无论后续执行结果如何，都不能继续认为手里有方块
        if released:
            state_manager.update_holding(None)

    # ---------- Finish ----------
    elif action_type == "finish":
        return "finish"

    else:
        error = "invalid_action"

        print("非法 action:", action_type)

        state_manager.update_result(False)
        state_manager.update_error(error)

        state_manager.add_history(
            action_type,
            False,
            error
        )

        return False

    state_manager.update_result(success)
    state_manager.add_history(
        action_type,
        success,
        state_manager.get_error(),
    )

    return success


# ==================== Step-by-Step Agent ====================

def run_task(goal, viewer):
    global DEBUG_PLACE_FAILURE_USED
    DEBUG_PLACE_FAILURE_USED = False
    vision_records = []
    action_records = []
    planner_steps = 0
    invalid_actions = 0
    recovery_attempts = 0
    recovery_successes = 0
    recovery_pending = False
    episode_data = {}

    try:
        MAX_STEPS = 8
        state_manager.reset_task()

        try:
            gt_evaluator.start_episode(
                model,
                data,
            )
        except Exception as e:
            print("\n[GT Evaluation]")
            print("start_episode_error =", e)

        print("\n用户任务：", goal)
        print("当前机器人状态：", state_manager.get_state())

        for step in range(1, MAX_STEPS + 1):
            print(f"\n========== Step {step} ==========")

            try:
                action = ask_qwen(goal)
                planner_steps += 1
            except Exception as e:
                print("Qwen 调用失败：", e)
                return episode_data

            if (
                DEBUG_FORCE_INVALID_ACTION
                and step == 1
                and state_manager.get_state()["holding"] is None
            ):
                action = {
                    "action": "place",
                    "target": "center"
                }

                print("[Debug] 强制制造非法 Planner 动作")

            print("Qwen 下一步：", action)

            guard_result = action_guard.validate(
                action,
                state_manager.get_state()
            )

            if not guard_result["valid"]:
                invalid_actions += 1
                action_type = action.get("action")
                error = guard_result["error"]

                print(
                    f"[Action Guard] 拦截非法动作: "
                    f"action={action_type}, error={error}"
                )

                state_manager.update_action(action_type)
                state_manager.update_result(False)
                state_manager.update_error(error)

                state_manager.add_history(
                    action_type,
                    False,
                    error
                )

                recovery = recovery_policy.decide(
                    state_manager.get_state()
                )

                mode = recovery["mode"]
                reason = recovery["reason"]

                if mode in {"retry", "replan"}:
                    recovery_attempts += 1
                    recovery_pending = True

                print(
                    f"[Recovery] mode={mode}, reason={reason}"
                )

                if mode == "replan":
                    continue

                if mode == "abort":
                    print(
                        f"[Recovery] 任务终止，原因: {reason}"
                    )
                    return episode_data

                print(
                    f"[Action Guard] 非法动作无法安全处理，"
                    f"mode={mode}"
                )
                return episode_data

            while True:
                result = execute_action(
                    action,
                    viewer,
                    vision_records,
                )

                if (
                    action.get("action") == "pick"
                    and result is True
                ):
                    try:
                        pick_eval = gt_evaluator.evaluate_pick(
                            model,
                            data,
                        )

                        print("\n[GT Pick Evaluation]")
                        print(
                            "success =",
                            pick_eval["success"],
                        )
                        print(
                            "lift_height =",
                            pick_eval["lift_height"],
                        )
                        print(
                            "gripper_distance =",
                            pick_eval["gripper_distance"],
                        )
                        action_records.append({
                            "action": "pick",
                            "target": None,
                            "gt_success": pick_eval["success"],
                            "lift_height": pick_eval["lift_height"],
                            "gripper_distance": (
                                pick_eval["gripper_distance"]
                            ),
                        })
                    except Exception as e:
                        print("\n[GT Pick Evaluation]")
                        print("evaluation_error =", e)

                if (
                    action.get("action") == "place"
                    and result is True
                ):
                    try:
                        target_position = get_place_target(
                            action.get("target")
                        )
                        place_eval = gt_evaluator.evaluate_place(
                            model,
                            data,
                            target_position,
                        )

                        print("\n[GT Place Evaluation]")
                        print(
                            "success =",
                            place_eval["success"],
                        )
                        print(
                            "xy_error =",
                            place_eval["xy_error"],
                        )
                        action_records.append({
                            "action": "place",
                            "target": action.get("target"),
                            "gt_success": place_eval["success"],
                            "position_error": place_eval["xy_error"],
                        })
                    except Exception as e:
                        print("\n[GT Place Evaluation]")
                        print("evaluation_error =", e)

                if result == "finish":
                    print("\n任务完成")
                    return episode_data

                print("执行结果：", result)

                if result is True:
                    if recovery_pending:
                        recovery_successes += 1
                        recovery_pending = False
                    break

                recovery = recovery_policy.decide(
                    state_manager.get_state()
                )

                mode = recovery["mode"]
                reason = recovery["reason"]

                if mode in {"retry", "replan"}:
                    recovery_attempts += 1
                    recovery_pending = True

                print(
                    f"[Recovery] mode={mode}, reason={reason}"
                )

                if mode == "retry":
                    continue

                if mode == "replan":
                    break

                if mode == "abort":
                    print(
                        f"[Recovery] 任务终止，原因: {reason}"
                    )
                    return episode_data

                print(
                    f"[Recovery] 未知恢复模式，任务终止: {mode}"
                )
                return episode_data

        print("\n达到最大步骤数，停止任务")

    finally:
        episode_data.update({
            "task": goal,
            "vision_records": vision_records,
            "action_records": action_records,
            "planner_steps": planner_steps,
            "invalid_actions": invalid_actions,
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
        })

        print("\n[Benchmark Episode Data]")
        print("vision_records =", vision_records)
        print("action_records =", action_records)
        print("planner_steps =", planner_steps)
        print("invalid_actions =", invalid_actions)
        print("recovery_attempts =", recovery_attempts)
        print("recovery_successes =", recovery_successes)

    return episode_data


# ==================== Main ====================

def main():
    print("Qwen + Vision + Franka Panda")
    print("视觉系统：Florence-2 + Depth + Point Cloud")
    print("示例：把方块拿起来")
    print("示例：把方块拿起来放到右边")
    print("输入 exit 退出")

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            command = input("\n请输入任务 > ").strip()

            if command.lower() == "exit":
                break

            if command:
                run_task(command, viewer)

if __name__ == "__main__":
    main()

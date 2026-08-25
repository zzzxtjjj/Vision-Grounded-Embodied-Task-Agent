import os
import json
import time
import numpy as np
import mujoco
import mujoco.viewer

from openai import OpenAI
from dotenv import load_dotenv
from vision.perception import PerceptionSystem
from vision.verifier import (
    VisualPickVerifier,
    VisualPlaceVerifier,
)

from place_regions import get_place_target

# ==================== Pick失败测试开关 ====================

# False：
# 正常使用视觉系统估计的抓取位置。
#
# True：
# 仅用于测试 VisualPickVerifier，
# 人为把抓取位置沿 X 方向偏移 10 cm，
# 让机械臂大概率抓不到方块。
#
# 测试结束后必须改回 False。
DEBUG_FORCE_PICK_FAILURE = False

DEBUG_PICK_OFFSET = np.array(
    [0.10, 0.0, 0.0],
    dtype=float,
)
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
box_id = model.body("box").id


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

# ==================== pick视觉验证系统 ====================

pick_verifier = VisualPickVerifier(
    min_lift_height=0.05
)

# ==================== Place视觉验证系统 ====================

place_verifier = VisualPlaceVerifier(
    xy_tolerance=0.06
)

# ==========================================================

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


# ==================== Ground Truth ====================
# 这里只用于判断实验是否成功。
# 不再用于决定机器人去哪抓。

def get_box_pos():
    mujoco.mj_forward(model, data)
    return data.xpos[box_id].copy()


# ==================== Robot State ====================

robot_state = {
    "holding": None,
    "last_action": None,
    "last_success": None,
    "last_target": None
}

def reset_task_history():
    """
    开始一个新用户任务时，清除上一个任务的执行历史。

    注意：
    holding 不重置，因为它表示机器人当前真实是否抓着物体。
    """
    robot_state["last_action"] = None
    robot_state["last_success"] = None
    robot_state["last_target"] = None

# ==================== Vision Pick Skill ====================

def pick_object_at(grasp_position, viewer=None):
    """
    根据视觉系统给出的世界坐标抓取物体。

    grasp_position:
        [x, y, z]
        单位：米
        坐标系：MuJoCo World
    """

    grasp_position = np.asarray(grasp_position, dtype=float)

    print("\n  [Pick] 视觉抓取位置:", grasp_position)

    

    # 根据视觉位置生成三个运动关键点
    pre_grasp = grasp_position + np.array([0.0, 0.0, 0.10])
    grasp_pos = grasp_position.copy()
    lift_pos = grasp_position + np.array([0.0, 0.0, 0.20])

    print("  [Pick] pre_grasp:", pre_grasp)
    print("  [Pick] grasp_pos:", grasp_pos)
    print("  [Pick] lift_pos:", lift_pos)

    print("  [Pick] 打开夹爪")
    open_gripper(viewer)

    print("  [Pick] 移动到视觉目标上方")
    move_arm_to(pre_grasp, 4.0, viewer)

    print("  [Pick] 下降到视觉抓取位置")
    move_arm_to(grasp_pos, 3.5, viewer)

    print("  [Pick] 闭合夹爪")
    close_gripper(viewer)

    print("  [Pick] 抬起")
    move_arm_to(lift_pos, 4.0, viewer)

    print("  [Pick] 机械臂动作执行完成")

    # 注意：
    # 这里不再判断 Pick 是否成功。
    #
    # pick_object_at() 只负责“执行动作”。
    # 成功判断交给外层的 VisualPickVerifier。
    return True


# ==================== Place Skill ====================

def place_object(target, viewer=None):
    """
    使用固定 Workspace 目标执行 Place，
    并通过视觉判断最终是否放置成功。
    Runtime 不读取物体 Ground Truth。
    """

    # 1. 获取固定放置目标
    try:
        target_position = get_place_target(target)
    except ValueError as e:
        print("  [Place] 非法放置目标:", e)
        return False

    print(f"\n  [Place] 目标区域: {target}")
    print("  [Place] 固定目标:", target_position)

    # 2. 生成运动关键点
    above = target_position + np.array([0.0, 0.0, 0.20])
    place_position = target_position.copy()
    retreat = target_position + np.array([0.0, 0.0, 0.10])

    print("  [Place] above:", above)
    print("  [Place] place:", place_position)
    print("  [Place] retreat:", retreat)

    # 3. 执行 Place
    print(f"  [Place] 移动到 {target} 上方")
    move_arm_to(above, 4.0, viewer)

    print("  [Place] 下降")
    move_arm_to(place_position, 3.5, viewer)

    print("  [Place] 松开夹爪")
    open_gripper(viewer)

    # 等待物体落稳
    for _ in range(300):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

    print("  [Place] 撤离")
    move_arm_to(retreat, 3.0, viewer)

    # 4. Place 后重新进行视觉观察
    print("  [Vision] Place 后重新观察:", VISION_TARGET)

    after_observation = perception.observe_object(
        target_name=VISION_TARGET,
        mj_model=model,
        mj_data=data,
    )

    # 5. 视觉验证
    verification = place_verifier.verify(
        observation=after_observation,
        target_position=target_position,
    )

    print("  [Verifier] status =", verification.status)
    print("  [Verifier] message =", verification.message)
    print("  [Verifier] object_position =", verification.object_position)
    print("  [Verifier] target_position =", verification.target_position)
    print("  [Verifier] xy_error =", verification.xy_error)

    return verification.success


# ==================== Qwen Planner ====================

SYSTEM_PROMPT = """
你是 Franka Panda 机械臂的高层任务规划器。

用户会给出最终任务，例如：

“把方块拿起来”
“把方块拿起来放到右边”

你每次只能生成一个下一步动作。

允许动作只有：

pick
place
finish

place 的 target 只能是：

left
right
center

机器人状态包含：

holding:
当前是否抓着 box。

last_action:
上一次执行动作。

last_success:
上一次动作是否成功。

last_target:
上一次 place 的目标。


====================
任务完成规则
====================

1. 如果用户只要求拿起方块：

当 last_action == "pick"
并且 last_success == true

说明任务已经完成。

下一步必须输出 finish。


2. 如果用户要求把方块放到某个位置：

例如“把方块拿起来放到右边”。

当 last_action == "place"
并且 last_success == true
并且 last_target == "right"

说明任务已经完成。

下一步必须输出 finish。

left 和 center 同样处理。


====================
动作规划规则
====================

1. 如果任务需要操作方块，
   holding == null，
   生成 pick。

2. 如果 holding == "box"，
   并且用户要求放置，
   生成对应的 place。

3. 任务完成后必须生成 finish。

4. 如果上一步失败，根据当前状态重新规划。


====================
禁止事项
====================

禁止语言模型生成：

XYZ坐标
关节角
速度
加速度
力矩

物体的位置由视觉系统负责，
不是语言模型负责。


必须只返回 JSON：

{
    "action": "pick/place/finish",
    "target": "left/right/center 或 null"
}
"""


def ask_qwen(goal):
    # 不再把真实 box_position 告诉 Qwen
    state = {
        **robot_state,
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

def execute_action(action, viewer=None):
    action_type = action.get("action")
    target = action.get("target")

    # ---------- Pick ----------
    if action_type == "pick":

        if robot_state["holding"] is not None:
            print("非法动作：已经抓着物体")
            return False

        print("\n  [Vision] 开始定位目标:", VISION_TARGET)

        # ------------------------------------------------------
        # 通过统一的 PerceptionSystem 进行视觉观察
        # ------------------------------------------------------
        #
        # 现在 Agent 不再直接调用：
        #
        # get_object_position(...)
        #
        # 而是得到一个结构化的 ObjectObservation。
        #
        # observation 中包含：
        #
        # target_name
        # success
        # grasp_position
        # error
        # ------------------------------------------------------

        before_observation = perception.observe_object(
            target_name=VISION_TARGET,
            mj_model=model,
            mj_data=data,
        )

        # ------------------------------------------------------
        # 判断视觉感知是否成功
        # ------------------------------------------------------

        if not before_observation.success:

            print(
                "  [Vision] 定位失败:",
                before_observation.error,
            )

            success = False

        else:

            # --------------------------------------------------
            # 从结构化视觉结果中获取抓取位置
            # --------------------------------------------------

            grasp_position = before_observation.grasp_position
            # ------------------------------------------------------
            # 仅用于 VisualPickVerifier 的失败测试
            # ------------------------------------------------------

            if DEBUG_FORCE_PICK_FAILURE:

                grasp_position = (
                    grasp_position
                    + DEBUG_PICK_OFFSET
                )

                print(
                    "  [Debug] 强制制造 Pick 失败"
                )

                print(
                    "  [Debug] 偏移后的抓取位置:",
                    grasp_position,
                )
            print("  [Vision] 定位完成")

            print(
                "  [Vision] 目标:",
                before_observation.target_name,
            )

            print(
                "  [Vision] 世界坐标抓取点:",
                grasp_position,
            )

            # --------------------------------------------------
            # 视觉成功后，才允许进入 Pick Skill
            # --------------------------------------------------

            # ------------------------------------------------------
            # 1. 执行 Pick 动作
            # ------------------------------------------------------

            execution_completed = pick_object_at(
                grasp_position,
                viewer,
            )

            # ------------------------------------------------------
            # 2. 如果机械臂执行阶段本身失败，就不继续视觉验证
            # ------------------------------------------------------

            if not execution_completed:

                print(
                    "  [Pick] 机械臂执行过程失败"
                )

                success = False

            else:

                # --------------------------------------------------
                # 3. Pick 后重新进行一次视觉观察
                # --------------------------------------------------
                #
                # 这是闭环里非常重要的一步：
                #
                # Observe
                #   ↓
                # Act
                #   ↓
                # Observe Again
                # --------------------------------------------------

                print(
                    "  [Vision] Pick 后重新观察目标:",
                    VISION_TARGET,
                )

                after_observation = (
                    perception.observe_object(
                        target_name=VISION_TARGET,
                        mj_model=model,
                        mj_data=data,
                    )
                )

                # --------------------------------------------------
                # 4. 使用纯视觉 Verifier 判断 Pick 是否成功
                # --------------------------------------------------

                verification = pick_verifier.verify(
                    before=before_observation,
                    after=after_observation,
                )

                print(
                    "  [Verifier] status =",
                    verification.status,
                )

                print(
                    "  [Verifier] message =",
                    verification.message,
                )

                print(
                    "  [Verifier] height_change =",
                    verification.height_change,
                )

                # --------------------------------------------------
                # 5. 最终 Pick 成功与否来自视觉 Verifier
                # --------------------------------------------------

                success = verification.success

        robot_state["holding"] = "box" if success else None

    # ---------- Place ----------
    elif action_type == "place":

        if robot_state["holding"] != "box":
            print("非法动作：当前没有抓住 box")
            return False

        success = place_object(target, viewer)

        if success:
            robot_state["holding"] = None

        robot_state["last_target"] = target

    # ---------- Finish ----------
    elif action_type == "finish":
        return "finish"

    else:
        print("非法 action:", action_type)
        return False

    robot_state["last_action"] = action_type
    robot_state["last_success"] = success

    return success


# ==================== Step-by-Step Agent ====================

def run_task(goal, viewer):
    MAX_STEPS = 5

    # 新任务开始，清除上一个任务的执行历史
    reset_task_history()

    print("\n用户任务：", goal)
    print("当前机器人状态：", robot_state)

    for step in range(1, MAX_STEPS + 1):
        print(f"\n========== Step {step} ==========")

        try:
            action = ask_qwen(goal)

        except Exception as e:
            print("Qwen 调用失败：", e)
            return

        print("Qwen 下一步：", action)

        result = execute_action(action, viewer)

        if result == "finish":
            print("\n任务完成")
            return

        print("执行结果：", result)

    print("\n达到最大步骤数，停止任务")


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
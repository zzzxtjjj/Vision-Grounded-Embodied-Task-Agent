import argparse

import mujoco

from panda_qwen_agent import (
    model,
    data,
    run_task,
)

from evaluation.gt_evaluator import GTEvaluator
from evaluation.benchmark_recorder import BenchmarkRunner
from place_regions import get_place_target
import panda_qwen_agent as agent

benchmark_runner = BenchmarkRunner()

task_evaluator = GTEvaluator(
    min_lift_height=0.05,
    max_gripper_distance=0.08,
    max_place_xy_error=0.06,
)


def reset_episode(viewer):
    mujoco.mj_resetDataKeyframe(
        model,
        data,
        model.keyframe("home").id,
    )
    mujoco.mj_forward(model, data)
    agent.state_manager.update_holding(None)
    viewer.sync()


def evaluate_task(task_type, target=None):
    if task_type == "pick":
        result = task_evaluator.evaluate_pick(
            model,
            data,
        )

        return result["success"]

    if task_type == "place":
        if target not in {"left", "center", "right"}:
            raise ValueError(
                f"Invalid benchmark target: {target}"
            )

        target_position = get_place_target(
            target
        )

        result = task_evaluator.evaluate_place(
            model,
            data,
            target_position,
        )

        return result["success"]

    raise ValueError(
        f"Invalid benchmark task type: {task_type}"
    )


def run_benchmark_episode(
    instruction,
    task_type,
    viewer,
    target=None,
):
    reset_episode(viewer)

    task_evaluator.start_episode(
        model,
        data,
    )

    episode_data = run_task(instruction, viewer)

    task_success = evaluate_task(task_type, target)

    benchmark_runner.add_episode(
        task=instruction,
        task_success=task_success,
        vision_records=episode_data["vision_records"],
        action_records=episode_data["action_records"],
        planner_steps=episode_data["planner_steps"],
        invalid_actions=episode_data["invalid_actions"],
        recovery_attempts=episode_data["recovery_attempts"],
        recovery_successes=episode_data["recovery_successes"],
    )

    return task_success


def main():
    parser = argparse.ArgumentParser(
        description="Run a Vision-Grounded Agent benchmark suite."
    )
    parser.add_argument(
        "--suite",
        choices=("main", "complex"),
        default="main",
        help="Benchmark suite to run (default: main).",
    )
    args = parser.parse_args()

    benchmark_cases = {
        "main": BENCHMARK_CASES,
        "complex": COMPLEX_BENCHMARK_CASES,
    }[args.suite]
    result_path = (
        f"results/benchmark_{args.suite}_latest.json"
    )

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            for index, case in enumerate(
                benchmark_cases,
                start=1,
            ):
                print(
                    f"\n========== Benchmark "
                    f"{index}/{len(benchmark_cases)} =========="
                )
                print("Case:", case["name"])
                print("Instruction:", case["instruction"])
                print(
                    "Failure Injection:",
                    case["force_invalid_action"],
                )

                agent.DEBUG_FORCE_INVALID_ACTION = (
                    case["force_invalid_action"]
                )

                try:
                    success = run_benchmark_episode(
                        instruction=case["instruction"],
                        task_type=case["task_type"],
                        viewer=viewer,
                        target=case["target"],
                    )
                finally:
                    agent.DEBUG_FORCE_INVALID_ACTION = False

                print("Task Success =", success)

                # 每完成一轮立即保存，防止中途异常导致数据丢失
                benchmark_runner.save_results(
                    result_path
                )

    finally:
        agent.DEBUG_FORCE_INVALID_ACTION = False

        # 即使中途发生异常，也保存已经完成的 Episode
        if len(benchmark_runner.episodes) > 0:
            benchmark_runner.save_results(
                result_path
            )

    print("\n========== Benchmark Summary ==========")
    print(
        "Episodes:",
        len(benchmark_runner.episodes),
    )
    print(
        "Task Success Rate:",
        benchmark_runner.task_success_rate(),
    )
    print(
        "Vision Success Rate:",
        benchmark_runner.vision_success_rate(),
    )
    print(
        "Mean Localization Error:",
        benchmark_runner.mean_localization_error(),
    )
    print(
        "Pick Success Rate:",
        benchmark_runner.pick_success_rate(),
    )
    print(
        "Place Success Rate:",
        benchmark_runner.place_success_rate(),
    )
    print(
        "Mean Place Error:",
        benchmark_runner.mean_place_error(),
    )
    print(
        "Invalid Action Rate:",
        benchmark_runner.invalid_action_rate(),
    )
    print(
        "Average Planner Steps:",
        benchmark_runner.average_planner_steps(),
    )
    print(
        "Recovery Success Rate:",
        benchmark_runner.recovery_success_rate(),
    )

    print(
        "\nBenchmark results saved to:",
        result_path,
    )

# =================================================================
BENCHMARK_CASES = [
    # ==================== Pick：12 条 ====================
    {
        "name": "pick_01",
        "instruction": "把方块拿起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_02",
        "instruction": "抓起绿色方块",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_03",
        "instruction": "把绿色方块抓起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_04",
        "instruction": "把桌上的方块拿起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_05",
        "instruction": "请拿起这个绿色方块",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_06",
        "instruction": "把这个方块从桌面抓起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_07",
        "instruction": "抓住绿色物体并拿起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_08",
        "instruction": "请把绿色立方体提起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_09",
        "instruction": "把桌面上的绿色立方体抓起",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_10",
        "instruction": "拿起桌上的绿色物体",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_11",
        "instruction": "pick up the green cube",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "pick_12",
        "instruction": "lift the green cube",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },

    # ==================== Place Left：12 条 ====================
    {
        "name": "left_01",
        "instruction": "把方块拿起来放到左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_02",
        "instruction": "把绿色方块放到左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_03",
        "instruction": "把方块移动到左侧",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_04",
        "instruction": "请将绿色方块放在左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_05",
        "instruction": "把桌上的方块搬到左侧区域",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_06",
        "instruction": "将绿色立方体移动到左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_07",
        "instruction": "请把这个物体放到左侧",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_08",
        "instruction": "把方块抓起来，然后放左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_09",
        "instruction": "把绿色物体搬到左边去",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_10",
        "instruction": "绿色方块放左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_11",
        "instruction": "move the green cube to the left",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "left_12",
        "instruction": "place the green cube on the left",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },

    # ==================== Place Center：12 条 ====================
    {
        "name": "center_01",
        "instruction": "把方块拿起来放到中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_02",
        "instruction": "把绿色方块放到中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_03",
        "instruction": "将方块移动到中央区域",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_04",
        "instruction": "请把绿色方块放在中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_05",
        "instruction": "把桌上的方块搬到中心",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_06",
        "instruction": "把绿色立方体移到中央",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_07",
        "instruction": "请把这个物体放到中间区域",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_08",
        "instruction": "抓起方块，然后放到正中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_09",
        "instruction": "把绿色物体搬到中间去",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_10",
        "instruction": "绿色方块放中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_11",
        "instruction": "move the green cube to the center",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "center_12",
        "instruction": "place the green cube in the center",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },

    # ==================== Place Right：12 条 ====================
    {
        "name": "right_01",
        "instruction": "把方块拿起来放到右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_02",
        "instruction": "把绿色方块放到右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_03",
        "instruction": "将方块移动到右侧",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_04",
        "instruction": "请把绿色方块放在右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_05",
        "instruction": "把桌上的方块搬到右侧区域",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_06",
        "instruction": "将绿色立方体移动到右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_07",
        "instruction": "请把这个物体放到右侧",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_08",
        "instruction": "把方块抓起来，然后放右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_09",
        "instruction": "把绿色物体搬到右边去",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_10",
        "instruction": "绿色方块放右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_11",
        "instruction": "move the green cube to the right",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "right_12",
        "instruction": "place the green cube on the right",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },

    # ==================== Recovery：12 条 ====================
    {
        "name": "recovery_pick_01",
        "instruction": "把方块拿起来",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": True,
    },
    {
        "name": "recovery_pick_02",
        "instruction": "抓起绿色方块",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": True,
    },
    {
        "name": "recovery_left_01",
        "instruction": "把方块放到左边",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_left_02",
        "instruction": "把绿色物体移到左侧",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_center_01",
        "instruction": "把方块放到中间",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_center_02",
        "instruction": "把绿色物体移到中央",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_right_01",
        "instruction": "把方块放到右边",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_right_02",
        "instruction": "把绿色物体移到右侧",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_pick_en",
        "instruction": "pick up the green cube",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": True,
    },
    {
        "name": "recovery_left_en",
        "instruction": "move the cube to the left",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_center_en",
        "instruction": "move the cube to the center",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": True,
    },
    {
        "name": "recovery_right_en",
        "instruction": "move the cube to the right",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": True,
    },
]

COMPLEX_BENCHMARK_CASES = [
    # ==================== Complex Pick：6 条 ====================
    {
        "name": "complex_pick_01",
        "instruction": "请找到桌面上的绿色立方体，把它从当前的位置抓起来并保持在夹爪中。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "complex_pick_02",
        "instruction": "现在不需要移动到其他区域，只需要把桌上的绿色方块拿起来。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "complex_pick_03",
        "instruction": "先定位绿色物体，然后将这个立方体从桌面抓起。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "complex_pick_04",
        "instruction": "完成一个抓取任务：目标是桌面上的绿色方块，抓起后不用放置。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "complex_pick_05",
        "instruction": "Locate the green cube on the table and pick it up, but do not place it anywhere.",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },
    {
        "name": "complex_pick_06",
        "instruction": "把那个绿色的立方体拿起来就可以了，暂时不要把它移动到左边、中间或者右边。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": False,
    },

    # ==================== Complex Left：6 条 ====================
    {
        "name": "complex_left_01",
        "instruction": "请先把桌上的绿色方块抓起来，随后把它移动到工作区域的左侧。",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "complex_left_02",
        "instruction": "目标不是中间也不是右边，请把绿色立方体最终放到左侧区域。",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "complex_left_03",
        "instruction": "完成抓取以后，把这个物体送到左边，最后让它停留在左侧位置。",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "complex_left_04",
        "instruction": "桌上有一个绿色方块，请拿起它，然后选择左侧作为最终放置区域。",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "complex_left_05",
        "instruction": "Pick up the green cube first, then move it to the left side of the workspace.",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },
    {
        "name": "complex_left_06",
        "instruction": "Grab the cube and place it on the left, not in the center or on the right.",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": False,
    },

    # ==================== Complex Center：6 条 ====================
    {
        "name": "complex_center_01",
        "instruction": "请抓起绿色方块，并把它从当前位置移动到工作区域正中央。",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "complex_center_02",
        "instruction": "不要把这个方块留在左边或右边，最终应该把它放到中间区域。",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "complex_center_03",
        "instruction": "先完成对绿色立方体的抓取，之后把它放置在中央位置。",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "complex_center_04",
        "instruction": "把桌上的绿色物体拿起来，目标位置选择 center，也就是中间区域。",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "complex_center_05",
        "instruction": "Pick the green object up and place it in the middle of the workspace.",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },
    {
        "name": "complex_center_06",
        "instruction": "Move the cube to the center after picking it up; the left and right regions are not the target.",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": False,
    },

    # ==================== Complex Right：6 条 ====================
    {
        "name": "complex_right_01",
        "instruction": "找到绿色立方体并抓住它，之后把它移动到工作空间右侧。",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "complex_right_02",
        "instruction": "最终目标是右侧区域，请先抓取桌面上的绿色方块再完成放置。",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "complex_right_03",
        "instruction": "不要放到左边，也不要停在中间，把这个绿色物体最终送到右边。",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "complex_right_04",
        "instruction": "执行一次抓取和放置：抓取绿色立方体，放置目标选择右侧。",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "complex_right_05",
        "instruction": "After locating and picking up the green cube, place it on the right side.",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },
    {
        "name": "complex_right_06",
        "instruction": "The final destination is the right region. Pick up the cube first and move it there.",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": False,
    },

    # ==================== Complex Recovery：6 条 ====================
    {
        "name": "complex_recovery_pick_01",
        "instruction": "请定位桌上的绿色立方体并将它抓起来，抓取完成后保持拿住即可。",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": True,
    },
    {
        "name": "complex_recovery_pick_02",
        "instruction": "Locate the green cube, pick it up from the table, and keep holding it.",
        "task_type": "pick",
        "target": None,
        "force_invalid_action": True,
    },
    {
        "name": "complex_recovery_left",
        "instruction": "先抓住绿色方块，最终目标不是中央和右侧，而是把它放到左侧区域。",
        "task_type": "place",
        "target": "left",
        "force_invalid_action": True,
    },
    {
        "name": "complex_recovery_center",
        "instruction": "完成抓取后，把绿色立方体移动到中央区域，不要选择左右两侧。",
        "task_type": "place",
        "target": "center",
        "force_invalid_action": True,
    },
    {
        "name": "complex_recovery_right",
        "instruction": "请拿起桌上的绿色物体，并最终把它放到右侧，而不是左侧或中央。",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": True,
    },
    {
        "name": "complex_recovery_right_en",
        "instruction": "Pick up the green cube and place it on the right; the center and left regions are not the destination.",
        "task_type": "place",
        "target": "right",
        "force_invalid_action": True,
    },
]

# ============================================================
if __name__ == "__main__":
    main()

import numpy as np


class GTEvaluator:

    def __init__(
        self,
        min_lift_height=0.05,
        max_gripper_distance=0.08,
        max_place_xy_error=0.06
    ):
        self.min_lift_height = float(min_lift_height)
        self.max_gripper_distance = float(max_gripper_distance)
        self.max_place_xy_error = float(max_place_xy_error)
        self.initial_box_position = None

    def start_episode(self, mj_model, mj_data):
        box_id = mj_model.body("box").id

        self.initial_box_position = (
            mj_data.xpos[box_id].copy()
        )

    def evaluate_pick(self, mj_model, mj_data):
        if self.initial_box_position is None:
            raise RuntimeError(
                "必须先调用 start_episode()"
            )

        box_id = mj_model.body("box").id
        gripper_id = mj_model.site("gripper").id

        box_position = mj_data.xpos[box_id].copy()
        gripper_position = mj_data.site_xpos[gripper_id].copy()

        lift_height = box_position[2] - self.initial_box_position[2]

        gripper_distance = np.linalg.norm(box_position - gripper_position)

        lifted = lift_height >= self.min_lift_height

        near_gripper = gripper_distance <= self.max_gripper_distance 

        success = lifted and near_gripper

        return {
            "success": bool(success),
            "box_position": box_position.copy(),
            "gripper_position": gripper_position.copy(),
            "lift_height": float(lift_height),
            "gripper_distance": float(gripper_distance),
        }


    def evaluate_place(
        self,
        mj_model,
        mj_data,
        target_position,
    ):
        box_id = mj_model.body("box").id

        box_position = mj_data.xpos[box_id].copy()

        target_position = np.asarray(
            target_position,
            dtype=float,
        )

        xy_error = np.linalg.norm(box_position[:2] - target_position[:2])
        success = xy_error <= self.max_place_xy_error

        return {
            "success": bool(success),
            "box_position": box_position.copy(),
            "target_position": target_position.copy(),
            "xy_error": float(xy_error),
        }

    def evaluate_vision(
        self,
        mj_model,
        mj_data,
        vision_position,
    ):
        box_id = mj_model.body("box").id

        gt_position = (
            mj_data.xpos[box_id].copy()
        )

        vision_position = np.asarray(
            vision_position,
            dtype=float,
        )

        localization_error = np.linalg.norm(gt_position - vision_position)

        return {
            "success": True,
            "vision_position": vision_position.copy(),
            "gt_position": gt_position.copy(),
            "localization_error": float(localization_error),
        }
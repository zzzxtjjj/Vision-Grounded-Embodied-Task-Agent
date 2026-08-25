from dataclasses import dataclass
from typing import Optional

import numpy as np

from vision.perception import ObjectObservation


@dataclass
class PickVerificationResult:
    """
    Pick Skill 的视觉验证结果。

    这个结果完全来自视觉观察，
    不读取仿真器中的物体 Ground Truth。
    """

    # 是否确认 Pick 成功
    success: bool

    # 验证结果的状态名称
    #
    # 目前可能有：
    #
    # PICK_SUCCESS
    # PICK_FAILED
    # PERCEPTION_FAILED
    status: str

    # 视觉估计出的物体高度变化
    #
    # 单位：米
    #
    # 如果无法完成视觉比较，则为 None
    height_change: Optional[float]

    # 给 Agent / 日志看的详细说明
    message: str

    def as_dict(self):
        """
        转换成普通字典。

        后面可以方便地记录到：
        - Agent State
        - Execution History
        - Benchmark
        """

        return {
            "success": self.success,
            "status": self.status,
            "height_change": self.height_change,
            "message": self.message,
        }


class VisualPickVerifier:
    """
    使用执行前后的视觉结果判断 Pick 是否成功。

    当前第一版采用一个非常直观的规则：

        Pick 前物体视觉高度
                ↓
              Pick
                ↓
        Pick 后物体视觉高度
                ↓
           计算高度变化

    如果物体明显升高：

        PICK_SUCCESS

    否则：

        PICK_FAILED

    注意：

    这里使用的是视觉系统估计的 grasp_position，
    而不是 simulator Ground Truth。
    """

    def __init__(
        self,
        min_lift_height=0.05,
    ):
        """
        参数
        ----------
        min_lift_height:

            判断物体被成功抬起所要求的最小高度变化。

            默认：
                0.05 m

            也就是：
                5 cm
        """

        if min_lift_height <= 0:
            raise ValueError(
                "min_lift_height 必须大于 0"
            )

        self.min_lift_height = float(
            min_lift_height
        )

    def verify(
        self,
        before: ObjectObservation,
        after: ObjectObservation,
    ):
        """
        根据 Pick 前后的两次视觉观察验证抓取结果。

        参数
        ----------
        before:
            执行 Pick 之前的视觉观察。

        after:
            执行 Pick 之后重新拍摄得到的视觉观察。

        返回
        ----------
        PickVerificationResult
        """

        # ----------------------------------------------------
        # 1. 检查 Pick 前的视觉是否成功
        # ----------------------------------------------------

        if not before.success:

            return PickVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",
                height_change=None,
                message=(
                    "Pick 前视觉感知失败，"
                    f"无法进行视觉验证：{before.error}"
                ),
            )

        # ----------------------------------------------------
        # 2. 检查 Pick 后的视觉是否成功
        # ----------------------------------------------------

        if not after.success:

            # 非常重要：
            #
            # Pick 后看不到目标，
            # 不能直接认为抓取成功。
            #
            # 目标有可能：
            #
            # 1. 被夹爪挡住
            # 2. 掉出相机视野
            # 3. Florence-2 分割失败
            #
            # 所以当前阶段统一认为：
            #
            # 无法完成可靠验证。
            #

            return PickVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",
                height_change=None,
                message=(
                    "Pick 后视觉感知失败，"
                    f"无法确认抓取是否成功：{after.error}"
                ),
            )

        # ----------------------------------------------------
        # 3. 检查两次观察是不是同一个目标
        # ----------------------------------------------------

        if before.target_name != after.target_name:

            return PickVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",
                height_change=None,
                message=(
                    "Pick 前后观察的目标名称不一致："
                    f"{before.target_name} != "
                    f"{after.target_name}"
                ),
            )

        # ----------------------------------------------------
        # 4. 获取视觉估计的三维位置
        # ----------------------------------------------------

        before_position = np.asarray(
            before.grasp_position,
            dtype=np.float64,
        )

        after_position = np.asarray(
            after.grasp_position,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # 5. 基本数据检查
        # ----------------------------------------------------

        if (
            before_position.shape != (3,)
            or after_position.shape != (3,)
        ):

            return PickVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",
                height_change=None,
                message=(
                    "视觉位置格式错误，"
                    "无法进行 Pick 验证"
                ),
            )

        if (
            not np.all(np.isfinite(before_position))
            or not np.all(np.isfinite(after_position))
        ):

            return PickVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",
                height_change=None,
                message=(
                    "视觉位置包含 NaN 或 Inf，"
                    "无法进行 Pick 验证"
                ),
            )

        # ----------------------------------------------------
        # 6. 计算物体视觉高度变化
        # ----------------------------------------------------
        #
        # grasp_position 格式：
        #
        # [x, y, z]
        #
        # 所以索引 2 就是世界坐标中的 z。
        # ----------------------------------------------------

        before_z = float(
            before_position[2]
        )

        after_z = float(
            after_position[2]
        )

        height_change = (
            after_z - before_z
        )

        # ----------------------------------------------------
        # 7. 根据高度变化判断 Pick
        # ----------------------------------------------------

        if height_change >= self.min_lift_height:

            return PickVerificationResult(
                success=True,
                status="PICK_SUCCESS",
                height_change=height_change,
                message=(
                    "视觉检测到目标物体明显升高，"
                    "判断 Pick 成功"
                ),
            )

        return PickVerificationResult(
            success=False,
            status="PICK_FAILED",
            height_change=height_change,
            message=(
                "视觉检测到目标物体高度变化不足，"
                "判断 Pick 失败"
            ),
        )



# ====================================================================
# VisualPlaceVerifier

# Place 视觉验证结果
# ============================================================

@dataclass
class PlaceVerificationResult:
    """
    Place Skill 的纯视觉验证结果。

    这里不读取仿真器中的物体 Ground Truth。

    判断依据：

        Place 执行完成
              ↓
        重新通过视觉定位物体
              ↓
        得到视觉估计 XY
              ↓
        与固定目标区域 XY 比较
    """

    # 是否判断 Place 成功
    success: bool

    # 当前验证状态
    #
    # 可能值：
    #
    # PLACE_SUCCESS
    # PLACE_FAILED
    # PERCEPTION_FAILED
    status: str

    # 视觉估计物体位置
    #
    # shape: (3,)
    #
    # 如果视觉失败则为 None
    object_position: Optional[np.ndarray]

    # 目标放置位置
    #
    # shape: (3,)
    target_position: np.ndarray

    # XY 平面上的距离误差
    #
    # 单位：米
    #
    # 视觉失败时为 None
    xy_error: Optional[float]

    # 给日志 / Agent 使用的说明
    message: str

    def as_dict(self):
        """
        转换成普通字典。

        后续可以用于：
        - Agent State
        - Execution History
        - Benchmark
        """

        return {
            "success": self.success,
            "status": self.status,

            "object_position": (
                self.object_position.tolist()
                if self.object_position is not None
                else None
            ),

            "target_position": (
                self.target_position.tolist()
            ),

            "xy_error": self.xy_error,
            "message": self.message,
        }


# ============================================================
# Pure Vision Place Verifier
# ============================================================

class VisualPlaceVerifier:
    """
    使用视觉结果判断 Place 是否成功。

    当前 V1 规则：

        视觉估计物体最终位置
                ↓
        与 Place Target 比较
                ↓
        计算 XY 平面距离
                ↓
        如果距离 <= tolerance
                ↓
        PLACE_SUCCESS

    为什么只比较 XY？

    因为当前任务定义是：

        把方块放到 left / center / right 区域

    主要关注方块最终落在哪一个桌面区域。

    当前 V1 暂时不对最终 Z 高度做复杂判断。
    """

    def __init__(
        self,
        xy_tolerance=0.06,
    ):
        """
        参数
        ----------
        xy_tolerance:

            Place 成功允许的最大 XY 距离误差。

            默认：
                0.06 m

            即：
                6 cm
        """

        if xy_tolerance <= 0:
            raise ValueError(
                "xy_tolerance 必须大于 0"
            )

        self.xy_tolerance = float(
            xy_tolerance
        )

    def verify(
        self,
        observation: ObjectObservation,
        target_position,
    ):
        """
        根据 Place 后的视觉观察判断放置是否成功。

        参数
        ----------
        observation:

            Place 完成之后重新得到的
            ObjectObservation。

        target_position:

            固定工作空间中的目标位置。

            shape:
                (3,)

            例如：

                right
                [0.70, -0.12, 0.035]

        返回
        ----------
        PlaceVerificationResult
        """

        # ----------------------------------------------------
        # 1. 检查目标位置
        # ----------------------------------------------------

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        if target_position.shape != (3,):

            raise ValueError(
                "target_position 必须是 shape (3,) 的 XYZ 坐标，"
                f"当前得到：{target_position.shape}"
            )

        if not np.all(
            np.isfinite(target_position)
        ):

            raise ValueError(
                "target_position 中包含 NaN 或 Inf"
            )

        # ----------------------------------------------------
        # 2. 检查 Place 后视觉是否成功
        # ----------------------------------------------------

        if not observation.success:

            return PlaceVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",

                object_position=None,
                target_position=target_position,

                xy_error=None,

                message=(
                    "Place 后视觉感知失败，"
                    f"无法判断物体是否进入目标区域："
                    f"{observation.error}"
                ),
            )

        # ----------------------------------------------------
        # 3. 读取视觉估计的物体位置
        # ----------------------------------------------------

        object_position = np.asarray(
            observation.grasp_position,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # 4. 检查视觉位置是否合法
        # ----------------------------------------------------

        if object_position.shape != (3,):

            return PlaceVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",

                object_position=None,
                target_position=target_position,

                xy_error=None,

                message=(
                    "视觉返回的位置格式错误，"
                    "无法完成 Place 验证"
                ),
            )

        if not np.all(
            np.isfinite(object_position)
        ):

            return PlaceVerificationResult(
                success=False,
                status="PERCEPTION_FAILED",

                object_position=None,
                target_position=target_position,

                xy_error=None,

                message=(
                    "视觉返回的位置包含 NaN 或 Inf，"
                    "无法完成 Place 验证"
                ),
            )

        # ----------------------------------------------------
        # 5. 计算 XY 平面距离
        # ----------------------------------------------------
        #
        # object_position:
        #
        # [object_x, object_y, object_z]
        #
        # target_position:
        #
        # [target_x, target_y, target_z]
        #
        # 当前 Place V1 只比较：
        #
        # X 和 Y
        #
        # 所以使用 [:2]
        # ----------------------------------------------------

        xy_error = float(
            np.linalg.norm(
                object_position[:2]
                - target_position[:2]
            )
        )

        # ----------------------------------------------------
        # 6. 判断是否进入目标区域
        # ----------------------------------------------------

        if xy_error <= self.xy_tolerance:

            return PlaceVerificationResult(
                success=True,
                status="PLACE_SUCCESS",

                object_position=object_position,
                target_position=target_position,

                xy_error=xy_error,

                message=(
                    "视觉检测到目标物体位于指定放置区域内，"
                    "判断 Place 成功"
                ),
            )

        # ----------------------------------------------------
        # 7. 没有进入目标区域
        # ----------------------------------------------------

        return PlaceVerificationResult(
            success=False,
            status="PLACE_FAILED",

            object_position=object_position,
            target_position=target_position,

            xy_error=xy_error,

            message=(
                "视觉检测到目标物体没有进入指定放置区域，"
                "判断 Place 失败"
            ),
        )
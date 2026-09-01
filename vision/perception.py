from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ObjectObservation:
    """视觉系统返回的结构化目标观察结果。"""

    target_name: str
    success: bool
    grasp_position: Optional[np.ndarray]
    error: Optional[str] = None

    # 新增：用于判断当前分割结果是否可信
    mask_area: Optional[int] = None
    mask_ratio: Optional[float] = None

    def as_dict(self):
        return {
            "target_name": self.target_name,
            "success": self.success,
            "grasp_position": (
                self.grasp_position.tolist()
                if self.grasp_position is not None
                else None
            ),
            "error": self.error,
            "mask_area": self.mask_area,
            "mask_ratio": self.mask_ratio,
        }


class PerceptionSystem:
    """
    Pure Vision Runtime 的统一感知接口。

    当前流程：

    RGB-D
      ↓
    Florence-2
      ↓
    Mask
      ↓
    可信度检查
      ↓
    Depth + Point Cloud
      ↓
    3D grasp_position
    """

    def __init__(self, max_mask_ratio=0.10, max_object_height=0.10):
        # 如果目标 Mask 占整张图片超过 10%，
        # 当前小方块场景中认为分割结果高度可疑。
        self.max_mask_ratio = float(max_mask_ratio)
        # 最大物品高度阈值
        self.max_object_height = float(max_object_height)

    def observe_object(self, target_name, mj_model, mj_data):

        if not isinstance(target_name, str):
            return ObjectObservation(
                str(target_name), False, None,
                error="target_name 必须是字符串"
            )

        target_name = target_name.strip()

        if not target_name:
            return ObjectObservation(
                target_name, False, None,
                error="target_name 不能为空"
            )

        try:
            # 延迟导入，只有真正使用视觉时才加载 Florence-2
            from vision_system import (
                capture_rgbd,
                segment_object,
                save_debug_results,
                estimate_grasp_position,
            )

            # 1. 获取实时 RGB-D
            print("[Vision] 实时拍摄 RGB-D")
            rgb, depth = capture_rgbd(mj_model, mj_data)

            print("[Vision] RGB shape:", rgb.shape)
            print("[Vision] Depth shape:", depth.shape)

            # 2. Florence-2 分割
            print("[Vision] Florence-2 正在分割:", target_name)

            mask, image, mask_image, polygons = segment_object(
                rgb,
                target_name,
            )

            print("[Vision] 分割完成")

            # 3. 检查 Mask 是否合理
            mask_area = int(np.sum(mask))
            mask_ratio = float(mask_area / mask.size)

            print("[Vision] Mask像素数量:", mask_area)
            print("[Vision] Mask占比:", round(mask_ratio, 4))

            if mask_ratio > self.max_mask_ratio:
                return ObjectObservation(
                    target_name=target_name,
                    success=False,
                    grasp_position=None,
                    error=(
                        "视觉分割区域异常过大，"
                        f"mask_ratio={mask_ratio:.4f}"
                    ),
                    mask_area=mask_area,
                    mask_ratio=mask_ratio,
                )

            # 4. 保存调试结果
            save_debug_results(
                rgb,
                depth,
                image,
                mask,
                mask_image,
                polygons,
            )

            # 5. Mask + Depth → Point Cloud → 3D位置
            print("[Vision] 开始 Depth + Point Cloud 三维定位")

            grasp_position, geometry_info = estimate_grasp_position(
                mask,
                depth,
                mj_model,
                mj_data,
            )

            if geometry_info["object_height"] > self.max_object_height:
                return ObjectObservation(
                    target_name=target_name,
                    success=False,
                    grasp_position=None,
                    error=(
                        "目标三维高度异常，"
                        f"object_height={geometry_info['object_height']:.4f}"
                    ),
                    mask_area=mask_area,
                    mask_ratio=mask_ratio,
            )

            grasp_position = np.asarray(
                grasp_position,
                dtype=np.float64,
            )

            # 6. 检查三维位置是否合法
            if grasp_position.shape != (3,):
                raise RuntimeError(
                    f"grasp_position shape错误：{grasp_position.shape}"
                )

            if not np.all(np.isfinite(grasp_position)):
                raise RuntimeError(
                    "grasp_position 包含 NaN 或 Inf"
                )

            # 7. 返回可信视觉结果
            return ObjectObservation(
                target_name=target_name,
                success=True,
                grasp_position=grasp_position,
                error=None,
                mask_area=mask_area,
                mask_ratio=mask_ratio,
            )

        except Exception as e:
            return ObjectObservation(
                target_name=target_name,
                success=False,
                grasp_position=None,
                error=str(e),
            )

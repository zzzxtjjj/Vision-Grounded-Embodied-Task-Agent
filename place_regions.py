import numpy as np


# ============================================================
# 固定工作空间放置区域
# ============================================================
#
# 这些位置属于任务环境的预先标定配置，
# 不是运行过程中从 simulator 读取的物体 Ground Truth。
#
# 设计原则：
#
#   left / center / right
#
# 必须始终对应工作台中的固定区域，
# 不能根据当前方块真实位置动态生成。
#
# 坐标系：
# MuJoCo World Frame
#
# 格式：
# [x, y, z]
#
# 当前场景中 Panda 的主要操作区域位于 x ≈ 0.70。
#
# left / right 的方向继续沿用原项目约定：
#
#   left  -> +Y
#   right -> -Y
#
# z = 0.035 m 是当前 Place Skill 使用的
# 放置控制高度。
# ============================================================

PLACE_REGIONS = {
    "left": np.array(
        [0.70, 0.12, 0.035],
        dtype=np.float64,
    ),

    "center": np.array(
        [0.70, 0.00, 0.035],
        dtype=np.float64,
    ),

    "right": np.array(
        [0.70, -0.12, 0.035],
        dtype=np.float64,
    ),
}


def get_place_target(target_name):
    """
    根据语义目标名称获取固定的三维放置位置。

    参数
    ----------
    target_name:
        放置区域名称。

        当前支持：

        "left"
        "center"
        "right"

    返回
    ----------
    np.ndarray
        shape = (3,)

        格式：

        [x, y, z]

        坐标系：
        MuJoCo World Frame

    注意
    ----------
    返回的是 copy。

    这样外部代码即使修改返回值，
    也不会意外修改 PLACE_REGIONS 中保存的
    原始工作空间配置。
    """

    # --------------------------------------------------------
    # 1. 检查输入是不是字符串
    # --------------------------------------------------------

    if not isinstance(target_name, str):
        raise ValueError(
            "target_name 必须是字符串"
        )

    # 去掉首尾空格，并统一转成小写
    target_name = (
        target_name
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # 2. 检查目标区域是否存在
    # --------------------------------------------------------

    if target_name not in PLACE_REGIONS:

        valid_targets = ", ".join(
            PLACE_REGIONS.keys()
        )

        raise ValueError(
            f"未知放置区域：{target_name}。"
            f"当前支持：{valid_targets}"
        )

    # --------------------------------------------------------
    # 3. 返回固定工作空间目标
    # --------------------------------------------------------
    #
    # 这里绝对没有：
    #
    # data.xpos[box_id]
    #
    # 也没有：
    #
    # get_box_pos()
    #
    # 所以目标位置与当前方块的真实位置无关。
    # --------------------------------------------------------

    return PLACE_REGIONS[
        target_name
    ].copy()


# ============================================================
# 简单测试
# ============================================================

if __name__ == "__main__":

    for name in [
        "left",
        "center",
        "right",
    ]:

        target = get_place_target(name)

        print(
            f"{name:>6} -> {target}"
        )
import numpy as np
import mujoco
import torch

from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration


# ==================== 配置 ====================

MODEL_ID = "florence-community/Florence-2-base-ft"

CAMERA_NAME = "vision_camera"
WIDTH = 640
HEIGHT = 480

# 根据语言描述，把对应目标分割出来
TASK_PROMPT = "<REFERRING_EXPRESSION_SEGMENTATION>"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32


# ==================== Florence-2 ====================
# 这里只加载一次模型，不会每次抓取都重新加载

print("[Vision] Florence-2 device:", device)
print("[Vision] 正在加载 Florence-2...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

florence_model = Florence2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch_dtype
).to(device)

print("[Vision] Florence-2 加载完成")


# ==================== 实时 RGB-D ====================

def capture_rgbd(mj_model, mj_data):
    """
    从当前 MuJoCo 场景实时获取 RGB 和 Depth。
    """

    mujoco.mj_forward(mj_model, mj_data)

    # 渲染器 把MuJoCo 3D世界拍成图片
    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

    # RGB
    renderer.update_scene(mj_data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()

    # Depth
    renderer.enable_depth_rendering()
    renderer.update_scene(mj_data, camera=CAMERA_NAME)
    depth = renderer.render().copy()

    renderer.close()

    return rgb, depth


# ==================== Florence-2 分割 ====================

def segment_object(rgb, target_name):
    """
    输入:
        RGB图像
        目标文字，例如 "green cube"

    输出:
        bool mask
    """

    image = Image.fromarray(rgb).convert("RGB")

    prompt = TASK_PROMPT + target_name

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch_dtype)

    # 推理模式
    with torch.inference_mode():
        generated_ids = florence_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            # 关闭随机采样
            do_sample=False,
            num_beams=3
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=TASK_PROMPT,
        image_size=(image.width, image.height)
    )

    if TASK_PROMPT not in parsed_answer:
        raise RuntimeError("Florence-2 没有返回分割结果")

    segmentation_result = parsed_answer[TASK_PROMPT]
    polygons = segmentation_result.get("polygons", [])

    if not polygons:
        raise RuntimeError(f"没有找到目标: {target_name}")

    # Polygon -> Mask
    mask_image = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask_image)

    for object_polygons in polygons:
        for polygon in object_polygons:
            if len(polygon) < 6:
                continue

            points = []

            for i in range(0, len(polygon), 2):
                points.append((polygon[i], polygon[i + 1]))

            mask_draw.polygon(points, fill=255)

    mask = np.array(mask_image) > 0

    if np.sum(mask) == 0:
        raise RuntimeError(f"目标 {target_name} 的 Mask 为空")

    return mask, image, mask_image, polygons


# ==================== Debug结果保存 ====================

def save_debug_results(rgb, depth, image, mask, mask_image, polygons):
    """
    保存结果只是方便我们检查。
    后面的抓取算法不会再读取这些文件。
    """

    Image.fromarray(rgb).save("camera_rgb.png")
    np.save("camera_depth.npy", depth)

    mask_image.save("segmentation_mask.png")
    np.save("segmentation_mask.npy", mask)

    overlay = image.copy()
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    for object_polygons in polygons:
        for polygon in object_polygons:
            if len(polygon) < 6:
                continue

            points = []

            for i in range(0, len(polygon), 2):
                points.append((polygon[i], polygon[i + 1]))

            overlay_draw.polygon(
                points,
                fill=(255, 0, 0, 100),
                outline=(255, 0, 0, 255)
            )

    overlay.save("camera_segmentation.png")


# ==================== Mask + Depth -> Point Cloud ====================

def estimate_grasp_position(mask, depth, mj_model, mj_data):
    """
    根据当前 Mask 和 Depth 计算世界坐标中的抓取点。
    """

    if mask.shape != depth.shape:
        raise RuntimeError(
            f"Mask 和 Depth 尺寸不同: mask={mask.shape}, depth={depth.shape}"
        )

    # --------------------
    # 1. 获取目标Depth
    # --------------------
    """
    depth =
    [
    [1.5, 1.6, 1.7],
    [1.4, 1.1, 1.8]
    ]
    mask =
    [
    [False, False, False],
    [False, True, False]
    ]
    只取mask为True的depth值
    """
    object_depth = depth[mask]

    object_depth = object_depth[
        np.isfinite(object_depth) & (object_depth > 0)
    ]

    if len(object_depth) < 20:
        raise RuntimeError("有效目标 Depth 太少，无法进行三维定位")

    # --------------------
    # 2. IQR离群值过滤 Interquartile Range，四分位距
    # --------------------

    q1 = np.percentile(object_depth, 25)
    q3 = np.percentile(object_depth, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    valid_depth_mask = (
        mask
        & np.isfinite(depth)
        & (depth > 0)
        & (depth >= lower_bound)
        & (depth <= upper_bound)
    )

    valid_count = np.sum(valid_depth_mask)

    if valid_count < 20:
        raise RuntimeError("Depth过滤后有效点太少")

    print("[Vision] Mask像素数量:", np.sum(mask))
    print("[Vision] Depth有效像素:", valid_count)

    # --------------------
    # 3. 当前相机外参 Camera 在世界什么位置、朝哪个方向
    # --------------------

    mujoco.mj_forward(mj_model, mj_data)

    camera_id = mj_model.camera(CAMERA_NAME).id

    camera_pos = mj_data.cam_xpos[camera_id].copy()
    camera_rot = mj_data.cam_xmat[camera_id].reshape(3, 3).copy()

    # --------------------
    # 4. 当前相机内参
    # --------------------

    height, width = depth.shape

    fovy_deg = float(mj_model.cam_fovy[camera_id])
    fovy = np.deg2rad(fovy_deg)

    # 像素单位焦距
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy

    # 主点
    cx = width / 2.0
    cy = height / 2.0

    # --------------------
    # 5. Mask像素坐标
    # --------------------

    v_pixels, u_pixels = np.where(valid_depth_mask)

    Z = depth[v_pixels, u_pixels].astype(np.float64)

    # --------------------
    # 6. Pixel -> Camera 3D
    # --------------------

    X_cam = (u_pixels - cx) * Z / fx
    Y_cam = -(v_pixels - cy) * Z / fy
    Z_cam = -Z

    # 方块不是一个完整实体，而是由几百个空间小点组成，点云
    points_camera = np.column_stack([
        X_cam,
        Y_cam,
        Z_cam
    ])

    # --------------------
    # 7. Camera -> World
    # --------------------

    points_world = (camera_rot @ points_camera.T).T + camera_pos

    print("[Vision] Point Cloud:", points_world.shape)

    # --------------------
    # 8. 找物体顶部
    # --------------------
    # 取所有点的 z
    z_values = points_world[:, 2]

    top_threshold = np.percentile(z_values, 80)
    top_points = points_world[z_values >= top_threshold]

    if len(top_points) == 0:
        raise RuntimeError("没有找到物体顶部点云")

    top_center = np.median(top_points, axis=0)

    # --------------------
    # 9. 估计抓取高度
    # --------------------

    top_z = np.median(top_points[:, 2])
    bottom_z = np.percentile(z_values, 10)

    object_center_z = (top_z + bottom_z) / 2.0

    # +5 mm 是夹爪抓取控制偏移
    grasp_position = np.array([
        top_center[0],
        top_center[1],
        object_center_z + 0.005
    ])

    object_height = top_z - bottom_z

    geometry_info = {
        "top_z": float(top_z),
        "bottom_z": float(bottom_z),
        "object_height": float(object_height)
    }

    print("[Vision] top_z:", top_z)
    print("[Vision] bottom_z:", bottom_z)
    print("[Vision] object_height:", object_height)
    print("[Vision] grasp_position:", grasp_position)

    return grasp_position, geometry_info

import os
import cv2
import numpy as np
from pathlib import Path
import re

# ===================== 配置 =====================
LEFT_DIR = "left_dis"  # 左图像文件夹
RIGHT_DIR = "right_dis"  # 右图像文件夹
OUT_DISP_DIR = "disparities"

DISPLAY_TIME = 1.5  # 每对图显示时间（秒）

# ===================== SGBM 参数 =====================
minDisparity = 0
numDisparities = 16 * 9  # 覆盖更大深度范围
blockSize = 3  # 增大块大小
P1 = 8 * 3 * blockSize ** 2
P2 = 32 * 3 * blockSize ** 2
disp12MaxDiff = 1
preFilterCap = 31
uniquenessRatio = 10
speckleWindowSize = 50
speckleRange = 10
mode = cv2.STEREO_SGBM_MODE_HH

# ===================== 远点过滤参数 =====================
MAX_DISPARITY_THRESHOLD = 200.0  # 最大视差阈值，小于此值的视为远点（可调整）
MIN_DISPARITY_THRESHOLD = 50.0  # 最小视差阈值，小于此值的视为无效点


# ===================== 工具函数 =====================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def extract_number(filename):
    """从文件名中提取数字"""
    # 匹配文件名中的数字
    numbers = re.findall(r'\d+', str(filename))
    return int(numbers[-1]) if numbers else -1


def get_image_pairs_by_number(left_dir, right_dir):
    """根据数字匹配左右图像对"""
    left_path = Path(left_dir)
    right_path = Path(right_dir)

    # 支持多种图像格式
    img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    # 获取所有图像文件并按数字排序
    left_imgs = sorted(
        [f for f in left_path.iterdir() if f.suffix.lower() in img_extensions],
        key=extract_number
    )
    right_imgs = sorted(
        [f for f in right_path.iterdir() if f.suffix.lower() in img_extensions],
        key=extract_number
    )

    # 打印找到的文件信息
    print("找到的左图像:")
    for img in left_imgs:
        print(f"  {img.name} -> 数字: {extract_number(img)}")

    print("找到的右图像:")
    for img in right_imgs:
        print(f"  {img.name} -> 数字: {extract_number(img)}")

    # 根据数字匹配
    pairs = []
    left_by_num = {extract_number(f): f for f in left_imgs}
    right_by_num = {extract_number(f): f for f in right_imgs}

    # 找到共同的数字
    common_numbers = set(left_by_num.keys()) & set(right_by_num.keys())
    common_numbers = sorted([n for n in common_numbers if n >= 0])  # 排除无效数字

    for num in common_numbers:
        left_img = left_by_num[num]
        right_img = right_by_num[num]
        pairs.append((str(left_img), str(right_img), num))

    # 报告未匹配的文件
    unmatched_left = set(left_by_num.keys()) - set(common_numbers)
    unmatched_right = set(right_by_num.keys()) - set(common_numbers)

    if unmatched_left:
        print(f"⚠️ 未匹配的左图数字: {sorted(unmatched_left)}")
    if unmatched_right:
        print(f"⚠️ 未匹配的右图数字: {sorted(unmatched_right)}")

    return pairs


def enhanced_preprocess(img):
    """增强的图像预处理"""
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def filter_far_points(disp_float, mask):
    """
    过滤特别远的点（视差值很小的点）
    返回过滤后的视差图和新的掩码
    """
    # 创建新的掩码，排除视差值过大的点
    near_point_mask = disp_float < MAX_DISPARITY_THRESHOLD

    # 同时排除视差值过小的无效点
    valid_point_mask = disp_float > MIN_DISPARITY_THRESHOLD

    # 合并条件：既要是有效点，又不能是过远的点
    combined_mask = mask & near_point_mask & valid_point_mask

    # 创建过滤后的视差图（将远点设为0）
    filtered_disp = disp_float.copy()
    filtered_disp[~combined_mask] = 0

    return filtered_disp, combined_mask


def postprocess_disparity(disp):
    """视差图后处理"""
    # 转换为浮点数
    disp_float = disp.astype(np.float32) / 16.0

    # 过滤无效视差
    mask = disp_float > 0

    # 过滤远点
    disp_filtered, filtered_mask = filter_far_points(disp_float, mask)

    return disp_filtered, filtered_mask


# ===================== 主程序 =====================
def main():
    # 获取左右图像对（按数字匹配）
    img_pairs = get_image_pairs_by_number(LEFT_DIR, RIGHT_DIR)
    if not img_pairs:
        raise RuntimeError(
            f"未在 {LEFT_DIR} 和 {RIGHT_DIR} 中找到匹配的图像对")

    print(f"✅ 成功匹配 {len(img_pairs)} 对图像")
    print(f"📏 远点过滤参数: 最大视差阈值={MAX_DISPARITY_THRESHOLD}, 最小视差阈值={MIN_DISPARITY_THRESHOLD}")

    ensure_dir(OUT_DISP_DIR)

    # 初始化 SGBM 匹配器
    matcher = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        mode=mode
    )

    print("使用增强的图像预处理...")
    print(f"SGBM参数: 视差范围={numDisparities}, 块大小={blockSize}")

    # 遍历图像对计算视差
    for i, (left_path, right_path, num) in enumerate(img_pairs):
        print(f"处理第 {i + 1}/{len(img_pairs)} 对 (数字: {num}): {Path(left_path).name} + {Path(right_path).name}")

        # 读取图像
        L = cv2.imread(left_path)
        R = cv2.imread(right_path)
        if L is None or R is None:
            print(f"⚠️ 跳过第 {i} 对图像（无法读取）")
            continue

        # 检查图像尺寸
        if L.shape != R.shape:
            print(f"⚠️ 图像尺寸不匹配: {L.shape} vs {R.shape}，调整右图尺寸")
            R = cv2.resize(R, (L.shape[1], L.shape[0]))

        # 增强预处理
        grayL = enhanced_preprocess(L)
        grayR = enhanced_preprocess(R)

        # 计算视差
        disp = matcher.compute(grayL, grayR)

        # 后处理（包含远点过滤）
        disp_processed, valid_mask = postprocess_disparity(disp)

        # 可视化视差图
        disp_vis = cv2.normalize(disp_processed, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # 标记无效区域
        disp_color[~valid_mask] = 0

        # 保存结果（使用数字作为文件名）
        cv2.imwrite(f"{OUT_DISP_DIR}/{num:03d}_disp_color.png", disp_color)

        # 保存为int16格式（乘以16以保留小数精度）
        disp_int16 = (disp_processed * 16).astype(np.int16)
        np.save(f"{OUT_DISP_DIR}/{num:03d}_disp.npy", disp_int16)

        # 保存过滤掩码
        np.save(f"{OUT_DISP_DIR}/{num:03d}_mask.npy", valid_mask.astype(np.uint8))

        # 统计信息
        valid_ratio = np.sum(valid_mask) / valid_mask.size * 100
        if np.any(valid_mask):
            valid_disp = disp_processed[valid_mask]
            print(f"✅ 视差图生成：{num:03d}_disp_color.png")
            print(f"   有效像素: {valid_ratio:.1f}%")
            print(f"   视差范围: {valid_disp.min():.1f} - {valid_disp.max():.1f}")
            print(f"   保存格式: int16 (原始值 × 16)")

            # 远点过滤统计
            original_mask = disp.astype(np.float32) / 16.0 > 0
            original_valid_ratio = np.sum(original_mask) / original_mask.size * 100
            filtered_out_ratio = original_valid_ratio - valid_ratio
            print(f"   过滤掉的远点: {filtered_out_ratio:.1f}%")
        else:
            print(f"⚠️  无效视差图: {num:03d}")

        # 显示图像和视差图
        # 调整显示大小以适应屏幕
        max_display_height = 600
        display_scale = min(max_display_height / L.shape[0], 1.0)

        if display_scale < 1:
            new_size = (int(L.shape[1] * display_scale), int(L.shape[0] * display_scale))
            L_display = cv2.resize(L, new_size)
            R_display = cv2.resize(R, new_size)
            disp_color_display = cv2.resize(disp_color, new_size)
        else:
            L_display = L
            R_display = R
            disp_color_display = disp_color

        # 合并显示
        top_row = np.hstack((L_display, R_display))
        bottom_row = np.hstack((disp_color_display, np.zeros_like(disp_color_display)))  # 留空一半
        combined_display = np.vstack((top_row, bottom_row))

        cv2.imshow("左图 | 右图 | 视差图 (按Q退出，其他键继续)", combined_display)
        key = cv2.waitKey(int(DISPLAY_TIME * 1000)) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            print("⏹️ 用户中断处理")
            break

    cv2.destroyAllWindows()
    print("🎉 视差图生成全部完成。结果保存在：", OUT_DISP_DIR)


if __name__ == "__main__":
    main()
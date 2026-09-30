import os
import cv2
import numpy as np
from pathlib import Path

# ===================== 配置 =====================
CALIB_FILE = "stereo_calibration.xml"
LEFT_DIR = "left_bi"
RIGHT_DIR = "right_bi"
OUT_RECT_DIR = "rectified"  # 保存校正后的图

DISPLAY_TIME = 1  # 每对图显示时间（秒）
EPILINE_SPACING = 50  # 极线间距像素


# ===================== 工具函数 =====================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def read_image_list(folder):
    p = Path(folder)
    imgs = sorted([str(x) for x in p.iterdir()
                   if x.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]])
    return imgs


def draw_horizontal_epilines(img, spacing=EPILINE_SPACING, color=(0, 255, 0)):
    out = img.copy()
    h = out.shape[0]
    for y in range(spacing // 2, h, spacing):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), color, 1, cv2.LINE_AA)
    return out


# ===================== 主程序 =====================
def main():
    # ---- 读取标定文件 ----
    fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开标定文件：{CALIB_FILE}")

    left_cam = fs.getNode("left_camera_matrix").mat()
    left_dist = fs.getNode("left_distortion_coefficients").mat()
    right_cam = fs.getNode("right_camera_matrix").mat()
    right_dist = fs.getNode("right_distortion_coefficients").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1_mat = fs.getNode("P1").mat()
    P2_mat = fs.getNode("P2").mat()
    fs.release()

    # ---- 图像路径 ----
    left_imgs = read_image_list(LEFT_DIR)
    right_imgs = read_image_list(RIGHT_DIR)
    n_pairs = min(len(left_imgs), len(right_imgs))
    if n_pairs == 0:
        raise RuntimeError("未在 left_bi / right_bi 中找到匹配图像对")

    ensure_dir(OUT_RECT_DIR)

    # ---- 获取尺寸 ----
    sample = cv2.imread(left_imgs[0])
    H, W = sample.shape[:2]
    image_size = (W, H)

    # ---- 校正映射 ----
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_cam, left_dist, R1, P1_mat, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_cam, right_dist, R2, P2_mat, image_size, cv2.CV_16SC2)

    # ---- 遍历图像对进行校正 ----
    for i in range(n_pairs):
        L = cv2.imread(left_imgs[i])
        R = cv2.imread(right_imgs[i])
        if L is None or R is None:
            print(f"⚠️ 跳过第 {i} 对图像（无法读取）")
            continue

        # 极线校正
        L_rectified = cv2.remap(L, left_map1, left_map2, cv2.INTER_LINEAR)
        R_rectified = cv2.remap(R, right_map1, right_map2, cv2.INTER_LINEAR)

        # 画极线查看
        Lr_lines = draw_horizontal_epilines(L_rectified)
        Rr_lines = draw_horizontal_epilines(R_rectified)
        side_by_side = np.hstack((Lr_lines, Rr_lines))

        # 保存校正结果
        base = f"pair_{i:03d}"

        # 保存合并图像（带极线）
        cv2.imwrite(f"{OUT_RECT_DIR}/{base}_combined.png", side_by_side)

        # 分别保存左右校正图像（不带极线）
        cv2.imwrite(f"{OUT_RECT_DIR}/{base}_left.png", L_rectified)
        cv2.imwrite(f"{OUT_RECT_DIR}/{base}_right.png", R_rectified)

        print(f"✅ 校正完成：{base}_left.png, {base}_right.png, {base}_combined.png")

        # 显示
        cv2.imshow("Rectified (L|R)", side_by_side)
        key = cv2.waitKey(int(DISPLAY_TIME * 1000)) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("🎉 极线校正全部完成。校正图像保存在：", OUT_RECT_DIR)
    print("📁 输出文件：")
    print("   - [pair_XXX]_left.png: 校正后的左图像")
    print("   - [pair_XXX]_right.png: 校正后的右图像")
    print("   - [pair_XXX]_combined.png: 带极线的合并图像")
    print("  注意：要把最后用于制作视差的左右两张校正后的图分别放入left_dis和right_dis中！！！！")


if __name__ == "__main__":
    main()
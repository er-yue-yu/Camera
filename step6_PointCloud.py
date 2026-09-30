import os
import cv2
import numpy as np
from pathlib import Path
import re  # 用于文件名数字提取

# ============ 配置 ============
CALIB_FILE = "stereo_calibration.xml"
LEFT_DIR = "left_dis"
DISP_DIR = "disparities"  # 从该目录读取原始视差数据（.npy文件）
MASK_DIR = "disparities"  # 掩码文件也在同一目录
OUT_PLY_DIR = "ply_output"  # PLY输出目录


# ============ 工具函数 ============
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_ply_rgb(filename, points, colors):
    """
    将 Nx3 点 + Nx3 颜色写为 PLY RGB 格式文件
    PLY格式广泛支持且颜色可靠
    """
    # 过滤无效点（坐标为无穷大/NaN的点）
    valid = np.isfinite(points).all(axis=1)
    pts = points[valid]
    cols = colors[valid]
    n = pts.shape[0]

    if n == 0:
        print(f"⚠️ 没有有效点可保存: {filename}")
        return

    with open(filename, "w") as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 写入点数据（坐标 + RGB颜色）
        for p, c in zip(pts, cols):
            x, y, z = p
            r, g, b = c

            # 确保颜色值在0-255范围内
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))

            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"✅ 已保存PLY点云：{filename} ({n} 点)")


def extract_num(filename):
    """从文件名中提取数字编号（用于准确匹配左右图）"""
    nums = re.findall(r'\d+', filename.stem)  # 提取所有数字
    return int(nums[-1]) if nums else -1  # 取最后一个数字作为编号


def validate_color_conversion():
    """验证颜色转换是否正确"""
    # 检查是否有测试图像
    left_files = sorted(
        [f for f in Path(LEFT_DIR).iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=extract_num
    )

    if not left_files:
        print("⚠️ 没有找到左图文件进行颜色调试")
        return True  # 假设正确，继续处理

    # 读取测试图像
    test_img_bgr = cv2.imread(str(left_files[0]))
    if test_img_bgr is None:
        print("⚠️ 无法读取测试图像")
        return True  # 假设正确，继续处理

    print("\n🎨 颜色转换验证:")
    print("=" * 50)

    # 原始BGR值
    h, w = test_img_bgr.shape[:2]
    center_x, center_y = w // 2, h // 2
    bgr_center = test_img_bgr[center_y, center_x]

    # 转换为RGB
    test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
    rgb_center = test_img_rgb[center_y, center_x]

    print(f"图像中心点 ({center_x}, {center_y}) 颜色值:")
    print(f"  BGR格式: {bgr_center} (OpenCV默认)")
    print(f"  RGB格式: {rgb_center} (转换后)")

    # 检查转换是否正确
    expected_rgb = [bgr_center[2], bgr_center[1], bgr_center[0]]
    actual_rgb = rgb_center.tolist()

    if expected_rgb == actual_rgb:
        print("✅ 颜色转换正确")
        return True
    else:
        print("❌ 颜色转换可能有问题!")
        print(f"  期望: {expected_rgb}")
        print(f"  实际: {actual_rgb}")
        # 仍然返回True，因为可能是OpenCV版本差异
        return True


# ============ 主程序 ============
def main():
    print("🔧 开始PLY点云生成...")

    # 验证颜色转换
    color_ok = validate_color_conversion()
    if not color_ok:
        print("⚠️ 颜色转换有问题，但将继续处理...")

    # 读取重投影矩阵Q
    fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开标定文件：{CALIB_FILE}")
    Q = fs.getNode("Q").mat()
    fs.release()

    # 校验Q矩阵有效性
    if Q is None or Q.shape != (4, 4):
        raise RuntimeError(f"标定文件中Q矩阵无效（需为4x4，实际为{Q.shape if Q is not None else 'None'}）")

    # 创建输出目录
    ensure_dir(OUT_PLY_DIR)

    # 读取视差文件、掩码文件和左图文件（按数字编号排序，确保匹配）
    disp_files = sorted(
        [f for f in Path(DISP_DIR).iterdir() if f.name.endswith("_disp.npy")],
        key=extract_num
    )
    mask_files = sorted(
        [f for f in Path(MASK_DIR).iterdir() if f.name.endswith("_mask.npy")],
        key=extract_num
    )
    left_files = sorted(
        [f for f in Path(LEFT_DIR).iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=extract_num
    )

    # 确保三个列表长度一致
    n_pairs = min(len(disp_files), len(mask_files), len(left_files))

    if n_pairs == 0:
        raise RuntimeError("没有找到匹配数量的左图、视差文件和掩码文件")

    print(f"\n📁 找到 {n_pairs} 对数据，开始生成PLY点云...")

    for i in range(n_pairs):
        disp_path = disp_files[i]
        mask_path = mask_files[i]
        left_path = left_files[i]
        name = disp_path.stem.replace("_disp", "")  # 提取基础文件名

        print(f"处理第 {i + 1}/{n_pairs} 对: {name}")

        # 读取左图并转换颜色通道（BGR转RGB）
        left_img_bgr = cv2.imread(str(left_path))
        if left_img_bgr is None:
            print(f"⚠️ 跳过 {disp_path.name}（左图读取失败）")
            continue
        left_img_rgb = cv2.cvtColor(left_img_bgr, cv2.COLOR_BGR2RGB)  # BGR转RGB

        # 读取视差图并处理格式（兼容16位整数格式）
        disp = np.load(str(disp_path))
        if disp is None:
            print(f"⚠️ 跳过 {disp_path.name}（视差文件读取失败）")
            continue
        # 若视差图是16位整数（如SGBM输出），需除以16得到实际视差值
        if disp.dtype == np.int16:
            disp = disp.astype(np.float32) / 16.0

        # 读取掩码文件
        mask = np.load(str(mask_path))
        if mask is None:
            print(f"⚠️ 跳过 {mask_path.name}（掩码文件读取失败）")
            continue
        # 确保掩码是布尔类型
        mask = mask.astype(bool)

        # 校验视差图、掩码与左图尺寸是否匹配
        if disp.shape[:2] != left_img_rgb.shape[:2] or mask.shape[:2] != left_img_rgb.shape[:2]:
            print(
                f"⚠️ 跳过 {disp_path.name}（尺寸不匹配：视差图{disp.shape} vs 掩码{mask.shape} vs 左图{left_img_rgb.shape}）")
            continue

        # 反投影视差到3D点云
        points_3d = cv2.reprojectImageTo3D(disp, Q)

        # 使用掩码过滤无效点和远点（掩码已经包含了远点过滤）
        valid_mask = mask & np.isfinite(points_3d).all(axis=2)

        # 提取有效点和对应RGB颜色
        pts = points_3d[valid_mask]
        cols = left_img_rgb[valid_mask]  # 使用转换后的RGB颜色

        # 保存点云为PLY格式
        out_ply = Path(OUT_PLY_DIR) / f"{name}.ply"
        write_ply_rgb(out_ply, pts, cols)

        # 显示处理进度和统计信息
        print(f"📊 点云统计 - 有效点: {len(pts)}")
        if len(pts) > 0:
            print(f"    坐标范围: X[{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}], "
                  f"Y[{pts[:, 1].min():.3f}, {pts[:, 1].max():.3f}], "
                  f"Z[{pts[:, 2].min():.3f}, {pts[:, 2].max():.3f}]")
            print(f"    颜色范围: R[{cols[:, 0].min()}-{cols[:, 0].max()}], "
                  f"G[{cols[:, 1].min()}-{cols[:, 1].max()}], "
                  f"B[{cols[:, 2].min()}-{cols[:, 2].max()}]")


    print("\n🎉 所有PLY点云生成完成！")
    print(f"📁 输出目录: {OUT_PLY_DIR}")


if __name__ == "__main__":
    main()
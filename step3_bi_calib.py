import cv2
import numpy as np
import glob
import os


def read_camera_calibration(filename):
    """从XML文件读取单目标定参数"""
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    if not fs.isOpened():
        print(f"❌ 错误：无法打开文件 {filename}")
        return None, None, None

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()

    # reprojection_error 节点可能不存在
    node = fs.getNode("reprojection_error")
    reprojection_error = node.real() if not node.empty() else np.nan

    fs.release()

    return camera_matrix, dist_coeffs, reprojection_error


def list_images(folder):
    """读取指定文件夹中所有图像（支持 jpg/png/jpeg/bmp）"""
    exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def stereo_calibration():
    # ==================== 读取左右相机内参 ====================
    print("📷 读取相机参数...")
    left_camera_matrix, left_dist_coeffs, left_error = read_camera_calibration('camera_calibration_left.xml')
    right_camera_matrix, right_dist_coeffs, right_error = read_camera_calibration('camera_calibration_right.xml')

    if left_camera_matrix is None or right_camera_matrix is None:
        print("❌ 错误：无法读取相机参数文件")
        return

    print(f"左相机重投影误差: {left_error}")
    print(f"右相机重投影误差: {right_error}")

    # ==================== 棋盘格参数 ====================
    chessboard_size = (11, 8)  # (列, 行)：内部角点数（非方格数）
    square_size = 0.02        # 每个方格大小（米）

    # 准备棋盘格在世界坐标中的 3D 点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储角点
    objpoints = []       # 世界坐标系下的3D点
    left_imgpoints = []  # 左图像2D点
    right_imgpoints = [] # 右图像2D点

    # ==================== 获取左右图像 ====================
    left_images = list_images('left_bi')
    right_images = list_images('right_bi')

    if len(left_images) != len(right_images):
        print(f"⚠️ 警告：左右图像数量不匹配 ({len(left_images)} vs {len(right_images)})")

    n_pairs = min(len(left_images), len(right_images))
    print(f"共找到 {n_pairs} 对图像")

    # ==================== 检测角点 ====================
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"▶️ 处理第 {i + 1}/{n_pairs} 对图像: {os.path.basename(left_path)}")

        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f"⚠️ 警告：无法读取 {left_path} 或 {right_path}")
            continue

        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, flags)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, flags)

        if left_ret and right_ret:
            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            left_corners2 = cv2.cornerSubPix(left_gray, left_corners, (11, 11), (-1, -1), criteria)
            right_corners2 = cv2.cornerSubPix(right_gray, right_corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            left_imgpoints.append(left_corners2)
            right_imgpoints.append(right_corners2)

            # 画出检测结果
            cv2.drawChessboardCorners(left_img, chessboard_size, left_corners2, left_ret)
            cv2.drawChessboardCorners(right_img, chessboard_size, right_corners2, right_ret)
            combined = np.hstack((left_img, right_img))
            cv2.imshow('检测到的角点 (左|右)', combined)
            cv2.waitKey(300)
        else:
            print(f"⚠️ 第 {i + 1} 对图像角点检测失败")

    cv2.destroyAllWindows()

    if len(objpoints) < 3:
        print("❌ 错误：有效图像对太少（至少3对）")
        return

    print(f"✅ 成功找到 {len(objpoints)} 对有效图像")

    # ==================== 立体标定 ====================
    print("⚙️ 正在进行立体标定（固定内参）...")
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        left_gray.shape[::-1],
        criteria=criteria_stereo,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"🎯 立体标定 RMS 重投影误差: {rms:.6f} 像素")
    print(f"旋转矩阵 R:\n{R}")
    print(f"平移向量 T:\n{T}")

    # ==================== 立体校正 ====================
    print("🔧 计算立体校正参数...")
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        left_gray.shape[::-1], R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0=最大裁剪，1=保留所有视场
    )

    # ==================== 保存结果 ====================
    print("💾 保存结果到 stereo_calibration.xml...")
    fs = cv2.FileStorage('stereo_calibration.xml', cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print("❌ 错误：无法创建输出文件")
        return

    fs.write('rms_reprojection_error', rms)

    fs.write('left_camera_matrix', left_camera_matrix)
    fs.write('left_distortion_coefficients', left_dist_coeffs)

    fs.write('right_camera_matrix', right_camera_matrix)
    fs.write('right_distortion_coefficients', right_dist_coeffs)

    fs.write('rotation_matrix', R)
    fs.write('translation_vector', T)
    fs.write('essential_matrix', E)
    fs.write('fundamental_matrix', F)

    fs.write('R1', R1)
    fs.write('R2', R2)
    fs.write('P1', P1)
    fs.write('P2', P2)
    fs.write('Q', Q)

    fs.write('validPixROI1', validPixROI1)
    fs.write('validPixROI2', validPixROI2)

    fs.release()
    print("✅ 标定结果已保存到 stereo_calibration.xml")


if __name__ == "__main__":
    stereo_calibration()

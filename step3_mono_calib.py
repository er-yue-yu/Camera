import numpy as np
import cv2
import glob
import os


def calibrate_camera(images_folder, output_file):
    """
    相机标定函数

    参数:
    images_folder: 图片文件夹路径
    output_file: 输出标定文件路径
    """

    # 固定参数
    chessboard_size = (15, 8)  # 棋盘格内角点数量 (9列, 6行)
    square_size = 0.0085  # 棋盘格方格实际尺寸 (米)

    # 终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备对象点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # 存储所有图像的对象点和图像点
    objpoints = []  # 真实世界中的3D点
    imgpoints = []  # 图像中的2D点

    # 获取所有图像文件
    images = glob.glob(os.path.join(images_folder, '*.jpg')) + \
             glob.glob(os.path.join(images_folder, '*.png')) + \
             glob.glob(os.path.join(images_folder, '*.bmp'))

    if len(images) == 0:
        print(f"在文件夹 {images_folder} 中没有找到图像文件!")
        return False

    print(f"找到 {len(images)} 张图像，开始处理...")

    valid_images = 0

    for i, fname in enumerate(images):
        print(f"处理图像 {i + 1}/{len(images)}: {os.path.basename(fname)}")

        img = cv2.imread(fname)
        if img is None:
            print(f"  无法读取图像: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            valid_images += 1
            objpoints.append(objp)

            # 亚像素精确化
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

            # 在图像上绘制角点（可选）
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, chessboard_size, corners_refined, ret)

            # 显示结果（可选）
            cv2.imshow('Chessboard Corners', img_draw)
            cv2.waitKey(300)  # 显示300ms

            print(f"  成功找到角点")
        else:
            print(f"  未找到角点")

    cv2.destroyAllWindows()

    if valid_images < 5:
        print(f"有效图像数量不足 ({valid_images})，至少需要5张图像进行标定!")
        return False

    print(f"\n使用 {valid_images} 张有效图像进行相机标定...")

    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("相机标定失败!")
        return False

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)

    print("\n标定结果:")
    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs.flatten()}")
    print(f"平均重投影误差: {mean_error:.6f} 像素")

    # 保存标定结果到XML文件
    fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("reprojection_error", mean_error)
    fs.write("image_width", gray.shape[1])
    fs.write("image_height", gray.shape[0])
    fs.write("valid_images", valid_images)
    fs.release()

    print(f"\n标定结果已保存到: {output_file}")
    return True


def main():
    """
    主函数 - 分别标定左右相机
    """

    # 标定左相机
    print("=" * 50)
    print("开始标定左相机...")
    print("=" * 50)

    left_success = calibrate_camera(
        images_folder='left_mono',
        output_file='camera_calibration_left.xml'
    )

    if left_success:
        print("✓ 左相机标定完成!")
    else:
        print("✗ 左相机标定失败!")

    print("\n" + "=" * 50)
    print("开始标定右相机...")
    print("=" * 50)

    # 标定右相机
    right_success = calibrate_camera(
        images_folder='right_mono',
        output_file='camera_calibration_right.xml'
    )

    if right_success:
        print("✓ 右相机标定完成!")
    else:
        print("✗ 右相机标定失败!")

    print("\n" + "=" * 50)
    print("标定总结:")
    if left_success and right_success:
        print("✓ 左右相机标定均成功完成!")
        print("生成文件:")
        print("  - camera_calibration_left.xml")
        print("  - camera_calibration_right.xml")
    elif left_success:
        print("✓ 左相机标定成功，右相机标定失败")
    elif right_success:
        print("✓ 右相机标定成功，左相机标定失败")
    else:
        print("✗ 左右相机标定均失败")


if __name__ == "__main__":
    main()
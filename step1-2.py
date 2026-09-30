import cv2
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("=== 摄像采集程序 ===")
    print("请选择模式：")
    print("1. 双目模式（最后要拍一张用于制作视差图的左右图）")
    print("2. 单目模式")
    mode = input("请输入数字选择模式 (1 或 2)：").strip()

    if mode == "1":
        is_stereo = True
        print("已选择：双目模式")
    elif mode == "2":
        is_stereo = False
        print("已选择：单目模式")
        print("请选择使用相机：")
        print("1. 左相机")
        print("2. 右相机")
        cam_choice = input("请输入数字选择相机 (1 或 2)：").strip()
        if cam_choice not in ["1", "2"]:
            print("输入无效，默认选择左相机。")
            cam_choice = "1"
    else:
        print("输入无效，退出。")
        return

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    # 设置分辨率（假设左右各 640×480）
    width = 1280
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 计数器
    i = 0

    print("\n操作说明：")
    print("按空格键拍照并保存，按 q 退出。\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取帧")
            break

        # 分割左右相机
        img_left = frame[:, 0:width//2, :]
        img_right = frame[:, width//2:, :]

        if is_stereo:
            # 双目显示
            cv2.imshow("Left Camera", img_left)
            cv2.imshow("Right Camera", img_right)
        else:
            # 单目显示
            if cam_choice == "1":
                cv2.imshow("Mono Camera (Left)", img_left)
            else:
                cv2.imshow("Mono Camera (Right)", img_right)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("已退出。")
            break
        elif key == ord(' '):
            if is_stereo:
                left_path = os.path.join("left_bi", f"left_{i:03d}.jpg")
                right_path = os.path.join("right_bi", f"right_{i:03d}.jpg")
                ensure_dir("left_bi")
                ensure_dir("right_bi")
                cv2.imwrite(left_path, img_left)
                cv2.imwrite(right_path, img_right)
                print(f"已保存双目图像：{left_path}, {right_path}")
            else:
                if cam_choice == "1":
                    folder = "left_mono"
                    filename = f"left_{i:03d}.jpg"
                    img_to_save = img_left
                else:
                    folder = "right_mono"
                    filename = f"right_{i:03d}.jpg"
                    img_to_save = img_right
                ensure_dir(folder)
                save_path = os.path.join(folder, filename)
                cv2.imwrite(save_path, img_to_save)
                print(f"已保存单目图像：{save_path}")
            i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
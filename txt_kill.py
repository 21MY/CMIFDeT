import os

def check_txt_file(file_path):
    """
    检查TXT文件格式是否正确。
    :param file_path: TXT文件路径
    :return: 如果文件格式正确返回True，否则返回False
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # 检查每行是否有5个值
                if len(parts) != 5:
                    return False
                # 检查值是否为数字
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    return False
        return True
    except Exception as e:
        print(f"错误：检查文件 {file_path} 时发生异常 - {e}")
        return False

def check_and_delete_invalid_txt_files(folder_path):
    """
    检查文件夹下的所有TXT文件，删除不符合格式的文件。
    :param folder_path: 文件夹路径
    """
    # 遍历文件夹内的所有TXT文件
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue  # 跳过非TXT文件

        file_path = os.path.join(folder_path, filename)

        # 检查文件格式
        if not check_txt_file(file_path):
            # 删除不符合格式的文件
            os.remove(file_path)
            print(f"已删除不符合格式的文件: {filename}")

if __name__ == "__main__":
    # 设置文件夹路径
    folder_path = r'D:\TwoStream_Yolov8-main\TwoStream\data\datasets\labels\train'  # 替换为你的文件夹路径

    # 检查并删除不符合格式的文件
    check_and_delete_invalid_txt_files(folder_path)
    print("检查完成！")
import xml.etree.ElementTree as ET
import os

def convert(size, box):
    """
    将边界框坐标转换为YOLO格式。
    :param size: 图像尺寸 (width, height)
    :param box: 边界框坐标 (xmin, xmax, ymin, ymax)
    :return: YOLO格式的边界框 (x_center, y_center, width, height)
    """
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)

def convert_annotation(xml_files_path, save_txt_files_path, classes):
    """
    将VOC格式的XML文件转换为YOLO格式的TXT文件。
    :param xml_files_path: XML文件路径
    :param save_txt_files_path: 保存TXT文件的路径
    :param classes: 类别列表
    """
    # 确保保存路径存在
    os.makedirs(save_txt_files_path, exist_ok=True)

    # 遍历所有XML文件
    xml_files = os.listdir(xml_files_path)
    for xml_name in xml_files:
        if not xml_name.endswith('.xml'):
            continue  # 跳过非XML文件

        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')

        try:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 检查<size>标签
            size = root.find('size')
            if size is None:
                print(f"警告：文件 {xml_name} 中缺少 <size> 标签，已跳过")
                continue

            # 获取图像尺寸
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # 打开TXT文件准备写入
            with open(out_txt_path, 'w') as out_txt_f:
                # 遍历所有目标
                for obj in root.iter('object'):
                    # 检查<name>标签
                    name = obj.find('name')
                    if name is None:
                        print(f"警告：文件 {xml_name} 中某个目标缺少 <name> 标签，已跳过")
                        continue

                    # 获取类别名称
                    cls = name.text

                    # 检查类别是否在列表中
                    if cls not in classes:
                        print(f"警告：文件 {xml_name} 中类别 '{cls}' 不在类别列表中，已跳过")
                        continue

                    # 检查<difficult>标签
                    difficult = obj.find('difficult')
                    if difficult is not None and int(difficult.text) == 1:
                        print(f"警告：文件 {xml_name} 中某个目标标记为 difficult，已跳过")
                        continue

                    # 获取类别索引
                    cls_id = classes.index(cls)

                    # 检查<polygon>标签
                    polygon = obj.find('polygon')
                    if polygon is None:
                        print(f"警告：文件 {xml_name} 中某个目标缺少 <polygon> 标签，已跳过")
                        continue

                    # 提取多边形顶点坐标
                    x_coords = []
                    y_coords = []
                    for i in range(1, 5):  # 假设多边形有4个顶点
                        x = polygon.find(f'x{i}')
                        y = polygon.find(f'y{i}')
                        if x is None or y is None:
                            print(f"警告：文件 {xml_name} 中某个目标的 <polygon> 标签缺少顶点坐标，已跳过")
                            break
                        x_coords.append(float(x.text))
                        y_coords.append(float(y.text))
                    else:
                        # 计算边界框 (xmin, xmax, ymin, ymax)
                        xmin = min(x_coords)
                        xmax = max(x_coords)
                        ymin = min(y_coords)
                        ymax = max(y_coords)

                        # 转换为YOLO格式
                        bb = convert((w, h), (xmin, xmax, ymin, ymax))
                        out_txt_f.write(f"{cls_id} {' '.join(map(str, bb))}\n")

        except ET.ParseError as e:
            print(f"错误：文件 {xml_name} 解析失败 - {e}")
        except Exception as e:
            print(f"未知错误：文件 {xml_name} 处理失败 - {e}")

if __name__ == "__main__":
    # 1、指定YOLO类别
    classes1 = ["car", "truck", "bus", "van", "freight car"]

    # 2、VOC格式的XML标签文件路径
    xml_files1 = r'E:\trainlabel'

    # 3、转化为YOLO格式的TXT标签文件存储路径
    save_txt_files1 = r'E:\label_2w_txt'

    # 执行转换
    convert_annotation(xml_files1, save_txt_files1, classes1)

    # 保存类别列表到classes.txt
    with open(os.path.join(save_txt_files1, 'classes.txt'), 'w') as file:
        for class_name in classes1:
            file.write(class_name + '\n')
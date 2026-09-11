import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 加载模型（请确认以下路径正确）
model_ir = YOLO(r'D:\YOLO\ultralytics-main\runs\detect\LLVIP_ir_4090\weights\best.pt')
model_vis = YOLO(r'D:\YOLO\ultralytics-main\runs\detect\train71\weights\best.pt')
model_fusion = YOLO(r'D:\TwoStream_Yolov8-main\TwoStream\runs\detect\train17\weights\best.pt')


def validate_image(img):
    """确保图像为3通道BGR格式"""
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def process_single(image, model):
    """处理单模态图像（关键修复）"""
    try:
        # 输入预处理
        image = validate_image(image)
        if image is None:
            return None, "无效的输入图像"

        # 显式转换为模型需要的格式
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        # 使用正确的预测接口
        results = model(img_tensor)

        # 可视化结果（保持不变）
        # ... [原有可视化代码]

        return image, result_text
    except Exception as e:
        return None, f"单模态处理错误: {str(e)}"


def process_fusion(visible, infrared):
    """双模态处理（关键修复）"""
    try:
        # 输入验证
        visible = validate_image(visible)
        infrared = validate_image(infrared)
        if visible is None or infrared is None:
            return None, "无效的输入图像"

        # 统一尺寸
        infrared = cv2.resize(infrared, (visible.shape[1], visible.shape[0]))

        # 构建6通道输入
        combined = np.concatenate([visible, infrared], axis=2)

        # 转换为模型需要的张量格式 [1,6,H,W]
        input_tensor = torch.from_numpy(
            combined.transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0

        # 执行推理
        results = model_fusion(input_tensor)

        # 可视化结果（保持不变）
        # ... [原有可视化代码]

        return output_img, result_text
    except Exception as e:
        return None, f"双模态处理错误: {str(e)}"


# Gradio界面保持不变
# ... [原有界面代码]

if __name__ == "__main__":
    gr.close_all()
    interface.launch()
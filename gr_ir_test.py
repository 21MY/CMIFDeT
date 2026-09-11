import gradio as gr
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 设置文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型
model = YOLO(r'D:\TwoStream_Yolov8-main\TwoStream\runs\detect\LLVIP_ir_4090\weights\best.pt')


def process_image(image):
    try:
        # 转换图像格式
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存上传的图像
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg'), image)

        # 预测
        results = model(image, stream=False)

        # 绘制结果
        annotated_image = image.copy()
        detection_results = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]

                # 绘制
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_image, f'{model.names[int(cls)]}:{conf:.2f}',
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                detection_results.append(
                    f'Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, '
                    f'Box: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})'
                )

        # 保存结果
        cv2.imwrite(os.path.join(RESULT_FOLDER, 'result_image.jpg'), annotated_image)

        return annotated_image, '\n'.join(detection_results)

    except Exception as e:
        print(f"Error: {str(e)}")
        return image, f"Error: {str(e)}"


# 创建界面
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="上传图像"),
    outputs=[gr.Image(type="numpy", label="处理后的图像"), gr.Textbox(label="检测结果")],
    title="YOLOv8 图像检测",
    description="上传图像并使用YOLOv8模型进行检测"
)

# 启动应用
iface.launch(share=True)
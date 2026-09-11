import gradio as gr
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# 初始化模型
model = YOLO(r'D:\new-project\TwoStream_Yolov8-main\TwoStream\runs\new\rgb+r9\weights\best.pt')


def process_batch(visible_files, infrared_files, save_dir):
    """批量处理函数"""
    os.makedirs(save_dir, exist_ok=True)
    log = []

    # 校验文件数量匹配
    if len(visible_files) != len(infrared_files):
        return "错误：可见光与红外图像数量不匹配", ""

    for idx, (v_path, ir_path) in enumerate(zip(visible_files, infrared_files)):
        try:
            # 读取图像
            visible_img = cv2.imread(v_path.name)
            infrared_img = cv2.imread(ir_path.name, cv2.IMREAD_UNCHANGED)

            # 预处理
            if len(infrared_img.shape) == 2:
                infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
            visible_img = visible_img[:, :, :3]  # 去除alpha通道

            # 拼接通道
            combined = np.concatenate([visible_img, infrared_img], axis=2)

            # 推理
            results = model(combined)

            # 创建副本
            vis_result = visible_img.copy()
            ir_result = infrared_img.copy()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)  # 确保转换成float再四舍五入
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    # 格式化置信度为小数点后两位
                    conf_str = f"{conf:.2f}"

                    # 绘制可见光
                    cv2.rectangle(vis_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_result, f"{class_name} {conf_str}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

                    # 绘制红外
                    cv2.rectangle(ir_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ir_result, f"{class_name} {conf_str}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = f"pair_{idx + 1}_{timestamp}"

            vis_save_path = os.path.join(save_dir, f"{base_name}_visible.jpg")
            ir_save_path = os.path.join(save_dir, f"{base_name}_infrared.jpg")

            cv2.imwrite(vis_save_path, vis_result)
            cv2.imwrite(ir_save_path, ir_result)

            log.append(f"成功处理：{os.path.basename(v_path.name)} 和 {os.path.basename(ir_path.name)}")

        except Exception as e:
            log.append(f"处理失败：{v_path.name} - {str(e)}")

    return "\n".join(log), save_dir


# 创建界面
with gr.Blocks() as demo:
    gr.Markdown("## 双模态批量检测系统")

    with gr.Row():
        with gr.Column():
            visible_input = gr.Files(label="上传可见光图像", file_types=["image"])
            infrared_input = gr.Files(label="上传红外图像", file_types=["image"])
            save_dir = gr.Textbox(label="保存路径", value="./results")
            run_btn = gr.Button("开始处理")

        with gr.Column():
            output_log = gr.Textbox(label="处理日志", interactive=False)
            output_dir = gr.Textbox(label="结果保存位置", interactive=False)

    run_btn.click(
        fn=process_batch,
        inputs=[visible_input, infrared_input, save_dir],
        outputs=[output_log, output_dir]
    )

if __name__ == "__main__":
    demo.launch()
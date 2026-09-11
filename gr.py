# import gradio as gr
# import cv2
# import numpy as np
# import os
# from datetime import datetime
# from ultralytics import YOLO
# import base64
#
# # 初始化模型
# model = YOLO(r'D:\new-project\TwoStream_Yolov8-main\TwoStream\runs\new\rgb+r9\weights\best.pt')
#
# # 背景图片路径 - 请修改为你电脑上的实际图片路径
# BACKGROUND_IMAGE_PATH = r"D:\new-project\UAVgp\total\DJI_20250403152330_0003_D.JPG"  # 请修改这个路径
#
#
# def create_css_with_background(image_path):
#     """创建包含背景图像的CSS样式"""
#     if os.path.exists(image_path):
#         # 将图片转换为base64编码以便在CSS中使用
#         with open(image_path, "rb") as image_file:
#             base64_image = base64.b64encode(image_file.read()).decode()
#
#         css = f"""
#         .gradio-container {{
#             background-image: url("data:image/jpeg;base64,{base64_image}") !important;
#             background-size: cover !important;
#             background-repeat: no-repeat !important;
#             background-attachment: fixed !important;
#             background-position: center !important;
#         }}
#         .contain {{
#             display: flex !important;
#             min-height: 100vh !important;
#         }}
#         .panel {{
#             background: rgba(255, 255, 255, 0.92) !important;
#             backdrop-filter: blur(10px) !important;
#             border-radius: 15px !important;
#             padding: 20px !important;
#             margin: 10px !important;
#             border: 1px solid rgba(255, 255, 255, 0.3) !important;
#             box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
#         }}
#         """
#     else:
#         # 如果图片不存在，使用默认渐变背景
#         css = """
#         .gradio-container {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#         }
#         .panel {
#             background: rgba(255, 255, 255, 0.95) !important;
#             backdrop-filter: blur(10px) !important;
#             border-radius: 15px !important;
#             padding: 20px !important;
#             margin: 10px !important;
#             border: 1px solid rgba(255, 255, 255, 0.3) !important;
#             box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
#         }
#         """
#         print(f"警告: 背景图片路径 '{BACKGROUND_IMAGE_PATH}' 不存在，使用默认背景")
#
#     return css
#
#
# # 创建CSS
# custom_css = create_css_with_background(BACKGROUND_IMAGE_PATH)
#
#
# def process_batch(visible_files, infrared_files, save_dir):
#     """批量处理函数"""
#     os.makedirs(save_dir, exist_ok=True)
#     log = []
#     output_images = []
#
#     # 校验文件数量匹配
#     if len(visible_files) != len(infrared_files):
#         return "错误：可见光与红外图像数量不匹配", "", []
#
#     for idx, (v_path, ir_path) in enumerate(zip(visible_files, infrared_files)):
#         try:
#             # 读取图像
#             visible_img = cv2.imread(v_path.name)
#             infrared_img = cv2.imread(ir_path.name, cv2.IMREAD_UNCHANGED)
#
#             # 预处理
#             if len(infrared_img.shape) == 2:
#                 infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
#             visible_img = visible_img[:, :, :3]  # 去除alpha通道
#
#             # 拼接通道
#             combined = np.concatenate([visible_img, infrared_img], axis=2)
#
#             # 推理
#             results = model(combined)
#
#             # 创建副本
#             vis_result = visible_img.copy()
#             ir_result = infrared_img.copy()
#
#             for result in results:
#                 boxes = result.boxes.cpu().numpy()
#                 for box in boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     conf = round(float(box.conf[0]), 2)  # 确保转换成float再四舍五入
#                     cls_id = int(box.cls[0])
#                     class_name = model.names[cls_id]
#
#                     # 格式化置信度为小数点后两位
#                     conf_str = f"{conf:.2f}"
#
#                     # 绘制可见光
#                     cv2.rectangle(vis_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(vis_result, f"{class_name} {conf_str}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)
#
#                     # 绘制红外
#                     cv2.rectangle(ir_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(ir_result, f"{class_name} {conf_str}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)
#
#             # 保存结果
#             timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#             base_name = f"pair_{idx + 1}_{timestamp}"
#
#             vis_save_path = os.path.join(save_dir, f"{base_name}_visible.jpg")
#             ir_save_path = os.path.join(save_dir, f"{base_name}_infrared.jpg")
#
#             cv2.imwrite(vis_save_path, vis_result)
#             cv2.imwrite(ir_save_path, ir_result)
#
#             # 将BGR转换为RGB用于显示
#             vis_result_rgb = cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
#             ir_result_rgb = cv2.cvtColor(ir_result, cv2.COLOR_BGR2RGB)
#
#             # 添加处理前后的图片到输出列表
#             output_images.append((vis_result_rgb, f"可见光结果 {idx + 1}"))
#             output_images.append((ir_result_rgb, f"红外结果 {idx + 1}"))
#
#             log.append(f"成功处理：{os.path.basename(v_path.name)} 和 {os.path.basename(ir_path.name)}")
#
#         except Exception as e:
#             log.append(f"处理失败：{v_path.name} - {str(e)}")
#
#     return "\n".join(log), save_dir, output_images
#
#
# # 更新输入预览的函数
# def update_previews(visible_files, infrared_files):
#     visible_images = []
#     infrared_images = []
#
#     if visible_files:
#         for file in visible_files:
#             img = cv2.imread(file.name)
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             visible_images.append((img_rgb, os.path.basename(file.name)))
#
#     if infrared_files:
#         for file in infrared_files:
#             img = cv2.imread(file.name)
#             if len(img.shape) == 2:
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#             else:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             infrared_images.append((img, os.path.basename(file.name)))
#
#     return visible_images, infrared_images
#
#
# # 创建界面
# with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
#     gr.Markdown("""
#     # 🎯 双模态批量检测系统
#     **上传可见光和红外图像进行目标检测**
#     """)
#
#     with gr.Row():
#         with gr.Column(scale=1):
#             with gr.Column(elem_classes="panel"):
#                 gr.Markdown("### 📤 输入区域")
#                 visible_input = gr.Files(label="上传可见光图像", file_types=["image"],
#                                          file_count="multiple")
#                 infrared_input = gr.Files(label="上传红外图像", file_types=["image"],
#                                           file_count="multiple")
#
#             with gr.Column(elem_classes="panel"):
#                 gr.Markdown("### ⚙️ 设置")
#                 save_dir = gr.Textbox(label="保存路径", value="./results")
#                 run_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
#
#             with gr.Column(elem_classes="panel"):
#                 gr.Markdown("### 📋 日志")
#                 output_log = gr.Textbox(label="处理日志", interactive=False, lines=6)
#                 output_dir = gr.Textbox(label="结果保存位置", interactive=False)
#
#         with gr.Column(scale=2):
#             with gr.Column(elem_classes="panel"):
#                 gr.Markdown("### 📊 结果显示")
#                 with gr.Tabs():
#                     with gr.TabItem("🖼️ 检测结果"):
#                         output_gallery = gr.Gallery(
#                             label="检测结果图像",
#                             show_label=True,
#                             elem_id="gallery",
#                             columns=2,
#                             height="auto"
#                         )
#
#                     with gr.TabItem("📋 输入预览"):
#                         with gr.Row():
#                             visible_preview = gr.Gallery(
#                                 label="可见光输入预览",
#                                 show_label=True,
#                                 columns=2,
#                                 height="auto"
#                             )
#                         with gr.Row():
#                             infrared_preview = gr.Gallery(
#                                 label="红外输入预览",
#                                 show_label=True,
#                                 columns=2,
#                                 height="auto"
#                             )
#
#     # 当文件上传时更新预览
#     visible_input.change(
#         fn=update_previews,
#         inputs=[visible_input, infrared_input],
#         outputs=[visible_preview, infrared_preview]
#     )
#
#     infrared_input.change(
#         fn=update_previews,
#         inputs=[visible_input, infrared_input],
#         outputs=[visible_preview, infrared_preview]
#     )
#
#     run_btn.click(
#         fn=process_batch,
#         inputs=[visible_input, infrared_input, save_dir],
#         outputs=[output_log, output_dir, output_gallery]
#     )
#
# if __name__ == "__main__":
#     # 打印背景图片信息
#     if os.path.exists(BACKGROUND_IMAGE_PATH):
#         print(f"使用背景图片: {BACKGROUND_IMAGE_PATH}")
#     else:
#         print(f"背景图片未找到: {BACKGROUND_IMAGE_PATH}")
#         print("请修改代码中的 BACKGROUND_IMAGE_PATH 变量为你的图片路径")
#
#     demo.launch(
#         server_name="127.0.0.1",
#         server_port=7861,
#         share=True
#     )

import gradio as gr
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import base64

# Initialize model
model = YOLO(r'D:\new-project\TwoStream_Yolov8-main\TwoStream\runs\new\rgb+r9\weights\best.pt')

# Background image path - Please modify to your actual image path
BACKGROUND_IMAGE_PATH = r"C:\Users\28438\Downloads\绘制无人机茶园图片 (1).png"


def create_css_with_background(image_path):
    """Create CSS styles with background image"""
    if os.path.exists(image_path):
        # Convert image to base64 for use in CSS
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()

        css = f"""
        .gradio-container {{
            background-image: url("data:image/jpeg;base64,{base64_image}") !important;
            background-size: cover !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            background-position: center !important;
        }}
        .contain {{
            display: flex !important;
            min-height: 100vh !important;
        }}
        .panel {{
            background: rgba(255, 255, 255, 0.92) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        }}
        /* Text styling for better visibility on green background */
        .gradio-container h1, .gradio-container h2, .gradio-container h3 {{
            color: #ffffff !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7) !important;
            font-weight: bold !important;
        }}
        .gradio-container .markdown {{
            color: #ffffff !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7) !important;
        }}
        .gradio-container label {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7) !important;
            font-weight: bold !important;
        }}
        .gradio-container .textbox {{
            color: #333333 !important;
        }}
        """
    else:
        # Use default gradient background if image doesn't exist
        css = """
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        .panel {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        }
        """
        print(f"Warning: Background image path '{BACKGROUND_IMAGE_PATH}' does not exist, using default background")

    return css


# Create CSS
custom_css = create_css_with_background(BACKGROUND_IMAGE_PATH)


def process_batch(visible_files, infrared_files, save_dir):
    """Batch processing function"""
    os.makedirs(save_dir, exist_ok=True)
    log = []
    output_images = []

    # Validate file count match
    if len(visible_files) != len(infrared_files):
        return "Error: Visible and infrared image counts do not match", "", []

    for idx, (v_path, ir_path) in enumerate(zip(visible_files, infrared_files)):
        try:
            # Read images
            visible_img = cv2.imread(v_path.name)
            infrared_img = cv2.imread(ir_path.name, cv2.IMREAD_UNCHANGED)

            # Preprocessing
            if len(infrared_img.shape) == 2:
                infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
            visible_img = visible_img[:, :, :3]  # Remove alpha channel

            # Concatenate channels
            combined = np.concatenate([visible_img, infrared_img], axis=2)

            # Inference
            results = model(combined)

            # Create copies for results
            vis_result = visible_img.copy()
            ir_result = infrared_img.copy()

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)  # Ensure conversion to float before rounding
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    # Format confidence to 2 decimal places
                    conf_str = f"{conf:.2f}"

                    # Draw on visible image
                    cv2.rectangle(vis_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_result, f"{class_name} {conf_str}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

                    # Draw on infrared image
                    cv2.rectangle(ir_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ir_result, f"{class_name} {conf_str}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = f"pair_{idx + 1}_{timestamp}"

            vis_save_path = os.path.join(save_dir, f"{base_name}_visible.jpg")
            ir_save_path = os.path.join(save_dir, f"{base_name}_infrared.jpg")

            cv2.imwrite(vis_save_path, vis_result)
            cv2.imwrite(ir_save_path, ir_result)

            # Convert BGR to RGB for display
            vis_result_rgb = cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB)
            ir_result_rgb = cv2.cvtColor(ir_result, cv2.COLOR_BGR2RGB)

            # Add processed images to output list
            output_images.append((vis_result_rgb, f"Visible Result {idx + 1}"))
            output_images.append((ir_result_rgb, f"Infrared Result {idx + 1}"))

            log.append(f"Successfully processed: {os.path.basename(v_path.name)} and {os.path.basename(ir_path.name)}")

        except Exception as e:
            log.append(f"Processing failed: {v_path.name} - {str(e)}")

    return "\n".join(log), save_dir, output_images


# Function to update input previews
def update_previews(visible_files, infrared_files):
    visible_images = []
    infrared_images = []

    if visible_files:
        for file in visible_files:
            img = cv2.imread(file.name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visible_images.append((img_rgb, os.path.basename(file.name)))

    if infrared_files:
        for file in infrared_files:
            img = cv2.imread(file.name)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            infrared_images.append((img, os.path.basename(file.name)))

    return visible_images, infrared_images


# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Dual-Modal Batch Detection System
    **Upload visible and infrared images for object detection**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_classes="panel"):
                gr.Markdown("### 📤 Input Area")
                visible_input = gr.Files(label="Upload Visible Images", file_types=["image"],
                                         file_count="multiple")
                infrared_input = gr.Files(label="Upload Infrared Images", file_types=["image"],
                                          file_count="multiple")

            with gr.Column(elem_classes="panel"):
                gr.Markdown("### ⚙️ Settings")
                save_dir = gr.Textbox(label="Save Path", value="./results")
                run_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg")

            with gr.Column(elem_classes="panel"):
                gr.Markdown("### 📋 Log")
                output_log = gr.Textbox(label="Processing Log", interactive=False, lines=6)
                output_dir = gr.Textbox(label="Results Save Location", interactive=False)

        with gr.Column(scale=2):
            with gr.Column(elem_classes="panel"):
                gr.Markdown("### 📊 Results Display")
                with gr.Tabs():
                    with gr.TabItem("🖼️ Detection Results"):
                        output_gallery = gr.Gallery(
                            label="Detection Result Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            height="auto"
                        )

                    with gr.TabItem("📋 Input Preview"):
                        with gr.Row():
                            visible_preview = gr.Gallery(
                                label="Visible Input Preview",
                                show_label=True,
                                columns=2,
                                height="auto"
                            )
                        with gr.Row():
                            infrared_preview = gr.Gallery(
                                label="Infrared Input Preview",
                                show_label=True,
                                columns=2,
                                height="auto"
                            )

    # Update preview when files are uploaded
    visible_input.change(
        fn=update_previews,
        inputs=[visible_input, infrared_input],
        outputs=[visible_preview, infrared_preview]
    )

    infrared_input.change(
        fn=update_previews,
        inputs=[visible_input, infrared_input],
        outputs=[visible_preview, infrared_preview]
    )

    run_btn.click(
        fn=process_batch,
        inputs=[visible_input, infrared_input, save_dir],
        outputs=[output_log, output_dir, output_gallery]
    )

if __name__ == "__main__":
    # Print background image information
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        print(f"Using background image: {BACKGROUND_IMAGE_PATH}")
    else:
        print(f"Background image not found: {BACKGROUND_IMAGE_PATH}")
        print("Please modify the BACKGROUND_IMAGE_PATH variable in the code to your image path")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=True
    )
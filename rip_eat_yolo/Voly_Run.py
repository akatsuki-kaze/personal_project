from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
import sys
import os
import tkinter as tk
from tkinter import filedialog

def choose_input_source():
    print("请选择输入来源：")
    print("[1] 摄像头")
    print("[2] 视频文件")
    choice = input("请输入数字 (1 或 2): ").strip()

    if choice == "1":
        return 0, "摄像头"
    elif choice == "2":
        #选择视频文件
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mkv;*.mov"), ("所有文件", "*.*")]
        )
        if not video_path:
            print("未选择视频文件，程序退出")
            sys.exit(0)
        return video_path, video_path
    else:
        print("无效的输入，程序退出")
        sys.exit(1)

#------------------------------------非极大值抑制------------------------------------------------------------------------------------------------------
def non_max_suppression(boxes, scores, threshold):
    """
    非极大值抑制算法实现
    参数：
    boxes: 边界框坐标，格式为 [x_min, y_min, x_max, y_max]
    scores: 边界框对应的置信度得分
    threshold: NMS阈值，用于过滤重叠边界框的阈值
    返回值：
    selected_indices: 经过NMS筛选后选中的边界框索引
    """
 
    selected_indices = []
 
    # 根据得分降序排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
 
    while len(sorted_indices) > 0:
        max_index = sorted_indices[0]
        selected_indices.append(max_index)
 
        # 计算当前最大得分框与其他框的IoU
        max_box = boxes[max_index]
        other_boxes = [boxes[i] for i in sorted_indices[1:]]
        iou_scores = [calculate_iou(max_box, box) for box in other_boxes]
 
        # 根据IoU和阈值过滤边界框
        filtered_indices = [i for i, iou in zip(sorted_indices[1:], iou_scores) if iou < threshold]
        sorted_indices = filtered_indices
 
    return selected_indices
 
def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    参数：
    box1: 第一个边界框的坐标，格式为 [x_min, y_min, x_max, y_max]
    box2: 第二个边界框的坐标，格式为 [x_min, y_min, x_max, y_max]
    返回值：
    iou: 两个边界框的IoU
    """
 
    # 计算两个边界框的交集区域坐标
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
 
    # 计算交集区域的面积
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
 
    # 计算两个边界框的并集区域面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
 
    # 计算IoU
    iou = intersection_area / union_area if union_area > 0 else 0
 
    return iou
#----------------------------------------------------------------------------------------------------------------------------------------
#目标坐标检测
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
#------------------------------------------------------------------------------

def detect_media():
    # ======================= 配置区 =======================
    # 模型配置
    model_config = {
        'model_path': r'F:\16521\Voly_vision_yolov11\ultralytics-main\ultralytics\runs\train\exp42\weights\best.pt',  # 本地模型路径，注意配置！！！！！！！！！！！！！！！！！！！！！！！
        'download_url': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'  # 如果没有模型文件下载URL
    }
    
    # 推理参数
    predict_config = {
        'conf_thres': 0.4,     # 置信度阈值
        'iou_thres': 0.2,      # IoU阈值
        'imgsz': 640,           # 输入分辨率
        'line_width': 2,        # 检测框线宽
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    }
    # ====================== 配置结束 ======================

    try:
        # 选择输入来源
        input_source, source_desc = choose_input_source()

        # 初始化视频源
        cap = cv2.VideoCapture(input_source)
        if isinstance(input_source, int):
            # 如果使用摄像头，设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise IOError(f"无法打开视频源 ({source_desc})，请检查设备连接或文件路径。")

        # 询问是否保存推理出的视频文件
        save_video = False
        video_writer = None
        output_path = None
        answer = input("是否保存推理出的视频文件？(y/n): ").strip().lower()
        if answer == "y":
            save_video = True
            # 创建保存目录：代码文件所在目录下的 predict 文件夹
            save_dir = os.path.join(os.getcwd(), "predict")
            os.makedirs(save_dir, exist_ok=True)
            # 获取视频属性（宽度、高度、fps）
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 25  # 如果无法获取fps，设定默认值
            # 构造输出视频文件路径
            output_path = os.path.join(save_dir, "output_inference.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"推理视频将保存至: {output_path}")

        # 加载模型（带异常捕获）
        if not Path(model_config['model_path']).exists():
            if model_config['download_url']:
                print("开始下载模型...")
                YOLO(model_config['download_url']).download(model_config['model_path'])
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_config['model_path']}")

        # 初始化模型
        model = YOLO(model_config['model_path']).to(predict_config['device'])
        print(f"✅ 模型加载成功 | 设备: {predict_config['device'].upper()}")
        print(f"输入来源: {source_desc}")

        # 实时检测循环
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束或中断")
                break

            # 执行推理
            results = model.predict(
                source=frame,
                stream=True,  # 流式推理
                verbose=False,
                conf=predict_config['conf_thres'],
                iou=predict_config['iou_thres'],
                imgsz=predict_config['imgsz'],
                device=predict_config['device']
            )

            # 遍历生成器获取结果（取第一个结果）
            for result in results:
                annotated_frame = result.plot(line_width=predict_config['line_width'])
                #输出框坐标
                print(result.boxes.xywhn)
                break

            # 摄像头模式下显示FPS
            if isinstance(input_source, int):
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示实时画面
            cv2.imshow('YOLO Real-time Detection', annotated_frame)

            # 如保存视频，写入视频文件
            if save_video and video_writer is not None:
                video_writer.write(annotated_frame)

            # 按键退出q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        # 释放资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("✅ 检测结束")
        if save_video and output_path is not None:
            print(f"推理结果视频已保存至: {output_path}")

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        print("问题排查建议：")
        print("1. 检查视频源是否正确连接或文件路径是否正确")
        print("2. 确认模型文件路径正确")
        print("3. 检查CUDA是否可用（如需GPU加速）")
        print("4. 尝试降低分辨率设置")

if __name__ == "__main__":
    detect_media()


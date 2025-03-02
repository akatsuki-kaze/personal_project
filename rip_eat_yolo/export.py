from ultralytics import YOLO
 
# 加载模型
#model = YOLO('best.pt')  # 加载官方模型（示例）
model = YOLO('Ball_Tracker.pt')  # 加载自定义训练模型（示例）
 
# 导出模型
model.export(format='onnx')
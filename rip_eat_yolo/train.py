# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model=r'F:\16521\Voly_vision_yolov11\ultralytics-main\ultralytics\cfg\models\11\yolo11s.yaml')
    model.train(data=r'train.yaml',
                imgsz=640,
                epochs=300,#训练轮数
                batch=24,#一次训练张数
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )

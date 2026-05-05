from ultralytics import YOLO

model=YOLO('yolov8n.pt')

model.train(data='roboflow/data.yaml',epochs=50,imgsz=640,device='cpu')
print("Eğitim tamamlandı! Model 'runs/detect/train/weights/best.pt' konumuna kaydedildi.")
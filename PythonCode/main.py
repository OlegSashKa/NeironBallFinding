from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO(r'D:\VS_cod\Python\Neironka\runs\detect\train18\weights\best.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="config.yaml", epochs=20)
# yolo task=detect model=bestBolls.pt data=config.yaml epochs=20
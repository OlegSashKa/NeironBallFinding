import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm
import logging

# Отключение вывода информации от библиотеки YOLO
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Загрузка модели YOLOv8
model = YOLO(r'D:\VS_cod\Python\Neironka\runs\detect\train18\weights\best.pt')

# Укажите путь к директории с изображениями
input_dir = r'D:\VS_cod\Python\Neironka\TrainBall\image3\test\redball'  # Замените на путь к вашей папке с изображениями

classes = model.names

# Создание файла classes.txt
classes_file_path = os.path.join(input_dir, 'classes.txt')
with open(classes_file_path, 'w') as f:
    for class_id, class_name in classes.items():
        f.write(f"{class_name}\n")

# Функция для сохранения координат в формате (0 x1 y1 x2 y2)
def save_labels(class_id, x_center, y_center, width, height, filename):
    with open(filename, 'w') as f:
        f.write(f"{class_id} {round(x_center,6)} {round(y_center,6)} {round(width,6)} {round(height,6)}\n")

# Перебор всех файлов в указанной директории
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
for filename in tqdm(image_files, desc='Обработка изображений'):
    # Полный путь к изображению
    image_path = os.path.join(input_dir, filename)
    
    # Загрузка изображения
    frame = cv2.imread(image_path)
    if frame is None:
        continue

    # Выполнение предсказания
    result = model(frame, conf=0.6)

    if len(result[0].boxes) == 0:
        # Создание пустого файла, если объект не обнаружен
        label_filename = os.path.splitext(image_path)[0] + '.txt'
        open(label_filename, 'w').close()
    else:# Извлечение координат объектов
        for box in result[0].boxes:
            # Извлечение координат
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])  # Преобразование ID класса в целое число

            # Преобразование координат в процентное соотношение
                # Преобразование координат в центр и размеры
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            h, w, _ = frame.shape
            x_center = x_center/w
            y_center = y_center/h
            width = width/w
            height = height/h
            
            x_center = float(x_center.item())
            y_center = float(y_center.item())
            width = float(width.item())
            height = float(height.item())

            # Формирование имени файла для координат
            label_filename = os.path.splitext(image_path)[0] + '.txt'

            # Сохранение координат в файл
            save_labels(class_id, x_center, y_center, width, height, label_filename)

print("Обработка изображений завершена.")
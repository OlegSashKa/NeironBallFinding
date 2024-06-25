import cv2
import os
from tqdm import tqdm

# Укажите путь к директории с изображениями
input_dir = r'RedBall\train\image2'  # Замените на путь к вашей папке с изображениями

# Функция для поворота изображения и сохранения результата
def rotate_and_save(image_path, angle):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f'Не удалось загрузить изображение: {image_path}')
        return

    # Получение размеров изображения
    height, width = image.shape[:2]

    # Вычисление центра изображения для поворота
    center = (width // 2, height // 2)

    # Поворот изображения на указанный угол
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))

    # Формирование имени файла для сохранения
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(input_dir, f'{base_name}_rotated_{angle}.jpg')

    # Сохранение повернутого изображения
    cv2.imwrite(output_path, rotated_image)

# Получение списка файлов для обработки
image_files = [filename for filename in os.listdir(input_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Инициализация tqdm для визуализации прогресса
progress_bar = tqdm(total=len(image_files), desc='Processing images', unit='image')

# Перебор всех файлов в указанной директории
for filename in image_files:
    # Полный путь к изображению
    image_path = os.path.join(input_dir, filename)
    
    # Поворот на 30 градусов
    rotate_and_save(image_path, 30)
    progress_bar.update(1)

    # Поворот на 180 градусов
    rotate_and_save(image_path, 180)
    progress_bar.update(1)

    # Поворот на 270 градусов
    rotate_and_save(image_path, 270)
    progress_bar.update(1)

# Завершение работы tqdm
progress_bar.close()

print("Обработка изображений завершена.")

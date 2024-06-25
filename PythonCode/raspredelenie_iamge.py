import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.8):
    # Создаем папки train и test в выходной директории
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Проходим по всем классам в исходной директории
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Создаем папки для текущего класса в train и test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Получаем список всех изображений в текущем классе
        images = os.listdir(class_dir)
        random.shuffle(images)
        
        # Разбиваем изображения на train и test
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Копируем изображения в соответствующие папки
        for image in train_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
        
        for image in test_images:
            shutil.copy(os.path.join(class_dir, image), os.path.join(test_class_dir, image))
    
    print(f'Dataset split completed. Train: {train_dir}, Test: {test_dir}')

# Пример использования:
input_folder = r'RedBall\train\image2'
output_folder = r'RedBall\train\image3'

split_dataset(input_folder, output_folder)

import os
import cv2

# Параметры
data_dir = 'dataset/train'  # Путь к папке с изображениями
target_size = (28, 28)  # Новый размер изображений

# Проходим по всем классам
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        # Проходим по всем изображениям в классе
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Проверяем, что это изображение
                # Загружаем изображение
                image = cv2.imread(img_path)

                # Проверяем, является ли изображение буквой (например, по имени файла)
                if class_name.isdigit():  # Если имя класса состоит из букв
                    # Изменяем размер изображения
                    resized_image = cv2.resize(image, target_size)

                    # Сохраняем измененное изображение
                    cv2.imwrite(img_path, resized_image)  # Перезаписываем оригинальное изображение

print("Изменение размера завершено.")







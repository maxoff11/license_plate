import cv2
import os

def process_images(input_folder, output_folder):
    # Создаем выходную папку, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Проходим по всем изображениям в папке
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Проверяем расширение файла
            # Загружаем изображение
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Преобразуем в градации серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Используем пороговую обработку для получения черного фона и белых символов
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Применяем морфологическую операцию для улучшения контуров
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Находим контуры
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Создаем черное изображение для вывода
            output_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # Заполняем контуры белым цветом
            cv2.drawContours(output_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            # Сохраняем обработанное изображение
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, output_image)

# Пример использования
input_folder = 'dataset/train/2'  # Укажите путь к папке с изображениями
output_folder = 'dataset/train/2x'  # Укажите путь к папке для сохранения результатов
process_images(input_folder, output_folder)

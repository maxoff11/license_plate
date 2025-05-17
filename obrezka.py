import cv2
import os

# Путь к папке с изображениями
input_folder = 'dataset/train/Y'
output_folder = 'dataset/train/Yx'

# Создаем выходную папку, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Проходим по всем изображениям в папке
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Загружаем изображение
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

        # Применяем пороговое значение для бинаризации
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если контуры найдены
        if contours:
            # Находим ограничивающий прямоугольник для первого контура
            x, y, w, h = cv2.boundingRect(contours[0])

            # Обрезаем изображение по ограничивающему прямоугольнику
            cropped_image = img[y:y+h, x:x+w]

            # Изменяем размер изображения обратно до 28x28
            contour_image_resized = cv2.resize(cropped_image, (28, 28))

            # Сохраняем полученное изображение
            cv2.imwrite(os.path.join(output_folder, filename), contour_image_resized)

print("Обработка завершена!")

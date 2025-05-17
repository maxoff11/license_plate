import cv2
import numpy as np

def find_characters(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Список для хранения найденных символов
    characters = []

    for contour in contours:
        # Получаем координаты ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Фильтруем слишком маленькие контуры
            character = image[y:y+h, x:x+w]
            characters.append(character)

    return characters

# Пример использования
characters = find_characters('train/resized_img/A001AA50.png')
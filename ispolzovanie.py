import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

# Загрузка предобученной модели для классификации символов
model = load_model('symbol_recognition_model3.h5')

# Предположим, у вас есть список символов
class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

# # Загрузка конфигурации и весов YOLO
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# # Загрузка классов
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]


def detect_plate_number(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if area < 10000:  # Настройка минимальной и максимальной площади
            plate_contour = contour
            break
        if len(approx) == 4:
            plate_contour = approx
            break
    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = cv2.imread(image_path)[y:y + h, x:x + w]  # Извлекаем цветное изображение номерного знака
        
        # Изменяем размер изображения до 324x70
        plate_image_resized = cv2.resize(plate_image, (324, 70))
        return plate_image_resized  # Возвращаем измененное изображение номерного знака
    return None  # Если рамка не найдена, возвращаем None

def find_license_plate(image):
    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
        return None
    cv2.imshow("1", image) #Исходное изображение

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Применяем размытие
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)  # Адаптивная пороговая обработка

    # # Применяем морфологические операции
    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Закрытие
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)   # Открытие

    cv2.imshow("2", thresh) #Бинарное изображение
    return(thresh)
    # Находим контуры номерного знака
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # license_plate_contour = None

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if 10 < area < 30:  # Настройка минимальной и максимальной площади
    #         license_plate_contour = contour
    #         break

    # if license_plate_contour is not None:
    #     x, y, w, h = cv2.boundingRect(license_plate_contour)
    #     license_plate_image = thresh[y:y+h, x:x+w]  # Извлекаем изображение номерного знака
    #     cv2.imshow("3", license_plate_image) #Изображение номерного знака
    #     return license_plate_image  # Возвращаем изображение номерного знака
    # else:
    #     print("Рамка номерного знака не найдена.")
    #     return None

def segment_characters(thresh):
    # Находим контуры символов и их иерархию
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    characters = []

    # Создаем копию бинарного изображения для отображения контуров
    contour_image = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)  # Создаем черное изображение с 3 каналами

    # Сортируем контуры по координате x
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 15 and h > 20 and w < 60 and h < 80:  # Фильтруем слишком маленькие контуры
            # Извлекаем символы с отступом
            padding = 2  # Размер отступа
            character = thresh[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]  # Извлекаем символы с отступом
            

            character_resized = cv2.resize(character, (28, 28))  # Изменяем размер до 28x28
            characters.append(character_resized)
            
            cv2.imshow("4", character_resized)  # Измененный символ
            cv2.waitKey(500)  # Ожидание 100 мс для просмотра
            
            # Генерируем случайный цвет
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

            # Рисуем контуры на изображении с случайным цветом
            cv2.drawContours(contour_image, [contour], -1, color.tolist(), thickness=cv2.FILLED)  # Заполняем контуры случайным цветом

    # Отображаем контуры
    cv2.imshow("5", contour_image)
    return characters

def classify_characters(characters):
    predictions = []
    for char in characters:
        char = cv2.resize(char, (28, 28))  # Измените порядок размеров
        char = char / 255.0  # Нормализация
        char = np.expand_dims(char, axis=-1)  # Добавляем размерность для каналов
        char = np.repeat(char, 3, axis=-1)  # Дублируем канал, чтобы получить 3 канала
        char = np.expand_dims(char, axis=0)  # Добавляем размерность для батча

        # Предсказание
        pred = model.predict(char)
        predicted_class = np.argmax(pred, axis=1)
        predictions.append(predicted_class)

    return predictions

def convert_predictions(predictions):
    # Преобразуем все предсказания в символы
    result = []
    # skip_next = False  # Флаг для пропуска следующего символа
    
    for i, pred in enumerate(predictions):
        char = class_labels[pred[0]]
        
        # Условия для обработки символов в соответствии с правилами
        # if skip_next:
        #     skip_next = False  # Сбрасываем флаг
        #     continue  # Пропускаем этот символ

        if char == 'O':
            result.append('O')  # Записываем '0' вместо 'O'
            # skip_next = True  # Устанавливаем флаг для пропуска следующего символа
        else:
            result.append(char)  # Добавляем текущий символ

    # Удаляем пустые элементы из результата
    result = [r for r in result if r]
    
    
    # if len(result)==8:
    #     if result[1]=='O':
    #         result[1]='0'
    #     if result[2]=='O':
    #         result[2]='0'
    #     if result[3]=='O':
    #         result[3]='0'
    #     if result[6]=='O':
    #         result[6]='0'
    #     if result[7]=='O':
    #         result[7]='0'
    # if len(result)==9:
    #     if result[1]=='O':
    #         result[1]='0'
    #     if result[2]=='O':
    #         result[2]='0'
    #     if result[3]=='O':
    #         result[3]='0'
    #     if result[6]=='O':
    #         result[6]='0'
    #     if result[7]=='O':
    #         result[7]='0'
    #     if result[8]=='O':
    #         result[8]='0'
    return result

# Основная функция
def main(image_path):
    plate_image = detect_plate_number(image_path)  # Получаем изображение номерного знака
    if plate_image is not None:
        thresh = find_license_plate(plate_image)  # Передаем обработанное изображение в find_license_plate
        if thresh is not None:
            characters = segment_characters(thresh)
            predicted_classes = classify_characters(characters)
            readable_classes = convert_predictions(predicted_classes)
            print("Предсказанные символы:", ''.join(readable_classes))
        else:
            print("Государственный знак не найден.")
    else:
        print("Рамка номерного знака не найдена.")

    cv2.waitKey(0)  # Ожидание нажатия клавиши
    cv2.destroyAllWindows()  # Закрытие всех окон

# Пример использования
main('fullnum/1.jpg')

# def evaluate_model(image_folder):
#     correct_predictions = 0
#     total_images = 0

#     # Проходим по всем изображениям в папке
#     for filename in os.listdir(image_folder):
#         if filename.endswith('.png') or filename.endswith('.jpg'):  # Проверяем расширение файла
#             total_images += 1
#             image_path = os.path.join(image_folder, filename)
#             thresh = find_license_plate(image_path)  # Получаем бинарное изображение
            
#             if thresh is not None:
#                 characters = segment_characters(thresh)  # Передаем бинарное изображение в сегментацию
#                 predicted_classes = classify_characters(characters)
#                 readable_classes = convert_predictions(predicted_classes)
#                 predicted_number = ''.join(readable_classes)  # Объединяем символы в строку
                
#                 # Извлекаем ожидаемое значение из имени файла (без расширения)
#                 expected_number = os.path.splitext(filename)[0]
                
#                 # Сравниваем предсказанное значение с ожидаемым
#                 if predicted_number == expected_number:
#                     correct_predictions += 1
#                     print(f"Правильное предсказание: {predicted_number} для {filename}")
#                 else:
#                     print(f"Неправильное предсказание: {predicted_number} для {filename}, ожидаемое: {expected_number}")
#             else:
#                 print(f"Государственный знак не найден для {filename}")

#     # Выводим результаты
#     print(f"Правильные предсказания: {correct_predictions} из {total_images}")

# # Пример использования
# evaluate_model('100')  # Укажите путь к папке с изображениями




# def detect_license_plate(image_path):
#     # Чтение изображения
#     image = cv2.imread(image_path)
#     height, width, _ = image.shape

#     # Подготовка изображения для YOLO
#     blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(output_layers)

#     boxes = []
#     confidences = []
#     class_ids = []

#     # Обработка выходных данных
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Порог уверенности
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Прямоугольник вокруг объекта
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Удаление дублирующихся прямоугольников
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Отображение результатов
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             if label == "license_plate":  # Проверяем, что это номерной знак
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imshow("Обнаруженные номерные знаки", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Пример использования
# detect_license_plate('fullnum/2.jpg')


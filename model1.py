import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Параметры
image_size = (28, 28)  # Размер изображений
batch_size = 32

# Путь к папке с данными
data_dir = 'C:/Users/Maks/Desktop/imba/PROJECT/dataset'  # Путь к папке с изображениями

# 1. Предобработка данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 2. Генераторы для обучения и тестирования
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=image_size,
    class_mode='sparse',
    batch_size=batch_size
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=image_size,
    class_mode='sparse',
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=image_size,
    class_mode='sparse',
    batch_size=batch_size
)

# Отображение примеров изображений из тренировочного набора
def display_sample_images(generator, num_images=9):
    # Получаем один батч изображений и меток
    images, labels = next(generator)

    # Создаем фигуру для отображения
    plt.figure(figsize=(10, 10))
    
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)  # 3x3 сетка
        plt.imshow(images[i])  # Отображаем изображение
        plt.title(f'Label: {labels[i]}')  # Отображаем метку
        plt.axis('off')  # Отключаем оси

    plt.show()

# Вызов функции для отображения изображений
display_sample_images(train_generator)

# 3. Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Количество классов
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Обучение модели
history = model.fit(train_generator, epochs=100, validation_data=validation_generator)

# 5. Оценка модели
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy}')

# Сохранение модели
model.save('symbol_recognition_model3.h5')

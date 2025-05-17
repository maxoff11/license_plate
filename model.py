import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.callbacks import EarlyStopping



# 1. Подготовка данных
# Путь к папке с изображениями
data_dir = 'train/resized_img'
# Загрузка меток из CSV файла
labels_df = pd.read_csv('datf.csv') 
# Разделение данных на обучающую и тестовую выборки
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)
# 2. Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 200, 3)), 
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # Слой Dropout
    layers.Dense(200, activation='relu'),
    layers.Dense(len(labels_df['label'].unique()), activation='softmax')  # Количество классов
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Колбек для остановки обучения
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Обучение модели
# Генератор изображений для увеличения данных
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'train',  # Папка с обучающими данными
    target_size=(100, 200),  
    class_mode='sparse',
    batch_size=32
)
test_generator = test_datagen.flow_from_directory(
    'test',  # Папка с тестовыми данными
    target_size=(100, 200),
    class_mode='sparse',
    batch_size=32
)
# Обучение модели
history = model.fit(train_generator, epochs=3, validation_data=test_generator, callbacks=[early_stopping])
# 5. Оценка модели
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy}')

# 6. Сохранение модели
model.save('license_plate_model1.h5')

# Получаем истинные метки и предсказания
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Подсчет количества угаданных и общего количества тестовых данных
total_test_samples = len(true_labels)
correct_predictions = np.sum(predicted_classes == true_labels)

print(f'Общее количество тестовых данных: {total_test_samples}')
print(f'Количество угаданных: {correct_predictions}')

# # Создаем матрицу путаницы
# cm = confusion_matrix(true_labels, predicted_classes)

# # Визуализируем матрицу путаницы
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()

# Визуализация предсказаний
def display_predictions(generator, model, num_images=5):
    images, labels = next(generator)
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f'True: {labels[i]}, Pred: {predicted_classes[i]}')
        plt.axis('off')
    plt.show()

# Вызов функции для отображения предсказаний
display_predictions(test_generator, model, num_images=5)
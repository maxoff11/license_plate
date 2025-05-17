import os
import pandas as pd
# Путь к папке с изображениями
data_dir = 'test\img'
# Список для хранения данных
data = []
# Проходим по всем файлам в папке
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):  # Убедитесь, что это изображение
        # Извлекаем номер из имени файла (например, A001AA50.png -> A001AA50)
        number = filename[:-4]  # Убираем .png
        data.append({'filename': filename, 'label': number})
# Создаем DataFrame и сохраняем в CSV
df = pd.DataFrame(data)
df.to_csv('tdatf.csv', index=False)
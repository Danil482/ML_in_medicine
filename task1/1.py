import cv2
import numpy as np
import os
import pandas as pd
import mahotas as mt
from PIL import Image, ImageEnhance


def extract_haralick_features(data_path, distances=[1, 5], gray_levels=256):
    """
    Извлекает признаки Харалика для изображений в папках YES и NO.
    Результаты для каждого расстояния (d) сохраняются в отдельные файлы.

    Parameters:
    data_path (str): Путь к корневой папке DATA
    distances (list): Список расстояний для матрицы смежности (d=1, d=5)
    gray_levels (int): Количество уровней серого (G=256)

    Returns:
    dict: Словарь с DataFrame для каждого комбо (yes/no и d)
    """

    # Создаем DataFrame для каждого комбо
    df_dict = {}
    for folder in ['yes', 'no']:
        for d in distances:
            key = f"{folder}_d{d}"
            df_dict[key] = pd.DataFrame()

    # Папки с изображениями
    yes_path = os.path.join(data_path, 'YES')
    no_path = os.path.join(data_path, 'NO')

    # Функция обработки одного изображения
    def process_image(image_path, distance, levels):
        # Чтение и предобработка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка чтения изображения: {image_path}")
            return None

        # Преобразование в LAB и применение CLAHE
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Преобразование в градации серого
        gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        # Масштабирование изображения до нужного количества уровней серого
        gray_image = (gray_image / 256.0 * levels).astype(np.uint8)

        # Вычисление признаков Харалика
        try:
            textures = mt.features.haralick(gray_image,
                                            distance=distance,
                                            ignore_zeros=True,
                                            return_mean=True)
            return textures
        except Exception as e:
            print(f"Ошибка при вычислении признаков Харалика для {image_path}: {e}")
            return None

    # Обработка изображений YES
    for filename in os.listdir(yes_path):
        if filename.endswith('.bmp'):
            file_path = os.path.join(yes_path, filename)

            for d in distances:
                features = process_image(file_path, d, gray_levels)
                if features is not None:
                    features_dict = {f'har_{i}': feature for i, feature in enumerate(features)}
                    df_dict[f"yes_d{d}"] = pd.concat([df_dict[f"yes_d{d}"], pd.DataFrame([features_dict])],
                                                     ignore_index=True)

    # Обработка изображений NO
    for filename in os.listdir(no_path):
        if filename.endswith('.bmp'):
            file_path = os.path.join(no_path, filename)

            for d in distances:
                features = process_image(file_path, d, gray_levels)
                if features is not None:
                    features_dict = {f'har_{i}': feature for i, feature in enumerate(features)}
                    df_dict[f"no_d{d}"] = pd.concat([df_dict[f"no_d{d}"], pd.DataFrame([features_dict])],
                                                    ignore_index=True)

    # Добавляем метки классов
    for d in distances:
        df_dict[f"yes_d{d}"]['label'] = 1
        df_dict[f"no_d{d}"]['label'] = 0

    # Сохраняем результаты в отдельные файлы
    for key, df in df_dict.items():
        filename = f'haralick_features_{key}.csv'
        df.to_csv(filename, index=False)

    print("Признаки Харалика успешно извлечены и сохранены в CSV файлы (по 2 для YES и NO, разделенные по d)")
    return df_dict


# Пример использования
if __name__ == "__main__":
    data_path = 'C:/Users/dania/PycharmProjects/pythonProject/medicine_ML/DATA'
    df_dict = extract_haralick_features(data_path, distances=[1, 5], gray_levels=256)

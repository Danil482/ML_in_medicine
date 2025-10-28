import cv2
import numpy as np
import os
import pandas as pd
import mahotas as mt
from PIL import Image


def extract_haralick_features(data_path, distances=[1, 5], gray_levels=256):
    """
    Извлекает признаки Харалика по 4 направлениям для каждого расстояния d.
    Для каждого d: 4 направления × 13 признаков = 52 признака.
    Объединяет d=1 и d=5 → 104 признака на образец.
    Сохраняет 4 CSV: yes_d1, yes_d5, no_d1, no_d5.

    Parameters:
    data_path (str): Путь к папке DATA с подпапками YES и NO
    distances (list): Список расстояний d (например, [1, 5])
    gray_levels (int): Количество уровней серого G

    Returns:
    dict: Словарь с DataFrame для каждой комбинации
    """

    # Словарь для хранения результатов
    df_dict = {}
    for folder in ['yes', 'no']:
        for d in distances:
            key = f"{folder}_d{d}"
            df_dict[key] = pd.DataFrame()

    # Папки с изображениями
    yes_path = os.path.join(data_path, 'YES')
    no_path = os.path.join(data_path, 'NO')

    # Проверка существования папок
    if not os.path.exists(yes_path):
        raise FileNotFoundError(f"Папка не найдена: {yes_path}")
    if not os.path.exists(no_path):
        raise FileNotFoundError(f"Папка не найдена: {no_path}")

    # Функция предобработки изображения
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка чтения: {image_path}")
            return None

        # LAB + CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        # Масштабирование до G уровней
        gray_scaled = (gray / 255.0 * (gray_levels - 1)).astype(np.uint8)
        return gray_scaled

    # Вычисление признаков Харалика по 4 направлениям
    def compute_haralick_4dir(gray_image, distance):
        features_list = []
        for angle in [0, 1, 2, 3]:  # 0° (0), 45° (1), 90° (2), 135° (3)
            try:
                textures = mt.features.haralick(
                    gray_image,
                    distance=distance,
                    ignore_zeros=True,
                    return_mean=False  # Получаем 13 признаков для каждого направления
                )
                features_list.append(textures[angle])  # Берем вектор для текущего направления
            except Exception as e:
                print(f"Ошибка для угла {angle*45}°: {e}")
                features_list.append(np.zeros(13))  # Заполняем нулями при ошибке
        return np.concatenate(features_list)  # 4 × 13 = 52 признака

    # Обработка изображений
    for folder, path in [('yes', yes_path), ('no', no_path)]:
        label = 1 if folder == 'yes' else 0
        print(f"Обработка папки: {folder.upper()} ({len(os.listdir(path))} изображений)")

        for filename in os.listdir(path):
            if filename.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg')):
                file_path = os.path.join(path, filename)
                gray_image = preprocess_image(file_path)
                if gray_image is None:
                    continue

                # Для каждого расстояния d
                for d in distances:
                    features_52 = compute_haralick_4dir(gray_image, d)
                    feature_names = [f'har_{i}_d{d}' for i in range(52)]
                    features_dict = dict(zip(feature_names, features_52))
                    features_dict['label'] = label

                    # Добавляем в соответствующий DataFrame
                    key = f"{folder}_d{d}"
                    df_dict[key] = pd.concat([
                        df_dict[key],
                        pd.DataFrame([features_dict])
                    ], ignore_index=True)

    # Сохраняем в CSV
    for key, df in df_dict.items():
        filename = f'haralick_features_{key}.csv'
        df.to_csv(filename, index=False)
        print(f"Сохранено: {filename} — {len(df)} образцов, {len(df.columns)-1} признаков")

    print(f"\nГотово! Извлечено признаков: 4 направления × 13 признаков × {len(distances)} d = "
          f"{4 * 13 * len(distances)} на образец (всего {len(df_dict['yes_d1']) + len(df_dict['no_d1'])} образцов)")
    return df_dict


# === ЗАПУСК ===
if __name__ == "__main__":
    data_path = 'C:/Users/dania/PycharmProjects/pythonProject/medicine_ML/DATA'
    df_dict = extract_haralick_features(data_path, distances=[1, 5], gray_levels=256)
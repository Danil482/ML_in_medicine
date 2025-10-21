import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Определение custom scorer для специфичности
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


# Scorers для cross_validate
scorers = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',  # Recall для положительного класса (label=1)
    'specificity': make_scorer(specificity)
}


# Функция для загрузки и объединения данных для d=1 и d=5 с префиксом 'contrast_'
def load_combined_data_with_contrast():
    # Загружаем данные
    try:
        df_yes_d1 = pd.read_csv('contrast_haralick_features_yes_d1.csv')
        df_no_d1 = pd.read_csv('contrast_haralick_features_no_d1.csv')
        df_yes_d5 = pd.read_csv('contrast_haralick_features_yes_d5.csv')
        df_no_d5 = pd.read_csv('contrast_haralick_features_no_d5.csv')
    except FileNotFoundError as e:
        print(f"Ошибка: Один или несколько файлов не найдены. {e}")
        return None, None

    # Проверяем, что столбцы с признаками совпадают по количеству
    if df_yes_d1.shape[1] != df_yes_d5.shape[1] or df_no_d1.shape[1] != df_no_d5.shape[1]:
        print("Ошибка: Разное количество признаков в файлах d=1 и d=5")
        return None, None

    # Переименовываем столбцы, чтобы различать d=1 и d=5
    feature_columns = [col for col in df_yes_d1.columns if col != 'label']
    df_yes_d1 = df_yes_d1.rename(columns={col: f"{col}_d1" for col in feature_columns})
    df_no_d1 = df_no_d1.rename(columns={col: f"{col}_d1" for col in feature_columns})
    df_yes_d5 = df_yes_d5.rename(columns={col: f"{col}_d5" for col in feature_columns})
    df_no_d5 = df_no_d5.rename(columns={col: f"{col}_d5" for col in feature_columns})

    # Проверяем, что количество строк совпадает
    if len(df_yes_d1) != len(df_yes_d5) or len(df_no_d1) != len(df_no_d5):
        print("Ошибка: Разное количество образцов в файлах d=1 и d=5")
        return None, None

    # Объединяем признаки по строкам (предполагая, что порядок изображений одинаков)
    df_yes = pd.concat([df_yes_d1.drop('label', axis=1), df_yes_d5], axis=1)
    df_no = pd.concat([df_no_d1.drop('label', axis=1), df_no_d5], axis=1)

    # Добавляем метку класса
    df_yes['label'] = df_yes_d1['label']
    df_no['label'] = df_no_d1['label']

    # Объединяем YES и NO
    df = pd.concat([df_yes, df_no], ignore_index=True)

    # Разделяем признаки и метки
    X = df.drop('label', axis=1)
    y = df['label']

    # Масштабирование признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# Список моделей с именами
def get_models():
    models = []

    # Логистическая регрессия
    models.append(('LogisticRegression', LogisticRegression(max_iter=2000)))  # Увеличен max_iter

    # SVM варианты
    models.append(('SVC', SVC()))
    models.append(('NuSVC', NuSVC()))
    models.append(('LinearSVC', LinearSVC(max_iter=20000, dual=False)))  # Увеличен max_iter, dual=False

    # Random Forest с разными деревьями
    for n in [20, 30, 40, 50, 60]:
        models.append((f'RandomForest_{n}', RandomForestClassifier(n_estimators=n, random_state=42)))

    # kNN с разными соседями
    for k in [2, 3, 4, 5, 6]:
        models.append((f'KNN_{k}', KNeighborsClassifier(n_neighbors=k)))

    return models


# Функция для проведения классификации и вычисления средних метрик
def evaluate_models(X, y, ks=[3, 4, 5]):
    models = get_models()
    results = []

    for name, model in models:
        mean_acc = []
        mean_sens = []
        mean_spec = []

        for k in ks:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            scores = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)

            mean_acc.append(np.mean(scores['test_accuracy']))
            mean_sens.append(np.mean(scores['test_sensitivity']))
            mean_spec.append(np.mean(scores['test_specificity']))

        # Усредняем по всем K
        avg_acc = np.mean(mean_acc)
        avg_sens = np.mean(mean_sens)
        avg_spec = np.mean(mean_spec)

        results.append({
            'model': name,
            'avg_accuracy': avg_acc,
            'avg_sensitivity': avg_sens,
            'avg_specificity': avg_spec
        })

    return pd.DataFrame(results)


# Основная функция
def run_classification_with_contrast():
    print("\n=== Оценка для объединенных признаков с повышенной контрастностью (d=1 и d=5, G=256) ===")

    X, y = load_combined_data_with_contrast()
    if X is None:
        return

    print(f"Размер данных: {X.shape[0]} образцов, {X.shape[1]} признаков")

    results_df = evaluate_models(X, y)

    # Топ-3 по accuracy
    top_acc = results_df.sort_values('avg_accuracy', ascending=False).head(3)
    print("\nТоп-3 модели по средней точности (accuracy):")
    print(top_acc[['model', 'avg_accuracy']].to_string(index=False))

    # Топ-3 по sensitivity
    top_sens = results_df.sort_values('avg_sensitivity', ascending=False).head(3)
    print("\nТоп-3 модели по средней чувствительности (sensitivity):")
    print(top_sens[['model', 'avg_sensitivity']].to_string(index=False))

    # Топ-3 по specificity
    top_spec = results_df.sort_values('avg_specificity', ascending=False).head(3)
    print("\nТоп-3 модели по средней специфичности (specificity):")
    print(top_spec[['model', 'avg_specificity']].to_string(index=False))

    # Сохраняем результаты в CSV
    results_df.to_csv('classification_results_combined_contrast.csv', index=False)
    print("\nРезультаты сохранены в 'classification_results_combined_contrast.csv'")


# Запуск
if __name__ == "__main__":
    run_classification_with_contrast()
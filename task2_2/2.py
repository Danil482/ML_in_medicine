import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import LinearSVC
from scipy.stats import mannwhitneyu, entropy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler
from medicine_ML.task1.task1_1 import extract_haralick_features


# Шаг 2: Загрузка и объединение данных
def load_combined_data_original():
    try:
        df_yes_d1 = pd.read_csv('haralick_features_yes_d1.csv')
        df_no_d1 = pd.read_csv('haralick_features_no_d1.csv')
        df_yes_d5 = pd.read_csv('haralick_features_yes_d5.csv')
        df_no_d5 = pd.read_csv('haralick_features_no_d5.csv')
    except FileNotFoundError as e:
        print(f"Ошибка: Один или несколько файлов не найдены. {e}")
        return None, None

    # Проверяем наличие пропущенных значений
    for df, name in [(df_yes_d1, 'yes_d1'), (df_no_d1, 'no_d1'), (df_yes_d5, 'yes_d5'), (df_no_d5, 'no_d5')]:
        if df.isnull().any().any():
            print(f"Предупреждение: Пропущенные значения в {name}. Заполняем нулями.")
            df.fillna(0, inplace=True)

    feature_columns = [col for col in df_yes_d1.columns if col != 'label']
    df_yes_d1 = df_yes_d1.rename(columns={col: f"{col}_d1" for col in feature_columns})
    df_no_d1 = df_no_d1.rename(columns={col: f"{col}_d1" for col in feature_columns})
    df_yes_d5 = df_yes_d5.rename(columns={col: f"{col}_d5" for col in feature_columns})
    df_no_d5 = df_no_d5.rename(columns={col: f"{col}_d5" for col in feature_columns})

    if len(df_yes_d1) != len(df_yes_d5) or len(df_no_d1) != len(df_no_d5):
        print("Ошибка: Разное количество образцов в файлах d=1 и d=5")
        return None, None

    df_yes = pd.concat([df_yes_d1.drop('label', axis=1), df_yes_d5], axis=1)
    df_no = pd.concat([df_no_d1.drop('label', axis=1), df_no_d5], axis=1)

    df_yes['label'] = df_yes_d1['label']
    df_no['label'] = df_no_d1['label']

    df = pd.concat([df_yes, df_no], ignore_index=True)

    X = df.drop('label', axis=1)
    y = df['label']

    # Масштабирование с сохранением DataFrame
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    return X_scaled, y

# Custom функции для методов из статьи
def wilcoxon_score(X, y):
    scores = []
    X = pd.DataFrame(X)
    y = pd.Series(y)
    for col in X.columns:
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        try:
            stat, _ = mannwhitneyu(group0, group1, alternative='two-sided')
            scores.append(stat)
        except ValueError:
            scores.append(0)  # Если группы слишком малы или идентичны
    return np.array(scores)

def bhattacharyya_distance(X, y):
    scores = []
    X = pd.DataFrame(X)
    y = pd.Series(y)
    for col in X.columns:
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        mu0, sigma0 = np.mean(group0), np.std(group0)
        mu1, sigma1 = np.mean(group1), np.std(group1)
        # Избегаем деления на ноль
        if sigma0 == 0 or sigma1 == 0:
            scores.append(0)
        else:
            dist = 0.25 * np.log(0.25 * (sigma0**2 / sigma1**2 + sigma1**2 / sigma0**2 + 2)) + \
                   0.25 * ((mu0 - mu1)**2 / (sigma0**2 + sigma1**2))
            scores.append(dist)
    return np.array(scores)

def relative_entropy(X, y, bins=10):
    scores = []
    X = pd.DataFrame(X)
    y = pd.Series(y)
    for col in X.columns:
        group0 = X[col][y == 0]
        group1 = X[col][y == 1]
        try:
            hist0, edges = np.histogram(group0, bins=bins, density=True)
            hist1, _ = np.histogram(group1, bins=edges, density=True)
            hist0 = np.where(hist0 == 0, 1e-10, hist0)
            hist1 = np.where(hist1 == 0, 1e-10, hist1)
            kl_div = entropy(hist0, hist1) + entropy(hist1, hist0)  # Symmetric KL
            scores.append(kl_div)
        except ValueError:
            scores.append(0)  # Если гистограмма не может быть построена
    return np.array(scores)

# Методы отбора признаков
def feature_selection_methods(X, y, k=13):  # Отбираем топ-13 из 26
    methods = {}

    # Проверка, что X - это DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X должен быть pandas.DataFrame")

    # Создаем словарь для хранения индексов столбцов
    feature_indices = {col: idx for idx, col in enumerate(X.columns)}

    # 1. Student's t-test (using f_classif as equivalent for two classes)
    try:
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        methods['t_test'] = {
            'columns': selected_cols,
            'indices': [feature_indices[col] for col in selected_cols]
        }
    except ValueError as e:
        print(f"Ошибка в t_test: {e}")
        methods['t_test'] = {
            'columns': X.columns[:k],
            'indices': list(range(k))
        }

    # 2. Wilcoxon sum-rank test
    try:
        selector = SelectKBest(wilcoxon_score, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        methods['wilcoxon'] = {
            'columns': selected_cols,
            'indices': [feature_indices[col] for col in selected_cols]
        }
    except ValueError as e:
        print(f"Ошибка в wilcoxon: {e}")
        methods['wilcoxon'] = {
            'columns': X.columns[:k],
            'indices': list(range(k))
        }

    # 3. Bhattacharyya distance
    try:
        selector = SelectKBest(bhattacharyya_distance, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        methods['bhattacharyya'] = {
            'columns': selected_cols,
            'indices': [feature_indices[col] for col in selected_cols]
        }
    except ValueError as e:
        print(f"Ошибка в bhattacharyya: {e}")
        methods['bhattacharyya'] = {
            'columns': X.columns[:k],
            'indices': list(range(k))
        }

    # 4. Relative entropy (KL divergence)
    try:
        selector = SelectKBest(relative_entropy, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        methods['relative_entropy'] = {
            'columns': selected_cols,
            'indices': [feature_indices[col] for col in selected_cols]
        }
    except ValueError as e:
        print(f"Ошибка в relative_entropy: {e}")
        methods['relative_entropy'] = {
            'columns': X.columns[:k],
            'indices': list(range(k))
        }

    # 5. SVM RFE
    try:
        estimator = LinearSVC(max_iter=20000, dual=False)
        selector = RFE(estimator, n_features_to_select=k, step=0.5)
        selector.fit(X, y)
        selected_cols = X.columns[selector.support_]
        methods['svm_rfe'] = {
            'columns': selected_cols,
            'indices': [feature_indices[col] for col in selected_cols]
        }
    except ValueError as e:
        print(f"Ошибка в svm_rfe: {e}")
        methods['svm_rfe'] = {
            'columns': X.columns[:k],
            'indices': list(range(k))
        }

    return methods

# Определение scorers
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

scorers = {
    'accuracy': 'accuracy',
    'sensitivity': 'recall',
    'specificity': make_scorer(specificity)
}

# Модели
def get_models():
    models = []
    models.append(('LogisticRegression', LogisticRegression(max_iter=2000)))
    models.append(('SVC', SVC()))
    models.append(('NuSVC', NuSVC()))
    models.append(('LinearSVC', LinearSVC(max_iter=20000, dual=False)))
    for n in [20, 30, 40, 50, 60]:
        models.append((f'RandomForest_{n}', RandomForestClassifier(n_estimators=n, random_state=42)))
    for k in [2, 3, 4, 5, 6]:
        models.append((f'KNN_{k}', KNeighborsClassifier(n_neighbors=k)))
    return models

# Оценка моделей для набора признаков
def evaluate_models_for_features(X_selected, y, ks=[3, 4, 5]):
    models = get_models()
    results = []
    for name, model in models:
        mean_acc, mean_sens, mean_spec = [], [], []
        for k_fold in ks:
            cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
            scores = cross_validate(model, X_selected, y, cv=cv, scoring=scorers, n_jobs=-1)
            mean_acc.append(np.mean(scores['test_accuracy']))
            mean_sens.append(np.mean(scores['test_sensitivity']))
            mean_spec.append(np.mean(scores['test_specificity']))
        results.append({
            'model': name,
            'avg_accuracy': np.mean(mean_acc),
            'avg_sensitivity': np.mean(mean_sens),
            'avg_specificity': np.mean(mean_spec)
        })
    return pd.DataFrame(results)

# Основная функция
def run_feature_selection_and_classification(data_path):
    # Извлечение признаков
    extract_haralick_features(data_path)

    # Загрузка данных
    X, y = load_combined_data_original()
    if X is None:
        return

    # Проверка данных
    print(f"Размер данных: {X.shape[0]} образцов, {X.shape[1]} признаков")
    if X.isnull().any().any():
        print("Предупреждение: Пропущенные значения в X. Заполняем нулями.")
        X.fillna(0, inplace=True)

    # Проверка типа X
    if not isinstance(X, pd.DataFrame):
        print("Ошибка: X не является DataFrame")
        return

    # Отбор признаков
    selected_features = feature_selection_methods(X, y)

    all_results = {}
    for method, info in selected_features.items():
        print(f"\n=== Оценка для метода отбора: {method} (отобрано {len(info['columns'])} признаков) ===")
        print(f"Выбранные признаки: {[f'{col} (index: {idx})' for col, idx in zip(info['columns'], info['indices'])]}")
        X_selected = X[info['columns']]

        results_df = evaluate_models_for_features(X_selected, y)
        all_results[method] = results_df

        # Топ-3 по accuracy
        top_acc = results_df.sort_values('avg_accuracy', ascending=False).head(3)
        print("\nТоп-3 модели по средней точности:")
        print(top_acc[['model', 'avg_accuracy']].to_string(index=False))

        # Топ-3 по sensitivity
        top_sens = results_df.sort_values('avg_sensitivity', ascending=False).head(3)
        print("\nТоп-3 модели по средней чувствительности:")
        print(top_sens[['model', 'avg_sensitivity']].to_string(index=False))

        # Топ-3 по specificity
        top_spec = results_df.sort_values('avg_specificity', ascending=False).head(3)
        print("\nТоп-3 модели по средней специфичности:")
        print(top_spec[['model', 'avg_specificity']].to_string(index=False))

        # Сохраняем результаты
        results_df.to_csv(f'classification_results_{method}.csv', index=False)

    # Сравнение методов: находим лучший по средней accuracy топ-модели
    comparison = []
    for method, df in all_results.items():
        best_acc = df['avg_accuracy'].max()
        comparison.append({'method': method, 'best_accuracy': best_acc})
    comparison_df = pd.DataFrame(comparison).sort_values('best_accuracy', ascending=False)
    print("\n=== Сравнение методов отбора по максимальной accuracy ===")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('feature_selection_comparison.csv', index=False)


# Запуск
if __name__ == "__main__":
    data_path = 'C:/Users/dania/PycharmProjects/pythonProject/medicine_ML/DATA'
    run_feature_selection_and_classification(data_path)
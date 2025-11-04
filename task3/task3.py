# task3_perceptron.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ============================
# 1. ЗАГРУЗКА ВСЕХ ПРИЗНАКОВ
# ============================
print("Загрузка признаков Харалика (104 шт.)...")

df_yes_d1 = pd.read_csv('haralick_features_yes_d1.csv')
df_yes_d5 = pd.read_csv('haralick_features_yes_d5.csv')
df_no_d1 = pd.read_csv('haralick_features_no_d1.csv')
df_no_d5 = pd.read_csv('haralick_features_no_d5.csv')

# Объединяем d=1 и d=5
X_yes = pd.concat([df_yes_d1.drop('label', axis=1), df_yes_d5.drop('label', axis=1)], axis=1)
X_no = pd.concat([df_no_d1.drop('label', axis=1), df_no_d5.drop('label', axis=1)], axis=1)

y_yes = np.ones(len(X_yes))
y_no = np.zeros(len(X_no))

X = np.vstack([X_yes, X_no])
y = np.hstack([y_yes, y_no])

print(f"Готово: {X.shape[0]} образцов × {X.shape[1]} признаков")

# ============================
# 2. ПЕРЕБОР РАЗМЕРА СКРЫТОГО СЛОЯ
# ============================
hidden_sizes = list(range(4, 105, 8))  # [4, 12, 20, ..., 96, 104]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("\nЗапуск обучения...")
for size in hidden_sizes:
    accs, sens, specs = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=(size,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            alpha=1e-4
        )
        mlp.fit(X_tr, y_tr)
        pred = mlp.predict(X_te)

        accs.append(accuracy_score(y_te, pred))
        sens.append(recall_score(y_te, pred, pos_label=1))
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    results.append({
        'neurons': size,
        'accuracy': round(np.mean(accs), 4),
        'sensitivity': round(np.mean(sens), 4),
        'specificity': round(np.mean(specs), 4)
    })
    print(f"{size:3d} нейронов → acc: {np.mean(accs):.4f} | sens: {np.mean(sens):.4f} | spec: {np.mean(specs):.4f}")

# ============================
# 3. СОХРАНЕНИЕ И ГРАФИК
# ============================
df_res = pd.DataFrame(results)
df_res.to_csv('perceptron_hidden_layer_results.csv', index=False)
print("\nТаблица сохранена → perceptron_hidden_layer_results.csv")

# Поиск лучшего
best = df_res.loc[df_res['accuracy'].idxmax()]
print(f"\nЛУЧШИЙ РЕЗУЛЬТАТ: {best['neurons']} нейронов")
print(f"Accuracy: {best['accuracy']}, Sensitivity: {best['sensitivity']}, Specificity: {best['specificity']}")

# График
plt.figure(figsize=(10, 6))
plt.plot(df_res['neurons'], df_res['accuracy'], 'o-', label='Accuracy', color='blue')
plt.plot(df_res['neurons'], df_res['sensitivity'], 's-', label='Sensitivity', color='green')
plt.plot(df_res['neurons'], df_res['specificity'], '^-', label='Specificity', color='red')
plt.xlabel('Количество нейронов в скрытом слое')
plt.ylabel('Метрика (среднее по 5-fold CV)')
plt.title('Зависимость качества персептрона от размера скрытого слоя\n(104 признака Харалика, без отбора)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(hidden_sizes)
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig('perceptron_hidden_layer_plot.png', dpi=200)
plt.show()

print("График сохранён → perceptron_hidden_layer_plot.png")
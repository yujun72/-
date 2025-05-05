import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_titanic

# 建立資料夾（避免 FileNotFoundError）
os.makedirs('model', exist_ok=True)

# 載入資料
df, (X, y) = preprocess_titanic('train.csv')

# 檢查資料中是否有缺失值
print("Missing values in X:\n", X.isnull().sum())
print("Missing values in y:\n", y.isnull().sum())

# 如果有缺失值，可以考慮進行填補
# X.fillna(X.mean(), inplace=True)  # 這裡是填補數值型特徵的缺失值
# 或者使用其他方法來處理缺失值

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 設定模型與超參數搜尋範圍
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用 GridSearchCV 搜尋最佳超參數
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)

# 輸出最佳參數與分類報告
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("model/confusion_matrix.png")
plt.show()

# 儲存最佳模型
joblib.dump(best_model, 'model/titanic_best_model.pkl')

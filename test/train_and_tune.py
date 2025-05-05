# train_and_tune.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_titanic

df, _ = preprocess_titanic('train.csv')
X = df.drop(columns=['Survived'])
y = df['Survived']

# GridSearch 調參（以 RF 為例）
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X, y)

print("Best Params:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)

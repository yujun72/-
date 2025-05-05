import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_titanic(filepath):
    df = pd.read_csv(filepath)

    # 處理缺失值
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 編碼分類欄位
    label_encoders = {}
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 選擇要用的欄位
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    
    X = df[features]
    y = df[target]

    return df, (X, y)

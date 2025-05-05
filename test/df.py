import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 產生分類與回歸模擬資料
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)

# 整合成 DataFrame
df = pd.DataFrame(X_class, columns=['Age', 'Income', 'PurchaseFreq', 'Membership'])
df['Segment'] = y_class
df['PurchaseAmount'] = y_reg
df['Gender'] = np.random.choice(['Male', 'Female'], size=1000)

# 分群（非必要但推薦）
X_cluster = df[['Age', 'Income', 'PurchaseFreq']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

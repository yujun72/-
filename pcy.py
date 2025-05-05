import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 設定風格
sns.set(style="whitegrid")

# 建立分類資料（用於模擬特徵）
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# 建立回歸目標值
X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)

# 整合成 DataFrame
df = pd.DataFrame(X_class, columns=['Age', 'Income', 'PurchaseFreq', 'Membership'])
df['Segment'] = y_class
df['PurchaseAmount'] = y_reg
df['Gender'] = np.random.choice(['Male', 'Female'], size=1000)

# 選擇分群使用的欄位
X_cluster = df[['Age', 'Income', 'PurchaseFreq']]

# 執行 KMeans (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

# PCA 降維
pca = PCA(n_components=2)
components = pca.fit_transform(X_cluster)
df['PC1'] = components[:, 0]
df['PC2'] = components[:, 1]

# 繪圖
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60, alpha=0.8)
plt.title('PCA 可視化：KMeans 分群 (k=3)', fontsize=14)
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend(title='群編號')
plt.tight_layout()
plt.show()

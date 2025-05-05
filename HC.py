import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

plt.rcParams['font.family'] = 'Microsoft JhengHei'
hours = list(range(24)) * 7
traffic = [50 + 30*np.sin((h-7)/12*np.pi) + np.random.normal(0, 5) for h in hours]
df = pd.DataFrame({"hour": hours, "traffic": traffic})
X = StandardScaler().fit_transform(df[["hour", "traffic"]])
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 4))
dendrogram(linked, truncate_mode='lastp', p=10)
plt.title("階層式聚類樹狀圖")
plt.grid(True)
plt.show()

model = AgglomerativeClustering(n_clusters=3)
df["cluster"] = model.fit_predict(X)

plt.figure(figsize=(10, 4))
plt.scatter(df.index, df["traffic"], c=df["cluster"], cmap="tab10")
plt.title("車流量分群（Hierarchical）")
plt.grid(True)
plt.show()

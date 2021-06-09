import numpy as np
from sklearn.cluster import KMeans

# Explicitly setting initial points to match those given in Q1 - normally you would not do this for KMeans.
# It's generally better to let sklearn handle the initialisation and set a fixed random_state if you need reproducability.
points = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
initials=np.array([[1],[2]])

clusters = KMeans(n_clusters=2, init=initials).fit(points)
print(clusters.cluster_centers_)
print(clusters.labels_)
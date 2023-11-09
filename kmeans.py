import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data
X = np.array([7, 5, 10, 4, 3, 11, 14, 6, 13, 12])
Y = np.array([18, 25, 14, 20, 23, 16, 13, 22, 9, 11])

# Gabungkan X dan Y sebagai features
data = np.column_stack((X, Y))

# Nilai awal pusat cluster
initial_centroids = np.array([[7, 12], [10, 7]])

# Jumlah kluster yang diinginkan
n_clusters = 2

# Inisialisasi model KMeans dengan nilai awal pusat cluster yang ditentukan
kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=42)

# Melakukan clustering
kmeans.fit(data)

# Mendapatkan label kluster untuk setiap titik data
labels = kmeans.labels_

# Mendapatkan posisi centroid
centroids = kmeans.cluster_centers_

# Membuat plot untuk visualisasi clustering
plt.scatter(X, Y, c=labels, cmap='viridis', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering with Initial Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

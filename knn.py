import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Data set
x_train = np.array([7, 5, 10, 4, 3, 11, 14, 6, 13, 12])
y_train = np.array([18, 25, 14, 20, 23, 16, 13, 22, 9, 11])
# 0 = kurang, 1 = cukup, 2 = baik
categories = np.array([1, 2, 1, 2, 2, 1, 0, 2, 0, 0])

# Inisialisasi model KNN dengan K=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Melatih model
knn_model.fit(list(zip(x_train, y_train)), categories)

# Data baru
x_new = np.array([7])
y_new = np.array([19])

# Melakukan prediksi kategori untuk data baru
prediction = knn_model.predict([(x_new[0], y_new[0])])
if prediction==0:
    prediction = 'kurang'
elif prediction==1:
    prediction = 'cukup'
elif prediction==2:
    prediction = 'baik'

# Visualisasi data set
plt.scatter(x_train, y_train, c=categories, cmap='viridis', label='Data Set')
plt.scatter(x_new, y_new, c='red', marker='x', label='Data Baru')

# Menampilkan prediksi pada grafik
plt.text(x_new, y_new, f' Prediksi: {prediction}', fontsize=12, ha='right')

# Menambahkan label dan legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')

# Menampilkan grafik
plt.show()

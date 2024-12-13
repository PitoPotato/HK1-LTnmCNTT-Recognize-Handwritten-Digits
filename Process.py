#-------------libraries-------------

import numpy as np
import tensorflow as tf
from collections import Counter
#---------------Files---------------
import Globals



#------------------------------------


# Cách 1
def vectorization(array):
    newArray = array.reshape(-1, Globals.IMG_SIZE * Globals.IMG_SIZE)
    newArray = np.array([each.flatten() for each in array])
    return newArray
 
# Cách 2
def AveragePooling(array):
    pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
    array = pooling_layer(array.reshape(-1, 28, 28, 1)).numpy()
    array = array.reshape(array.shape[0], -1)
    return array
 
# Cách 3
def calculate_histogram(array, num_bins=32):
    histograms = np.zeros((array.shape[0], num_bins), dtype=np.float64)
    for i, img in enumerate(array):
        hist, _ = np.histogram(img, bins=num_bins, range=(0, 256))
        histograms[i] = hist
    return histograms

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
 
def knn(x_train, y_train, x_test, k):
    predictions = []
    for test_sample in x_test:
        distances = euclidean_distance(x_train, test_sample)
        
        #Chỗ này có thể tối ưu = cách tự code hàm chỉ lấy ra k thằng nhỏ nhất (không cần sort lại cả mảng 60k thành phần)
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
 
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
 
    return np.array(predictions)

def new_knn(x_train, y_train, x_test, k):
    predictions = []
    for test_sample in x_test:

        distances = np.linalg.norm(x_train - test_sample, axis=1)
        
        k_indices = np.argpartition(distances, k)[:k]
        k_nearest_labels = y_train[k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    
    return np.array(predictions)

def split_by_label(x_train, y_train):
    labels = np.unique(y_train)
    split_data = {label: x_train[y_train == label] for label in labels}
    return split_data
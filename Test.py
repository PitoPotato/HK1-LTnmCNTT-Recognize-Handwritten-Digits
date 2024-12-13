import gzip
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
 
 
IMG_SIZE = 28
 
def load_mnist(path, kind):
    label_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    image_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)
 
    with gzip.open(label_path, "rb") as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
 
    with gzip.open(image_path, "rb") as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = (
            np.frombuffer(buffer, dtype=np.uint8)
            .reshape(len(labels), IMG_SIZE, IMG_SIZE)
            .astype(np.float64)
        )
 
    return images, labels
 
# Cách 1
def vectorization(array):
    newArray = array.reshape(-1, IMG_SIZE * IMG_SIZE)
    newArray = np.array([each.flatten() for each in array])
    return newArray
 
# Cách 2
def AvergagePooling(array):
    pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
    array = pooling_layer(array.reshape(-1, 28, 28, 1)).numpy()
    array = array.reshape(array.shape[0], -1)
    return array
 
# Cách 3
def calculate_histogram(array):
    histograms = np.zeros((array.shape[0], 256), dtype=np.float64)
    for i, img in enumerate(array):
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
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
 
def split_by_label(x_train, y_train):
    labels = np.unique(y_train)
    split_data = {label: x_train[y_train == label] for label in labels}
    return split_data
 
def display_images_with_predictions(x_test, y_test, y_pred, image_size=7):
    fig, ax = plt.subplots(
        nrows=6,
        ncols=5,
        sharex=True,
        sharey=True,
        figsize=(15, 15)
    )
 
    ax = ax.flatten()
 
    for i in range(30): 
        img = x_test[i].reshape(image_size, image_size) 
        label = y_test[i]
        prediction = y_pred[i]
        ax[i].set_title(f"True: {label}\nPred: {prediction}", fontsize=10)
        ax[i].imshow(img, cmap="Greys", interpolation="nearest")
        ax[i].axis("off")
 
    plt.tight_layout()
    plt.show()
 
x_train, y_train = load_mnist(r"Databases/", kind="train")
x_test, y_test = load_mnist(r"Databases/", kind="t10k")
 
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
 
# Cách 1: Vectorization 
x_train = vectorization(x_train)
x_test = vectorization(x_test)
 
# Cách 2: Average Pooling
#x_train = AvergagePooling(x_train)
#x_test = AvergagePooling(x_test)
 
# Cách 3: Histogram 
# x_train_256 = calculate_histogram(x_train)
# x_test_256 = calculate_histogram(x_test)
 
k = 100
y_pred = knn(x_train, y_train, x_test[:50], k)


accuracy = accuracy_score(y_test[:50], y_pred)

print(f"Độ chính xác trên 50 mẫu test đầu tiên: {accuracy * 100:.2f}%")
 
display_images_with_predictions(x_test, y_test, y_pred, 28)
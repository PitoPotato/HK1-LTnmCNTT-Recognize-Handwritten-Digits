#-------------libraries-------------
import gzip
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
#---------------Files---------------
import Globals
import InOut
import Process



#------------------------------------
 
x_train, y_train = InOut.load_mnist(r"Databases/", kind="train")
x_test, y_test = InOut.load_mnist(r"Databases/", kind="t10k")
 
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

x_train1 = Process.vectorization(x_train)
x_train2 = Process.AveragePooling(x_train)
x_train3 = Process.calculate_histogram(x_train)
x_test1 = Process.vectorization(x_test)
x_test2 = Process.AveragePooling(x_test)
x_test3 = Process.calculate_histogram(x_test)
 
 
k_values = list(range(1, 50, 1))
accuracies1 = []
accuracies2 = []
accuracies3 = []
 
print("Starting KNN evaluations...")
for k in k_values:
    print(f"Evaluating for k={k}...")
    y_pred1 = Process.new_knn(x_train1, y_train, x_test1, k)
    y_pred2 = Process.new_knn(x_train2, y_train, x_test2, k)
    y_pred3 = Process.new_knn(x_train3, y_train, x_test3, k)
 
    acc1 = (y_pred1 == y_test).mean() * 100
    acc2 = (y_pred2 == y_test).mean() * 100
    acc3 = (y_pred3 == y_test).mean() * 100
 
    accuracies1.append(acc1)
    accuracies2.append(acc2)
    accuracies3.append(acc3)

 
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies1, marker='o', linestyle='-', color='b', label="Vectorization")
plt.plot(k_values, accuracies2, marker='s', linestyle='--', color='r', label="Average Pooling")
plt.plot(k_values, accuracies3, marker='^', linestyle=':', color='g', label="Histogram")
 
 
plt.xlabel("k (Number of Neighbors)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Accuracy of modal.", fontsize=14)
 
 
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()
 

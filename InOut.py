#-------------libraries-------------
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
#---------------Files---------------
import Globals


#------------------------------------
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
            .reshape(len(labels), Globals.IMG_SIZE, Globals.IMG_SIZE)
            .astype(np.float64)
        )
 
    return images, labels

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
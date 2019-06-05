"""
Michael S. Emanuel
Tue Jun  4 16:29:19 2019
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(f'TF version = {tf.__version__}')

fashion_mnist = keras.datasets.fashion_mnist

(img_trn, label_trn), (img_tst, label_tst) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f'Shape of img_trn = {img_trn.shape}')
print(f'Length of label_trn = {len(label_trn)}')

plt.figure()
plt.imshow(img_trn[0])
plt.colorbar()
plt.grid(False)
plt.show()
img_trn = img_trn / 255.0
img_tst = img_tst / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_trn[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[label_trn[i]], fontsize='10')
plt.show()


model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
        ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(img_trn, label_trn, epochs=10)

loss_tst, acc_tst = model.evaluate(img_tst, label_tst)
print(f'Test loss={loss_tst:0.3f}, accuracy={acc_tst:0.3f}')

pred = model.predict(img_tst)

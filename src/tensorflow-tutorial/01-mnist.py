"""
Michael S. Emanuel
Tue Jun  4 16:16:57 2019
"""

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_trn, y_trn, epochs=5)
model.evaluate(x_tst, y_tst)

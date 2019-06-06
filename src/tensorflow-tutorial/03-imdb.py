"""
Michael S. Emanuel
Tue Jun  4 21:19:04 2019
"""

import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Review TF environment
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# Split data into train, validation and test
split_trn_val = tfds.Split.TRAIN.subsplit([6, 4])
split_trn_val_tst = (split_trn_val, tfds.Split.TEST)
(data_trn, data_val), data_tst = tfds.load(
    name="imdb_reviews", 
    split=split_trn_val_tst,
    as_supervised=True)

# Explore data
text_trn, label_trn = next(iter(data_trn.batch(10)))
# text_trn[0]
# label_trn

# Download an embedding layer
embedding_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
embedding = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)
# Convert example text into features; text_trn has shape (10,); x_trn has shape (10,20)
x_trn = embedding(text_trn)

# Build model
model = tf.keras.Sequential()
model.add(embedding)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# Summary of model
model.summary()

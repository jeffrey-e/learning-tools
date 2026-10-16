"""Forward pass of the tiny CNN with TensorFlow ops and a Keras Dense layer.

Generated from the Tiny CNN Walkthrough page (index.html) with its default settings:
the vertical-line image and the hand-set weights. The page's copy buttons give a
version with your own drawing, weights or training settings.
"""

import numpy as np
import tensorflow as tf

np.set_printoptions(precision=4, suppress=True)   # hides float32 noise such as 0.90000004

img = tf.constant([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]], dtype=tf.float32)
x_in = tf.reshape(img, (1, 6, 6, 1))   # batch, height, width, channels
print(x_in.shape)

K = tf.constant([[1, 0, -1],
                 [1, 0, -1],
                 [1, 0, -1]], dtype=tf.float32)   # vertical kernel
filters = tf.reshape(K, (3, 3, 1, 1))   # height, width, in channels, out channels
print(filters.shape)

conv = tf.nn.conv2d(x_in, filters, strides=1, padding="VALID")
print(conv.shape)
print(conv[0, :, :, 0].numpy())

act = tf.nn.relu(conv)
print(act[0, :, :, 0].numpy())

pool = tf.nn.max_pool2d(act, ksize=2, strides=2, padding="VALID")
print(pool.shape)
print(pool[0, :, :, 0].numpy())

KV = tf.constant([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]], dtype=tf.float32)
KH = tf.constant([[1, 1, 1],
                  [0, 0, 0],
                  [-1, -1, -1]], dtype=tf.float32)
filters = tf.stack([KV, KH], axis=-1)[:, :, tf.newaxis, :]   # (3, 3, 1, 2)

conv = tf.nn.conv2d(x_in, filters, strides=1, padding="VALID")
act = tf.nn.relu(conv)
pool = tf.nn.max_pool2d(act, ksize=2, strides=2, padding="VALID")
print(pool.shape)   # 2 feature maps of 2×2

# channels first, so the order matches the page: x0-x3 vertical, x4-x7 horizontal
x = tf.keras.layers.Flatten()(tf.transpose(pool, [0, 3, 1, 2]))
print(x.shape)
print(x.numpy())

W = tf.constant([[1, 1, 1, 1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, 1, 1, 1, 1],
                 [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]], dtype=tf.float32)   # (4, 8): one row per class
b = tf.constant([0, 0, -3, 2], dtype=tf.float32)

dense = tf.keras.layers.Dense(4)
dense.build((None, 8))
dense.set_weights([tf.transpose(W).numpy(), b.numpy()])   # Keras stores the kernel as (8, 4)
logits = dense(x)
print(logits.numpy())

p = tf.nn.softmax(logits)[0]
print(p.numpy().round(3))

CLASSES = ["Vertical line", "Horizontal line", "Cross / box", "Empty"]
print(CLASSES[int(tf.argmax(p))])

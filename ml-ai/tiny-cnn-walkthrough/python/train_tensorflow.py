"""Train the tiny CNN with Keras model.fit and the Adam optimizer.

Generated from the Tiny CNN Walkthrough page (index.html) with its default settings:
the vertical-line image and the hand-set weights. The page's copy buttons give a
version with your own drawing, weights or training settings.
"""

import numpy as np
import tensorflow as tf

def make_dataset():
    X, y = [], []
    for w in (1, 2):                                   # vertical and horizontal lines
        for c in range(7 - w):
            v = np.zeros((6, 6)); v[:, c:c + w] = 1
            X += [v, v.T.copy()]; y += [0, 1]
    crosses = [(c, r, 2) for c in (1, 2, 3) for r in (1, 2, 3)] + [(1, 1, 1), (3, 2, 1), (4, 4, 1)]
    for c, r, w in crosses:
        m = np.zeros((6, 6)); m[:, c:c + w] = 1; m[r:r + w, :] = 1
        X.append(m); y.append(2)
    for s in (4, 5, 6):                                # square outlines
        for r in range(7 - s):
            for c in range(7 - s):
                m = np.zeros((6, 6)); m[r:r + s, c:c + s] = 1; m[r + 1:r + s - 1, c + 1:c + s - 1] = 0
                X.append(m); y.append(2)
    X += [np.zeros((6, 6))] * 6; y += [3] * 6          # empty images
    return np.array(X), np.array(y)   # 54 images, labels 0-3

X, y = make_dataset()
model = tf.keras.Sequential([
    tf.keras.Input(shape=(6, 6, 1)),
    tf.keras.layers.Conv2D(2, 3, use_bias=False, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Permute((3, 1, 2)),    # channels first, so the flatten order matches the page
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X[..., None], y, batch_size=8, epochs=100, verbose=0)
print("loss", round(history.history["loss"][-1], 3), "accuracy", round(history.history["accuracy"][-1], 2))

# the learned weights, in the page's layout
kernels = model.layers[0].get_weights()[0]       # (3, 3, 1, 2)
W_keras, b = model.layers[-1].get_weights()      # (8, 4) and (4,)
print(kernels[:, :, 0, 0].round(2))              # first kernel
print(kernels[:, :, 0, 1].round(2))              # second kernel
print(W_keras.T.round(2))                        # (4, 8): one row per class, like the table
print(b.round(2))

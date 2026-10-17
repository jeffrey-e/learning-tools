# Image recognition walkthrough: husky or malamute? Run from the project folder.
# It needs the photos in images/, data/photos.json and the weights in python/.

import json

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image

photo = "images/c0159.jpg"   # labelled malamute
x = np.asarray(Image.open(photo).convert("RGB"), dtype="float32")[None]   # a batch of one photo
print(x.shape)            # (photos, height, width, red/green/blue)
print("centre pixel (red, green, blue):", x[0, 112, 112].astype(int).tolist())

def build_scratch():
    """Model A: a small CNN, trained from scratch on our photos only."""
    model = keras.Sequential([keras.Input((224, 224, 3)), layers.Rescaling(1 / 255, name="rescale")])
    for i, filters in enumerate([16, 32, 64, 128]):
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i + 1}"))
        model.add(layers.BatchNormalization(name=f"bn{i + 1}"))
        model.add(layers.ReLU(name=f"relu{i + 1}"))
        if i < 3:
            model.add(layers.MaxPooling2D(name=f"pool{i + 1}"))
    model.add(layers.GlobalAveragePooling2D(name="gap"))   # one number per filter: its average over the photo
    model.add(layers.Dropout(0.3, name="dropout"))
    model.add(layers.Dense(2, activation="softmax", name="classify"))
    return model

scratch = build_scratch()
scratch.load_weights("python/scratch.weights.h5")   # the trained weights
print(scratch.count_params(), "parameters, all learned from our photos")
p = scratch.predict(x, verbose=0)[0]
print(f"husky {p[0]:.3f}, malamute {p[1]:.3f}")

# Model B: MobileNetV2, trained on 1.2 million ImageNet photos, without its last layer (frozen: never changed)
base = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling="avg", weights="imagenet")
prep = keras.applications.mobilenet_v2.preprocess_input   # pixels 0..255 -> -1..1
head = keras.Sequential([keras.Input((1280,)), layers.Dropout(0.2, name="dropout"), layers.Dense(2, activation="softmax", name="classify")])
head.load_weights("python/transfer_head.weights.h5")   # the only part trained on our photos

features = base.predict(prep(x.copy()), verbose=0)    # 1280 numbers that describe the photo
p = head.predict(features, verbose=0)[0]
print(features.shape, "features;", base.count_params(), "pretrained parameters,", head.count_params(), "trained")
print(f"husky {p[0]:.3f}, malamute {p[1]:.3f}")

# Both models, block by block: the layers grouped the way the page draws them
import re

def block_name(n):
    m = re.match(r"block_(\d+)_", n)
    if m:
        return "block " + m.group(1)        # MobileNetV2's 17 inverted residual blocks
    if n.startswith("expanded_conv"):
        return "block 0"
    if n in ("Conv1", "bn_Conv1", "Conv1_relu"):
        return "stem"
    if n in ("Conv_1", "Conv_1_bn", "out_relu"):
        return "last conv"
    m = re.match(r"(conv|bn|relu|pool)(\d+)$", n)
    if m:
        return "conv " + m.group(2)         # model A: convolution, (batch norm,) ReLU, pooling
    if n == "gap" or n.startswith("global_average_pooling"):
        return "average pooling"
    return n

def blocks(layers, part):
    rows = []
    for layer in layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        name, shape = block_name(layer.name), tuple(layer.output.shape)[1:]
        if rows and rows[-1][0] == name:     # same block: its output is its last layer's
            rows[-1][1], rows[-1][2] = shape, rows[-1][2] + layer.count_params()
        else:
            rows.append([name, shape, layer.count_params(), part])
    return rows

for title, rows in (("Model A", blocks(scratch.layers, "trained")),
                    ("Model B", blocks(base.layers, "frozen") + blocks(head.layers, "trained"))):
    print(title)
    for name, shape, params, part in rows:
        print(f"{name:>16} {str(shape):>16} {params:>10,}  {part}")

# The original model with its own last layer: 1000 ImageNet classes, including both breeds
full = keras.applications.MobileNetV2(weights="imagenet")
top = keras.applications.mobilenet_v2.decode_predictions(full.predict(prep(x.copy()), verbose=0), top=3)[0]
for _, name, prob in top:
    print(f"{name.replace('_', ' ')} {prob:.3f}")

# Class activation map: the last feature maps (7 x 7 positions x 1280), weighted by the head's weights for the predicted class
fmap = keras.Model(base.input, base.get_layer("out_relu").output)
A = fmap.predict(prep(x.copy()), verbose=0)[0]          # (7, 7, 1280)
W = head.get_layer("classify").get_weights()[0]         # (1280, 2)
c = int(p.argmax())
cam = A @ W[:, c]                                       # (7, 7): high = this part of the photo pushed towards class c
print("predicted:", ["husky", "malamute"][c])
for row in cam:
    print(" ".join(f"{v:6.2f}" for v in row))

# Both models on all test photos (never used for training or for choosing settings)
photos = json.load(open("data/photos.json"))
test = [p for p in photos if p["split"] == "test"]
Xt = np.stack([np.asarray(Image.open(p["file"]).convert("RGB"), dtype="float32") for p in test])
yt = np.array([p["label"] == "malamute" for p in test]).astype(int)
pa = scratch.predict(Xt, verbose=0, batch_size=32).argmax(1)
pb = head.predict(base.predict(prep(Xt.copy()), verbose=0, batch_size=32), verbose=0).argmax(1)
for name, pred in (("A from scratch", pa), ("B transfer learning", pb)):
    cm = [[int(((yt == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    print(f"{name}: accuracy {(pred == yt).mean():.3f}; rows = true husky, malamute: {cm}")

// The Keras code beside each step of the image recognition walkthrough, with the exact output it prints,
// shared by index.html (browser) and the tests (Node). Pure functions, no DOM access.
// The outputs come from data.js, which tools/train.py wrote with the same Keras models;
// tests/check_outputs.py runs every cell in real TensorFlow and compares.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ImgPage = api;
})(globalThis, function () {
  'use strict';

  const pyFixed = (v, d) => (Object.is(v, -0) ? '-' : '') + v.toFixed(d);
  const pyNum = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  const pyTuple = xs => '(' + xs.join(', ') + (xs.length === 1 ? ',' : '') + ')';

  // Model A's layers for a configuration (same as tools/experiments.py)
  function scratchCode(cfg) {
    const n = cfg.widths.length;
    return [
      'def build_scratch():',
      '    """Model A: a small CNN, trained from scratch on our photos only."""',
      '    model = keras.Sequential([keras.Input((224, 224, 3)), layers.Rescaling(1 / 255, name="rescale")])',
      ...(cfg.size !== 224 ? [`    model.add(layers.Resizing(${cfg.size}, ${cfg.size}, name="resize"))   # fewer pixels: chosen in the experiments`] : []),
      `    for i, filters in enumerate([${cfg.widths.join(', ')}]):`,
      `        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=${cfg.bn ? 'False' : 'True'}, name=f"conv{i + 1}"))`,
      ...(cfg.bn ? ['        model.add(layers.BatchNormalization(name=f"bn{i + 1}"))'] : []),
      '        model.add(layers.ReLU(name=f"relu{i + 1}"))',
      `        if i < ${n - 1}:`,
      '            model.add(layers.MaxPooling2D(name=f"pool{i + 1}"))',
      '    model.add(layers.GlobalAveragePooling2D(name="gap"))   # one number per filter: its average over the photo',
      '    model.add(layers.Dropout(0.3, name="dropout"))',
      '    model.add(layers.Dense(2, activation="softmax", name="classify"))',
      '    return model',
    ];
  }

  // One cell per section. photo: the selected test photo (a row of data.photos).
  function pythonCells(photo, data) {
    const ca = data.config.scratch, cb = data.config.transfer;
    const probs = p => `husky ${pyFixed(1 - p, 3)}, malamute ${pyFixed(p, 3)}`;
    const m = data.metrics;
    return [
      {
        id: 'load',
        code: [
          'import json',
          '',
          'import numpy as np',
          'import tensorflow as tf',
          'import keras',
          'from keras import layers',
          'from PIL import Image',
          '',
          `photo = "${photo.file}"   # labelled ${photo.label}`,
          'x = np.asarray(Image.open(photo).convert("RGB"), dtype="float32")[None]   # a batch of one photo',
          'print(x.shape)            # (photos, height, width, red/green/blue)',
          'print("centre pixel (red, green, blue):", x[0, 112, 112].astype(int).tolist())',
        ].join('\n'),
        out: `(1, 224, 224, 3)\ncentre pixel (red, green, blue): [${photo.px.join(', ')}]`,
      },
      {
        id: 'scratch',
        code: [
          ...scratchCode(ca),
          '',
          'scratch = build_scratch()',
          'scratch.load_weights("python/scratch.weights.h5")   # the trained weights',
          'print(scratch.count_params(), "parameters, all learned from our photos")',
          'p = scratch.predict(x, verbose=0)[0]',
          'print(f"husky {p[0]:.3f}, malamute {p[1]:.3f}")',
        ].join('\n'),
        out: `${data.params.scratch} parameters, all learned from our photos\n${probs(photo.pa)}`,
      },
      {
        id: 'train_a',
        train: true,
        code: [
          '# Training model A (the chosen experiment). Random flips, rotations and zooms make every epoch\'s photos a bit different.',
          'photos = json.load(open("data/photos.json"))',
          'X = np.stack([np.asarray(Image.open(p["file"]).convert("RGB"), dtype="float32") for p in photos])',
          'y = np.array([p["label"] == "malamute" for p in photos]).astype(int)',
          'split = np.array([p["split"] for p in photos])',
          'train, val = split == "train", split == "val"',
          'weights = {k: train.sum() / (2 * (y[train] == k).sum()) for k in (0, 1)}   # malamutes are rarer: weigh them up',
          '',
          'keras.utils.set_random_seed(0)',
          'augment = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.15)])',
          'ds = tf.data.Dataset.from_tensor_slices((X[train], y[train])).shuffle(512, seed=0).batch(16)',
          ...(ca.augment ? ['ds = ds.map(lambda a, b: (augment(a, training=True), b))'] : []),
          'model = build_scratch()',
          `model.compile(optimizer=keras.optimizers.Adam(${pyNum(ca.lr)}), loss="sparse_categorical_crossentropy", metrics=["accuracy"])`,
          `history = model.fit(ds, validation_data=(X[val], y[val]), epochs=${ca.epochs}${ca.class_weight ? ', class_weight=weights' : ''}, verbose=0)`,
          'print(f"validation accuracy {history.history[\'val_accuracy\'][-1]:.3f}")',
        ].join('\n'),
        out: `validation accuracy ${data.history.scratch.val_accuracy.slice(-1)[0].toFixed(3)}`,
      },
      {
        id: 'transfer',
        code: [
          '# Model B: MobileNetV2, trained on 1.2 million ImageNet photos, without its last layer (frozen: never changed)',
          'base = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling="avg", weights="imagenet")',
          'prep = keras.applications.mobilenet_v2.preprocess_input   # pixels 0..255 -> -1..1',
          `head = keras.Sequential([keras.Input((1280,)), layers.Dropout(${pyNum(cb.dropout)}, name="dropout"), layers.Dense(2, activation="softmax", name="classify")])`,
          'head.load_weights("python/transfer_head.weights.h5")   # the only part trained on our photos',
          '',
          'features = base.predict(prep(x.copy()), verbose=0)    # 1280 numbers that describe the photo',
          'p = head.predict(features, verbose=0)[0]',
          'print(features.shape, "features;", base.count_params(), "pretrained parameters,", head.count_params(), "trained")',
          'print(f"husky {p[0]:.3f}, malamute {p[1]:.3f}")',
        ].join('\n'),
        out: `(1, 1280) features; ${data.params.base} pretrained parameters, ${data.params.head} trained\n${probs(photo.pb)}`,
      },
      {
        id: 'arch',
        code: [
          '# Both models, block by block: the layers grouped the way the page draws them',
          'import re',
          '',
          'def block_name(n):',
          '    m = re.match(r"block_(\\d+)_", n)',
          '    if m:',
          '        return "block " + m.group(1)        # MobileNetV2\'s 17 inverted residual blocks',
          '    if n.startswith("expanded_conv"):',
          '        return "block 0"',
          '    if n in ("Conv1", "bn_Conv1", "Conv1_relu"):',
          '        return "stem"',
          '    if n in ("Conv_1", "Conv_1_bn", "out_relu"):',
          '        return "last conv"',
          '    m = re.match(r"(conv|bn|relu|pool)(\\d+)$", n)',
          '    if m:',
          '        return "conv " + m.group(2)         # model A: convolution, (batch norm,) ReLU, pooling',
          '    if n == "gap" or n.startswith("global_average_pooling"):',
          '        return "average pooling"',
          '    return n',
          '',
          'def blocks(layers, part):',
          '    rows = []',
          '    for layer in layers:',
          '        if isinstance(layer, keras.layers.InputLayer):',
          '            continue',
          '        name, shape = block_name(layer.name), tuple(layer.output.shape)[1:]',
          '        if rows and rows[-1][0] == name:     # same block: its output is its last layer\'s',
          '            rows[-1][1], rows[-1][2] = shape, rows[-1][2] + layer.count_params()',
          '        else:',
          '            rows.append([name, shape, layer.count_params(), part])',
          '    return rows',
          '',
          'for title, rows in (("Model A", blocks(scratch.layers, "trained")),',
          '                    ("Model B", blocks(base.layers, "frozen") + blocks(head.layers, "trained"))):',
          '    print(title)',
          '    for name, shape, params, part in rows:',
          '        print(f"{name:>16} {str(shape):>16} {params:>10,}  {part}")',
        ].join('\n'),
        out: [['Model A', data.arch.scratch], ['Model B', data.arch.transfer]].flatMap(([title, rows]) => [title, ...rows.map(([name, shape, params, part]) =>
          `${name.padStart(16)} ${pyTuple(shape).padStart(16)} ${params.toLocaleString('en-US').padStart(10)}  ${part}`)]).join('\n'),
      },
      {
        id: 'imagenet',
        code: [
          '# The original model with its own last layer: 1000 ImageNet classes, including both breeds',
          'full = keras.applications.MobileNetV2(weights="imagenet")',
          'top = keras.applications.mobilenet_v2.decode_predictions(full.predict(prep(x.copy()), verbose=0), top=3)[0]',
          'for _, name, prob in top:',
          '    print(f"{name.replace(\'_\', \' \')} {prob:.3f}")',
        ].join('\n'),
        out: photo.imagenet.map(([name, p]) => `${name} ${p.toFixed(3)}`).join('\n'),
      },
      {
        id: 'train_b',
        train: true,
        code: [
          '# Training model B\'s head: the base is frozen, so the features are computed once and only 2 562 numbers are learned',
          'F = base.predict(prep(X.copy()), verbose=0, batch_size=32)   # (photos, 1280)',
          'keras.utils.set_random_seed(0)',
          'new_head = keras.Sequential([keras.Input((1280,)), layers.Dropout(' + pyNum(cb.dropout) + '), layers.Dense(2, activation="softmax")])',
          `new_head.compile(optimizer=keras.optimizers.Adam(${pyNum(cb.lr)}), loss="sparse_categorical_crossentropy", metrics=["accuracy"])`,
          `history = new_head.fit(F[train], y[train], validation_data=(F[val], y[val]), epochs=${cb.epochs}, batch_size=16${cb.class_weight ? ', class_weight=weights' : ''}, verbose=0)`,
          'print(f"validation accuracy {history.history[\'val_accuracy\'][-1]:.3f}")',
        ].join('\n'),
        out: `validation accuracy ${data.history.transfer.val_accuracy.slice(-1)[0].toFixed(3)}`,
      },
      {
        id: 'cam',
        code: [
          '# Class activation map: the last feature maps (7 x 7 positions x 1280), weighted by the head\'s weights for the predicted class',
          'fmap = keras.Model(base.input, base.get_layer("out_relu").output)',
          'A = fmap.predict(prep(x.copy()), verbose=0)[0]          # (7, 7, 1280)',
          'W = head.get_layer("classify").get_weights()[0]         # (1280, 2)',
          'c = int(p.argmax())',
          'cam = A @ W[:, c]                                       # (7, 7): high = this part of the photo pushed towards class c',
          'print("predicted:", ["husky", "malamute"][c])',
          'for row in cam:',
          '    print(" ".join(f"{v:6.2f}" for v in row))',
        ].join('\n'),
        out: [`predicted: ${photo.pb > 0.5 ? 'malamute' : 'husky'}`, ...photo.cam.map(r => r.map(v => pyFixed(v, 2).padStart(6)).join(' '))].join('\n'),
      },
      {
        id: 'evaluate',
        code: [
          '# Both models on all test photos (never used for training or for choosing settings)',
          'photos = json.load(open("data/photos.json"))',
          'test = [p for p in photos if p["split"] == "test"]',
          'Xt = np.stack([np.asarray(Image.open(p["file"]).convert("RGB"), dtype="float32") for p in test])',
          'yt = np.array([p["label"] == "malamute" for p in test]).astype(int)',
          'pa = scratch.predict(Xt, verbose=0, batch_size=32).argmax(1)',
          'pb = head.predict(base.predict(prep(Xt.copy()), verbose=0, batch_size=32), verbose=0).argmax(1)',
          'for name, pred in (("A from scratch", pa), ("B transfer learning", pb)):',
          '    cm = [[int(((yt == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)]',
          '    print(f"{name}: accuracy {(pred == yt).mean():.3f}; rows = true husky, malamute: {cm}")',
        ].join('\n'),
        out: [['A from scratch', m.scratch.test], ['B transfer learning', m.transfer.test]]
          .map(([name, t]) => `${name}: accuracy ${t.accuracy.toFixed(3)}; rows = true husky, malamute: [[${t.confusion[0].join(', ')}], [${t.confusion[1].join(', ')}]]`).join('\n'),
      },
    ];
  }

  // The copy button's script: every cell except the training cells.
  const fullScript = cells =>
    '# Image recognition walkthrough: husky or malamute? Run from the project folder.\n' +
    '# It needs the photos in images/, data/photos.json and the weights in python/.\n\n' +
    cells.filter(c => !c.train).map(c => c.code).join('\n\n') + '\n';

  // Class activation map from the last feature maps [h][w][k] and the Dense weights W[k][class]
  function cam(maps, W, c) {
    return maps.map(row => row.map(cell => cell.reduce((s, v, k) => s + v * W[k][c], 0)));
  }

  return { pythonCells, fullScript, cam, pyFixed };
});

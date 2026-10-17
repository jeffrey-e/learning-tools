// The maths behind the log autoencoder walkthrough, shared by index.html (browser) and the tests (Node).
// Pure functions, no DOM access. pythonCells() writes the same computation as TensorFlow/Keras code,
// together with the output that code prints; tests/check_outputs.py runs it and checks every line.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LogAE = api;
})(globalThis, function () {
  'use strict';

  // Labels by rule, first match wins (same rules and order as tools/train_model.py).
  // The model never sees these labels: they only pick the training lines and score the results.
  const RULES = [
    ['attack', String.raw`\[preauth\]|Failed |[Ii]nvalid user|authentication failure|check pass|maximum authentication attempts|ignoring max retries|more authentication failures|Bad protocol version|Did not receive identification|message repeated|Connection reset`],
    ['legit', String.raw`^Accepted |session opened|session closed|disconnected by user|^Disconnected from (user \S+ )?[0-9.]+ port \d+$|^Starting session`],
  ];
  const RULE_RE = RULES.map(([name, rule]) => [name, new RegExp(rule)]);
  function label(msg) {
    for (const [name, re] of RULE_RE) if (re.test(msg)) return name;
    return 'other';
  }

  // Keras weights in get_weights() order, by name ("enc_conv1.kernel" and so on).
  function unpack(w) {
    const p = {};
    w.names.forEach((name, i) => { p[name] = w.weights[i]; });
    return p;
  }

  // Characters to ids. Code points, as in Python (an emoji is one character). 0 = padding, 1 = unknown.
  function encode(msg, w) {
    const chars = Array.from(msg).slice(0, w.L);
    const index = new Map(w.vocab.map((c, i) => [c, i]));
    const ids = chars.map(c => (index.has(c) ? index.get(c) : 1));
    return { chars, n: chars.length, truncated: Array.from(msg).length > w.L, ids: ids.concat(new Array(w.L - ids.length).fill(0)) };
  }

  // ---- Layers on plain arrays: X is [positions][channels] ----
  // Conv1D with padding="same": position t sees t-2 … t+2 (kernel 5), zeros beyond the edges.
  function conv1d(X, W, b, relu) {
    const K = W.length, Cin = W[0].length, Cout = b.length, pad = (K - 1) / 2;
    return X.map((_, t) => {
      const out = b.slice();
      for (let k = 0; k < K; k++) {
        const row = X[t + k - pad];
        if (!row) continue;
        for (let c = 0; c < Cin; c++) {
          const v = row[c], Wkc = W[k][c];
          if (v !== 0) for (let o = 0; o < Cout; o++) out[o] += v * Wkc[o];
        }
      }
      return relu ? out.map(v => Math.max(0, v)) : out;
    });
  }
  const maxpool = X => Array.from({ length: X.length / 2 }, (_, t) => X[2 * t].map((v, c) => Math.max(v, X[2 * t + 1][c])));
  const upsample = X => X.flatMap(row => [row.slice(), row.slice()]);
  function softmax(row) {
    const max = Math.max(...row);
    const e = row.map(v => Math.exp(v - max));
    const sum = e.reduce((s, v) => s + v, 0);
    return e.map(v => v / sum);
  }
  const argmax = row => row.reduce((best, v, i) => (v > row[best] ? i : best), 0);

  // Same symbols as the generated Python: "·" where the model predicts padding, "?" for unknown.
  const shown = (vocab, i) => (i === 0 ? '·' : i === 1 ? '?' : vocab[i]);

  // One forward pass through the autoencoder, keeping every intermediate array for the page.
  function run(msg, w) {
    const p = unpack(w);
    const { chars, n, truncated, ids } = encode(msg, w);
    const emb = ids.map(i => p['embedding.embeddings'][i].slice());
    const enc1 = conv1d(emb, p['enc_conv1.kernel'], p['enc_conv1.bias'], true);
    const pool1 = maxpool(enc1);
    const enc2 = conv1d(pool1, p['enc_conv2.kernel'], p['enc_conv2.bias'], true);
    const z = maxpool(enc2);
    const up1 = upsample(z);
    const dec1 = conv1d(up1, p['dec_conv1.kernel'], p['dec_conv1.bias'], true);
    const up2 = upsample(dec1);
    const dec2 = conv1d(up2, p['dec_conv2.kernel'], p['dec_conv2.bias'], true);
    const probs = conv1d(dec2, p['output.kernel'], p['output.bias'], false).map(softmax);
    const nll = ids.slice(0, n).map((id, t) => -Math.log(Math.max(probs[t][id], 1e-7)));
    const recon = probs.slice(0, n).map(row => shown(w.vocab, argmax(row)));
    const score = n ? nll.reduce((s, v) => s + v, 0) / n : NaN;
    return { chars, n, truncated, ids, emb, enc1, pool1, enc2, z, up1, dec1, up2, dec2, probs, nll, recon, score };
  }

  // The k most likely characters at one position.
  const topk = (probs, vocab, k) => probs.map((p, i) => ({ i, c: vocab[i], p })).sort((a, b) => b.p - a.p).slice(0, k);

  // Counts at a threshold, from the precomputed scores in data.js.
  function detect(lines, threshold) {
    const g = { val: [0, 0], testLegit: [0, 0], attack: [0, 0], other: [0, 0] };
    for (const [, lab, split, score] of lines) {
      const key = lab === 'attack' ? 'attack' : lab === 'other' ? 'other' : split === 'val' ? 'val' : split === 'test' ? 'testLegit' : null;
      if (!key) continue;
      g[key][1]++;
      if (score > threshold) g[key][0]++;
    }
    const caught = g.attack[0], falseAlarms = g.testLegit[0];
    return { ...g, caught, missed: g.attack[1] - caught, falseAlarms, precision: caught / Math.max(caught + falseAlarms, 1), recall: caught / Math.max(g.attack[1], 1) };
  }

  // ---- Python output formatting (must match what the generated code prints) ----
  const pyFixed = (v, d) => (Object.is(v, -0) ? '-' : '') + v.toFixed(d);
  const pyNum = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  const pyPct = (v, width) => ((v * 100).toFixed(1) + '%').padStart(width);
  const pyShape = dims => '(' + dims.join(', ') + (dims.length === 1 ? ',' : '') + ')';
  // Python's repr() of a short str without control characters.
  const pyRepr = s => (s.includes("'") && !s.includes('"') ? `"${s.replace(/\\/g, '\\\\')}"` : `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`);
  const row3 = row => row.map(v => pyFixed(v, 3).padStart(7)).join(' ');

  // The page's TensorFlow/Keras code, one cell per section, each with the exact output it prints.
  // line: the selected or typed message; pos: the position inspected in the decoder step.
  function pythonCells(line, w, data, { threshold = w.threshold, pos = 0 } = {}) {
    const msgs = data.lines.map(r => r[0]);
    const labels = msgs.map(label);
    const count = name => labels.filter(l => l === name).length;
    const legitN = count('legit');
    const nTrain = Math.trunc(0.6 * legitN), nVal = Math.trunc(0.8 * legitN) - nTrain, nTest = legitN - nTrain - nVal;
    const r = run(line, w);
    const n = r.n;
    pos = Math.min(Math.max(pos, 0), Math.max(n - 1, 0));
    const cells = [
      {
        id: 'data',
        code: [
          'import json',
          'import re',
          '',
          'import numpy as np',
          'import keras',
          'from keras import layers',
          '',
          'lines = open("data/auth.log", encoding="utf-8").read().splitlines()',
          '# Keep the sshd lines, and only their message: the date, host and process id change on every line',
          String.raw`SSHD = re.compile(r"^\w{3} +\d+ [\d:]+ \S+ sshd\[\d+\]: (.*)$")`,
          'messages = [m.group(1) for line in lines if (m := SSHD.match(line))]',
          'print(len(lines), "lines in auth.log,", len(messages), "from sshd")',
          'print(messages[1])',
        ].join('\n'),
        out: `${data.total} lines in auth.log, ${msgs.length} from sshd\n${msgs[1]}`,
      },
      {
        id: 'labels',
        code: [
          '# Labels by rule, first match wins. The model never sees them: they pick the training lines',
          '# and tell us afterwards which detections were right.',
          'RULES = [',
          ...RULES.map(([name, rule]) => `    ("${name}", r"${rule}"),`),
          ']',
          '',
          'def label(msg):',
          '    for name, rule in RULES:',
          '        if re.search(rule, msg):',
          '            return name',
          '    return "other"',
          '',
          'labels = [label(m) for m in messages]',
          'for name in ("legit", "attack", "other"):',
          '    print(f"{name:>6} {labels.count(name):5d} {labels.count(name) / len(labels):6.1%}")',
          '',
          '# Train on legitimate lines only: 60% train, 20% validation (sets the threshold), 20% test.',
          '# Every attack line is test data: the model never sees one during training.',
          'legit = [i for i, lab in enumerate(labels) if lab == "legit"]',
          'np.random.default_rng(1).shuffle(legit)',
          'n = len(legit)',
          'train_ids, val_ids, test_ids = legit[:int(0.6 * n)], legit[int(0.6 * n):int(0.8 * n)], legit[int(0.8 * n):]',
          'attack_ids = [i for i, lab in enumerate(labels) if lab == "attack"]',
          'other_ids = [i for i, lab in enumerate(labels) if lab == "other"]',
          'print(f"{len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} + {len(attack_ids)} test")',
        ].join('\n'),
        out: [
          ...['legit', 'attack', 'other'].map(name => `${name.padStart(6)} ${String(count(name)).padStart(5)} ${pyPct(count(name) / labels.length, 6)}`),
          `${nTrain} train, ${nVal} validation, ${nTest} + ${count('attack')} test`,
        ].join('\n'),
      },
      {
        id: 'chars',
        code: [
          'weights = json.load(open("python/weights.json"))',
          'vocab = weights["vocab"]   # every character in the training lines; 0 = padding, 1 = unknown',
          'index = {c: i for i, c in enumerate(vocab)}',
          `L = ${w.L}                    # every line becomes exactly L ids: cut or padded`,
          '',
          'def encode(msg):',
          '    ids = [index.get(c, 1) for c in msg[:L]]',
          '    return ids + [0] * (L - len(ids))',
          '',
          `line = ${JSON.stringify(line)}`,
          'x = np.array([encode(line)])',
          'n = min(len(line), L)',
          'print(len(vocab), "tokens:", "".join(vocab[2:]))',
          'print(x[0, :n].tolist())',
          'print(L - n, "padding positions")',
        ].join('\n'),
        out: [
          `${w.vocab.length} tokens: ${w.vocab.slice(2).join('')}`,
          '[' + r.ids.slice(0, n).join(', ') + ']',
          `${w.L - n} padding positions`,
        ].join('\n'),
      },
      {
        id: 'model',
        code: [
          'def build_model(V):',
          '    return keras.Sequential([',
          `        keras.Input((L,), dtype="int32"),`,
          '        layers.Embedding(V, 8, name="embedding"),',
          '        # Encoder: squeeze 128 positions into 32',
          '        layers.Conv1D(16, 5, padding="same", activation="relu", name="enc_conv1"),',
          '        layers.MaxPooling1D(2, name="enc_pool1"),',
          '        layers.Conv1D(4, 5, padding="same", activation="relu", name="enc_conv2"),',
          '        layers.MaxPooling1D(2, name="bottleneck"),',
          '        # Decoder: expand back to 128 positions',
          '        layers.UpSampling1D(2, name="dec_up1"),',
          '        layers.Conv1D(4, 5, padding="same", activation="relu", name="dec_conv1"),',
          '        layers.UpSampling1D(2, name="dec_up2"),',
          '        layers.Conv1D(16, 5, padding="same", activation="relu", name="dec_conv2"),',
          '        layers.Conv1D(V, 1, activation="softmax", name="output"),   # a probability per character',
          '    ])',
          '',
          'model = build_model(len(vocab))',
          'model.set_weights([np.array(v, dtype="float32") for v in weights["weights"]])   # the trained weights',
          'print(model.count_params(), "parameters")',
          '',
          '# Run the line through the layers one by one and keep every output',
          'outs, h = {}, x',
          'for layer in model.layers:',
          '    h = layer(h)',
          '    outs[layer.name] = np.array(h)[0]',
          '    print(f"{layer.name:>11} {str(tuple(h.shape)):>14}")',
        ].join('\n'),
        out: [
          `${w.params} parameters`,
          ...[['embedding', r.emb], ['enc_conv1', r.enc1], ['enc_pool1', r.pool1], ['enc_conv2', r.enc2], ['bottleneck', r.z],
            ['dec_up1', r.up1], ['dec_conv1', r.dec1], ['dec_up2', r.up2], ['dec_conv2', r.dec2], ['output', r.probs]]
            .map(([name, a]) => `${name.padStart(11)} ${pyShape([1, a.length, a[0].length]).padStart(14)}`),
        ].join('\n'),
      },
    ];
    if (n) {
      const show = Math.min(n, 5);
      cells.push(
        {
          id: 'embedding',
          code: [
            '# Each id picks one learned row of 8 numbers',
            'emb = outs["embedding"]   # (128, 8)',
            `for t in range(${show}):`,
            '    print(f"{t:>3} {line[t]!r:>5} " + " ".join(f"{v:7.4f}" for v in emb[t]))',
          ].join('\n'),
          out: r.emb.slice(0, show).map((row, t) => `${String(t).padStart(3)} ${pyRepr(r.chars[t]).padStart(5)} ` + row.map(v => pyFixed(v, 4).padStart(7)).join(' ')).join('\n'),
        },
        {
          id: 'encoder',
          code: [
            '# enc_conv1: 16 filters, each looks at 5 neighbouring characters (positions t-2 … t+2)',
            `print("enc_conv1 at position ${pos}:")`,
            `print(" ".join(f"{v:.3f}" for v in outs["enc_conv1"][${pos}]))`,
            '# The bottleneck: 32 positions × 4 numbers. Everything the decoder gets to see',
            'z = outs["bottleneck"]',
            'for t in range(4):',
            '    print(f"z[{t}] " + " ".join(f"{v:7.3f}" for v in z[t]))',
          ].join('\n'),
          out: [
            `enc_conv1 at position ${pos}:`,
            r.enc1[pos].map(v => pyFixed(v, 3)).join(' '),
            ...r.z.slice(0, 4).map((row, t) => `z[${t}] ` + row3(row)),
          ].join('\n'),
        },
        {
          id: 'decoder',
          code: [
            'probs = outs["output"]   # (128, V): a probability for every character, at every position',
            `pos = ${pos}`,
            'print("true character:", repr(line[pos]))',
            'for i in np.argsort(-probs[pos])[:3]:',
            '    print(f"{vocab[i]!r:>8} {probs[pos, i]:.3f}")',
          ].join('\n'),
          out: [
            `true character: ${pyRepr(r.chars[pos])}`,
            ...topk(r.probs[pos], w.vocab, 3).map(t => `${pyRepr(t.c).padStart(8)} ${pyFixed(t.p, 3)}`),
          ].join('\n'),
        },
        {
          id: 'reconstruct',
          code: [
            '# Error per character: -log(probability the model gave the true character)',
            'p_true = probs[np.arange(n), x[0, :n]]',
            'error = -np.log(np.maximum(p_true, 1e-7))',
            'recon = "".join({0: "·", 1: "?"}.get(i, vocab[i]) for i in probs[:n].argmax(axis=1))',
            'score = error.mean()',
            `threshold = ${pyNum(threshold)}`,
            'print(line[:n])',
            'print(recon)',
            'print(f"score {score:.3f}, threshold {threshold}:", "anomaly" if score > threshold else "normal")',
          ].join('\n'),
          out: [
            r.chars.join(''),
            r.recon.join(''),
            `score ${pyFixed(r.score, 3)}, threshold ${pyNum(threshold)}: ${r.score > threshold ? 'anomaly' : 'normal'}`,
          ].join('\n'),
        },
      );
    }
    const h = w.history;
    cells.push(
      {
        id: 'training',
        train: true,   // shown with the output of the real run; the checker only runs it briefly
        code: [
          '# Training: the input is also the target. Padding positions get weight 0 in the loss.',
          'X = np.array([encode(m) for m in messages])',
          'mask = (X > 0).astype("float32")',
          'keras.utils.set_random_seed(0)',
          'fresh = build_model(len(vocab))',
          'fresh.compile(optimizer="adam", loss="sparse_categorical_crossentropy")',
          'history = fresh.fit(X[train_ids], X[train_ids], sample_weight=mask[train_ids],',
          '                    validation_data=(X[val_ids], X[val_ids], mask[val_ids]),',
          `                    epochs=${w.epochs}, batch_size=32, verbose=0)`,
          'print(f"loss {history.history[\'loss\'][-1]:.4f}, validation loss {history.history[\'val_loss\'][-1]:.4f}")',
        ].join('\n'),
        out: `loss ${h.loss[h.loss.length - 1].toFixed(4)}, validation loss ${h.val_loss[h.val_loss.length - 1].toFixed(4)}`,
      },
      (() => {
        const d = detect(data.lines, threshold);
        const groups = [['validation legit', d.val], ['test legit', d.testLegit], ['attack', d.attack], ['other', d.other]];
        return {
          id: 'detect',
          code: [
            '# Score every line with the trained model, then compare with the threshold',
            'X = np.array([encode(m) for m in messages])',
            'P = model.predict(X, verbose=0, batch_size=256)',
            'p_true = np.take_along_axis(P, X[..., None], axis=2)[..., 0]',
            'real = X > 0',
            'scores = (-np.log(np.maximum(p_true, 1e-7)) * real).sum(axis=1) / np.maximum(real.sum(axis=1), 1)',
            '',
            `threshold = ${pyNum(threshold)}`,
            'flagged = scores > threshold',
            'groups = [("validation legit", val_ids), ("test legit", test_ids), ("attack", attack_ids), ("other", other_ids)]',
            'for name, ids in groups:',
            '    print(f"{name:>16}: {flagged[ids].sum():4d} of {len(ids):4d} flagged")',
            'caught, false_alarms = flagged[attack_ids].sum(), flagged[test_ids].sum()',
            'print(f"precision {caught / max(caught + false_alarms, 1):.3f}, recall {caught / len(attack_ids):.3f}")',
          ].join('\n'),
          out: [
            ...groups.map(([name, [k, of]]) => `${name.padStart(16)}: ${String(k).padStart(4)} of ${String(of).padStart(4)} flagged`),
            `precision ${d.precision.toFixed(3)}, recall ${d.recall.toFixed(3)}`,
          ].join('\n'),
        };
      })(),
    );
    return cells;
  }

  // The copy button's script: every cell except the (slow) training cell.
  const fullScript = cells =>
    '# Log autoencoder walkthrough: anomaly detection in SSH auth.log with a Keras Conv1D autoencoder.\n' +
    '# Run from the project folder (it reads data/auth.log and python/weights.json).\n\n' +
    cells.filter(c => !c.train).map(c => c.code).join('\n\n') + '\n';

  return { RULES, label, unpack, encode, conv1d, maxpool, upsample, softmax, run, topk, detect, pythonCells, fullScript, pyFixed };
});

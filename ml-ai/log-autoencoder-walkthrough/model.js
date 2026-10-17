// The maths behind the log autoencoder walkthrough, shared by index.html (browser) and the tests (Node).
// Pure functions, no DOM access. pythonCells() writes the same computation as TensorFlow/Keras code,
// together with the output that code prints; tests/check_outputs.py runs it and checks every line.
// Every function takes one tokenizer's model (an entry of AE_WEIGHTS): its tokenizer, length L,
// vocabulary, weights and, for BPE, its merges.
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

  // ---- Tokenizers (same rules as the Python in pythonCells and tools/train_model.py) ----
  // The u flag makes JavaScript match whole code points, as Python does (an emoji is one character).
  const placeholder = p => (/^[0-9]/.test(p) ? (p.includes('.') ? '<ip>' : '<num>') : p);
  const MASKED = /[0-9]+(?:\.[0-9]+){3}|[0-9]+|[^\n]/gu;
  const WORDS = /[0-9]+(?:\.[0-9]+){3}|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^ ]/gu;
  const BPE_PIECE = /[A-Za-z_]+|[0-9]+| |[^\n]/gu;
  function mergePair(word, a, b) {
    const out = [];
    for (let i = 0; i < word.length; i++) {
      if (i + 1 < word.length && word[i] === a && word[i + 1] === b) { out.push(a + b); i++; }
      else out.push(word[i]);
    }
    return out;
  }
  const bpeCache = new WeakMap();   // per model: piece -> its subword tokens
  function tokenize(msg, w) {
    switch (w.tokenizer) {
      case 'chars': return Array.from(msg);
      case 'masked': return (msg.match(MASKED) || []).map(placeholder);
      case 'words': return (msg.match(WORDS) || []).map(placeholder);
      case 'bpe': {
        if (!bpeCache.has(w)) bpeCache.set(w, new Map());
        const cache = bpeCache.get(w);
        return (msg.match(BPE_PIECE) || []).flatMap(p => {
          if (!cache.has(p)) cache.set(p, w.merges.reduce((parts, [a, b]) => mergePair(parts, a, b), Array.from(p)));
          return cache.get(p);
        });
      }
      default: throw new Error('unknown tokenizer ' + w.tokenizer);
    }
  }
  // How tokens are joined back into text: words were separated by spaces, everything else keeps its spaces.
  const joiner = w => (w.tokenizer === 'words' ? ' ' : '');

  // Keras weights in get_weights() order, by name ("enc_conv1.kernel" and so on).
  function unpack(w) {
    const p = {};
    w.names.forEach((name, i) => { p[name] = w.weights[i]; });
    return p;
  }

  // Tokens to ids: 0 = padding, 1 = unknown. Every line becomes exactly L ids.
  function encode(msg, w) {
    const all = tokenize(msg, w);
    const tokens = all.slice(0, w.L);
    const index = new Map(w.vocab.map((c, i) => [c, i]));
    const ids = tokens.map(t => (index.has(t) ? index.get(t) : 1));
    return { tokens, n: tokens.length, truncated: all.length > w.L, ids: ids.concat(new Array(w.L - ids.length).fill(0)) };
  }

  // ---- Layers on plain arrays: X is [positions][channels] ----
  // Conv1D with padding="same": position t sees t-2 … t+2 (kernel 5), zeros beyond the edges.
  function conv1d(X, W, b, relu) {
    const K = W.length, Cin = W[0].length, pad = (K - 1) / 2;
    return X.map((_, t) => {
      const out = b.slice();
      for (let k = 0; k < K; k++) {
        const row = X[t + k - pad];
        if (!row) continue;
        for (let c = 0; c < Cin; c++) {
          const v = row[c], Wkc = W[k][c];
          if (v !== 0) for (let o = 0; o < out.length; o++) out[o] += v * Wkc[o];
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
    const { tokens, n, truncated, ids } = encode(msg, w);
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
    return { tokens, n, truncated, ids, emb, enc1, pool1, enc2, z, up1, dec1, up2, dec2, probs, nll, recon, score };
  }

  // The k most likely tokens at one position.
  const topk = (probs, vocab, k) => probs.map((p, i) => ({ i, c: vocab[i], p })).sort((a, b) => b.p - a.p).slice(0, k);

  // Counts at a threshold, from the precomputed scores in data.js (k = the tokenizer's index).
  function detect(lines, threshold, k) {
    const g = { val: [0, 0], testLegit: [0, 0], attack: [0, 0], other: [0, 0] };
    for (const [, lab, split, scores] of lines) {
      const key = lab === 'attack' ? 'attack' : lab === 'other' ? 'other' : split === 'val' ? 'val' : split === 'test' ? 'testLegit' : null;
      if (!key) continue;
      g[key][1]++;
      if (scores[k] > threshold) g[key][0]++;
    }
    const caught = g.attack[0], falseAlarms = g.testLegit[0];
    return { ...g, caught, missed: g.attack[1] - caught, falseAlarms, precision: caught / Math.max(caught + falseAlarms, 1), recall: caught / Math.max(g.attack[1], 1) };
  }

  // ---- Filtering a pasted batch of lines ----
  // A full auth.log line: keep the message of an sshd line, skip other programs. Anything else is taken as a bare message.
  const SSHD = /^\w{3} +\d+ [\d:]+ \S+ sshd\[\d+\]: (.*)$/;
  const OTHER = /^\w{3} +\d+ [\d:]+ \S+ [^\s\[:]+(\[\d+\])?: /;
  function parseLine(raw) {
    const m = raw.match(SSHD);
    if (m) return { msg: m[1] };
    return OTHER.test(raw) ? { skipped: 'not sshd' } : { msg: raw };
  }
  const batchLines = text => text.split(/\r?\n/).filter(l => l.trim() !== '');

  // Scores every message with every model. A line is flagged by a model when its score is above that
  // model's threshold: the selected model uses `threshold` (the slider), the others their default.
  // mode: 'selected' keeps what the selected model flags, 'any' what at least one flags, 'all' what all four flag.
  // A line with no tokens scores 0, as in the Python (an empty mean is guarded there).
  function filterBatch(text, models, { tok, threshold, mode = 'selected' }) {
    const items = [], skipped = [];
    for (const raw of batchLines(text)) {
      const p = parseLine(raw);
      if (p.skipped) { skipped.push(raw); continue; }
      const scores = {}, flags = {};
      for (const [name, w] of Object.entries(models)) {
        const r = run(p.msg, w);
        scores[name] = r.n ? r.score : 0;
        flags[name] = scores[name] > (name === tok ? threshold : w.threshold);
      }
      const votes = Object.values(flags);
      const kept = mode === 'any' ? votes.some(Boolean) : mode === 'all' ? votes.every(Boolean) : flags[tok];
      items.push({ raw, msg: p.msg, scores, flags, kept });
    }
    return { items, skipped };
  }

  // ---- Python output formatting (must match what the generated code prints) ----
  const pyFixed = (v, d) => (Object.is(v, -0) ? '-' : '') + v.toFixed(d);
  // Python's repr() of a float, for the values the page uses (0.0001 and up: no exponent in either language).
  const pyNum = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  const pyPct = (v, width) => ((v * 100).toFixed(1) + '%').padStart(width);
  const pyShape = dims => '(' + dims.join(', ') + (dims.length === 1 ? ',' : '') + ')';
  // Python's repr() of a str without control characters, and of a list of them.
  const pyRepr = s => (s.includes("'") && !s.includes('"') ? `"${s.replace(/\\/g, '\\\\')}"` : `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`);
  const pyList = xs => '[' + xs.map(x => (Array.isArray(x) ? pyList(x) : pyRepr(x))).join(', ') + ']';
  const row3 = row => row.map(v => pyFixed(v, 3).padStart(7)).join(' ');

  const PLACEHOLDER_PY = [
    'def placeholder(p):',
    '    """A number becomes <num>, an IP address <ip>; anything else is kept."""',
    '    return ("<ip>" if "." in p else "<num>") if p[0] in "0123456789" else p',
    '',
  ];
  // The Python tokenize() for each tokenizer: the same rules as tokenize() above.
  const TOKENIZE_PY = {
    chars: [
      'def tokenize(msg):',
      '    return list(msg)   # every character is a token',
    ],
    masked: [
      ...PLACEHOLDER_PY,
      'def tokenize(msg):',
      '    # IP addresses and numbers become one placeholder token each; everything else is one token per character',
      String.raw`    return [placeholder(p) for p in re.findall(r"[0-9]+(?:\.[0-9]+){3}|[0-9]+|[^\n]", msg)]`,
    ],
    words: [
      ...PLACEHOLDER_PY,
      'def tokenize(msg):',
      '    # Words, numbers, IP addresses and single punctuation marks; spaces only separate them',
      String.raw`    return [placeholder(p) for p in re.findall(r"[0-9]+(?:\.[0-9]+){3}|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^ ]", msg)]`,
    ],
    bpe: [
      'merges = weights["merges"]   # learned from the training lines: the most frequent neighbouring pair first',
      'print(len(merges), "merges, the first ones:", merges[:6])',
      '',
      'def merge(word, a, b):',
      '    out, i = [], 0',
      '    while i < len(word):',
      '        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:',
      '            out.append(a + b)',
      '            i += 2',
      '        else:',
      '            out.append(word[i])',
      '            i += 1',
      '    return out',
      '',
      'cache = {}',
      'def tokenize(msg):',
      '    tokens = []',
      String.raw`    for piece in re.findall(r"[A-Za-z_]+|[0-9]+| |[^\n]", msg):   # words, numbers, spaces, other characters`,
      '        if piece not in cache:',
      '            parts = list(piece)',
      '            for a, b in merges:   # replay the merges in the order they were learned',
      '                parts = merge(parts, a, b)',
      '            cache[piece] = parts',
      '        tokens += cache[piece]',
      '    return tokens',
    ],
  };

  // The page's TensorFlow/Keras code, one cell per section, each with the exact output it prints.
  // line: the selected or typed message; w: the selected tokenizer's model; pos: the position inspected.
  function pythonCells(line, w, data, { threshold = w.threshold, pos = 0, batch = '' } = {}) {
    const k = data.tokenizers.indexOf(w.tokenizer);
    const msgs = data.lines.map(r => r[0]);
    const labels = msgs.map(label);
    const count = name => labels.filter(l => l === name).length;
    const legitN = count('legit');
    const nTrain = Math.trunc(0.6 * legitN), nVal = Math.trunc(0.8 * legitN) - nTrain, nTest = legitN - nTrain - nVal;
    const r = run(line, w);
    const n = r.n, L = w.L;
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
        id: 'tokens',
        code: [
          `weights = json.load(open("python/weights_${w.tokenizer}.json"))   # the trained model for this tokenizer`,
          'vocab = weights["vocab"]   # every token in the training lines; 0 = padding, 1 = unknown',
          'index = {t: i for i, t in enumerate(vocab)}',
          `L = ${L}                    # every line becomes exactly L ids: cut or padded`,
          '',
          ...TOKENIZE_PY[w.tokenizer],
          '',
          'def encode(msg):',
          '    ids = [index.get(t, 1) for t in tokenize(msg)[:L]]',
          '    return ids + [0] * (L - len(ids))',
          '',
          `line = ${JSON.stringify(line)}`,
          'toks = tokenize(line)[:L]',
          'x = np.array([encode(line)])',
          'n = len(toks)',
          'print(len(vocab), "tokens, for example", vocab[2:10])',
          'print(toks)',
          'print(x[0, :n].tolist())',
          'print(L - n, "padding positions")',
        ].join('\n'),
        out: [
          ...(w.tokenizer === 'bpe' ? [`${w.merges.length} merges, the first ones: ${pyList(w.merges.slice(0, 6))}`] : []),
          `${w.vocab.length} tokens, for example ${pyList(w.vocab.slice(2, 10))}`,
          pyList(r.tokens),
          '[' + r.ids.slice(0, n).join(', ') + ']',
          `${L - n} padding positions`,
        ].join('\n'),
      },
      {
        id: 'model',
        code: [
          'def build_model(V):',
          '    return keras.Sequential([',
          '        keras.Input((L,), dtype="int32"),',
          '        layers.Embedding(V, 8, name="embedding"),',
          `        # Encoder: squeeze ${L} positions into ${L / 4}`,
          '        layers.Conv1D(16, 5, padding="same", activation="relu", name="enc_conv1"),',
          '        layers.MaxPooling1D(2, name="enc_pool1"),',
          '        layers.Conv1D(4, 5, padding="same", activation="relu", name="enc_conv2"),',
          '        layers.MaxPooling1D(2, name="bottleneck"),',
          `        # Decoder: expand back to ${L} positions`,
          '        layers.UpSampling1D(2, name="dec_up1"),',
          '        layers.Conv1D(4, 5, padding="same", activation="relu", name="dec_conv1"),',
          '        layers.UpSampling1D(2, name="dec_up2"),',
          '        layers.Conv1D(16, 5, padding="same", activation="relu", name="dec_conv2"),',
          '        layers.Conv1D(V, 1, activation="softmax", name="output"),   # a probability per token',
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
            `emb = outs["embedding"]   # (${L}, 8)`,
            `for t in range(${show}):`,
            '    print(f"{t:>3} {toks[t]!r:>12} " + " ".join(f"{v:7.4f}" for v in emb[t]))',
          ].join('\n'),
          out: r.emb.slice(0, show).map((row, t) => `${String(t).padStart(3)} ${pyRepr(r.tokens[t]).padStart(12)} ` + row.map(v => pyFixed(v, 4).padStart(7)).join(' ')).join('\n'),
        },
        {
          id: 'encoder',
          code: [
            '# enc_conv1: 16 filters, each looks at 5 neighbouring tokens (positions t-2 … t+2)',
            `print("enc_conv1 at position ${pos}:")`,
            `print(" ".join(f"{v:.3f}" for v in outs["enc_conv1"][${pos}]))`,
            `# The bottleneck: ${L / 4} positions × 4 numbers. Everything the decoder gets to see`,
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
            `probs = outs["output"]   # (${L}, V): a probability for every token in the vocabulary, at every position`,
            `pos = ${pos}`,
            'print("true token:", repr(toks[pos]))',
            'for i in np.argsort(-probs[pos])[:3]:',
            '    print(f"{vocab[i]!r:>12} {probs[pos, i]:.3f}")',
          ].join('\n'),
          out: [
            `true token: ${pyRepr(r.tokens[pos])}`,
            ...topk(r.probs[pos], w.vocab, 3).map(t => `${pyRepr(t.c).padStart(12)} ${pyFixed(t.p, 3)}`),
          ].join('\n'),
        },
        {
          id: 'reconstruct',
          code: [
            '# Error per token: -log(probability the model gave the true token)',
            'p_true = probs[np.arange(n), x[0, :n]]',
            'error = -np.log(np.maximum(p_true, 1e-7))',
            'recon = [{0: "·", 1: "?"}.get(i, vocab[i]) for i in probs[:n].argmax(axis=1)]',
            'score = error.mean()',
            `threshold = ${pyNum(threshold)}`,
            `print(${JSON.stringify(joiner(w))}.join(toks))`,
            `print(${JSON.stringify(joiner(w))}.join(recon))`,
            'print(f"score {score:.4f}, threshold {threshold}:", "anomaly" if score > threshold else "normal")',
          ].join('\n'),
          out: [
            r.tokens.join(joiner(w)),
            r.recon.join(joiner(w)),
            `score ${pyFixed(r.score, 4)}, threshold ${pyNum(threshold)}: ${r.score > threshold ? 'anomaly' : 'normal'}`,
          ].join('\n'),
        },
      );
    }
    const h = w.history;
    const d = detect(data.lines, threshold, k);
    const groups = [['validation legit', d.val], ['test legit', d.testLegit], ['attack', d.attack], ['other', d.other]];
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
      {
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
          ...groups.map(([name, [kk, of]]) => `${name.padStart(16)}: ${String(kk).padStart(4)} of ${String(of).padStart(4)} flagged`),
          `precision ${d.precision.toFixed(3)}, recall ${d.recall.toFixed(3)}`,
        ].join('\n'),
      },
    );
    const raws = batchLines(batch);
    if (raws.length) {
      const f = filterBatch(batch, { [w.tokenizer]: w }, { tok: w.tokenizer, threshold, mode: 'selected' });
      const kept = f.items.filter(x => x.kept).map(x => ({ s: x.scores[w.tokenizer], msg: x.msg }));
      kept.sort((a, b) => b.s - a.s);
      cells.push({
        id: 'batch',
        code: [
          '# Filter your own lines: score each one with the model, keep only those it can\'t rebuild',
          `batch_raw = ${JSON.stringify(raws)}`,
          String.raw`OTHER = re.compile(r"^\w{3} +\d+ [\d:]+ \S+ [^\s\[:]+(\[\d+\])?: ")   # a line from another program`,
          '',
          'def message(line):',
          '    m = SSHD.match(line)',
          '    if m:',
          '        return m.group(1)              # an sshd line: keep only its message',
          '    return None if OTHER.match(line) else line   # another program: skip; otherwise a bare message',
          '',
          'batch = [m for m in map(message, batch_raw) if m is not None]',
          'Xb = np.array([encode(m) for m in batch])',
          'Pb = model.predict(Xb, verbose=0)',
          'pb = np.take_along_axis(Pb, Xb[..., None], axis=2)[..., 0]',
          'real_b = Xb > 0',
          'batch_scores = (-np.log(np.maximum(pb, 1e-7)) * real_b).sum(axis=1) / np.maximum(real_b.sum(axis=1), 1)',
          '',
          '# Each score belongs to the line in the same position: keep the original line, not the reconstruction',
          'remaining = [(s, m) for s, m in zip(batch_scores, batch) if s > threshold]',
          'for s, m in sorted(remaining, key=lambda t: -t[0]):',
          '    print(f"{s:8.4f}  {m}")',
          'print(f"{len(remaining)} of {len(batch)} lines remain ({len(batch_raw) - len(batch)} skipped: not sshd)")',
        ].join('\n'),
        out: [
          ...kept.map(x => `${pyFixed(x.s, 4).padStart(8)}  ${x.msg}`),
          `${kept.length} of ${f.items.length} lines remain (${f.skipped.length} skipped: not sshd)`,
        ].join('\n'),
      });
    }
    return cells;
  }

  // The copy button's script: every cell except the (slow) training cell.
  const fullScript = cells =>
    '# Log autoencoder walkthrough: anomaly detection in SSH auth.log with a Keras Conv1D autoencoder.\n' +
    '# Run from the project folder (it reads data/auth.log and python/weights_<tokenizer>.json).\n\n' +
    cells.filter(c => !c.train).map(c => c.code).join('\n\n') + '\n';

  return { RULES, label, parseLine, batchLines, filterBatch, tokenize, joiner, unpack, encode, conv1d, maxpool, upsample, softmax, run, topk, detect, pythonCells, fullScript, pyFixed };
});

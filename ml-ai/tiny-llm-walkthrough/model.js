// The maths behind the Tiny LLM walkthrough, shared by index.html (browser) and the tests (Node).
// Pure functions, no DOM access. pythonCells() writes the same computation as NumPy code, together
// with the output that code prints; tests/check_outputs.py runs it and checks every line.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TinyLLM = api;
})(globalThis, function () {
  'use strict';

  // ---- Linear algebra on plain arrays (rows of numbers) ----
  const matmul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const add = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));
  const norm = v => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

  // Softmax over the first `upto` entries; the rest (masked) get weight 0.
  function softmax(row, upto = row.length, temperature = 1) {
    const live = row.slice(0, upto);
    const max = Math.max(...live);
    const e = live.map(v => Math.exp((v - max) / temperature));
    const sum = e.reduce((s, v) => s + v, 0);
    return row.map((_, j) => (j < upto ? e[j] / sum : 0));
  }

  // Sinusoidal position encoding from "Attention Is All You Need": even columns sin, odd columns cos.
  function positional(n, d) {
    return Array.from({ length: n }, (_, pos) => {
      const row = new Array(d).fill(0);
      for (let i = 0; i < d / 2; i++) {
        const angle = pos / Math.pow(10000, (2 * i) / d);
        row[2 * i] = Math.sin(angle);
        row[2 * i + 1] = Math.cos(angle);
      }
      return row;
    });
  }

  // Free text to tokens: lowercase, then split into words (runs of a–z) and single punctuation marks.
  // Pieces in the vocabulary become tokens; the rest are reported as unknown and skipped.
  // Same rule as the generated Python: re.findall(r"[a-z]+|[^\sa-z]", text.lower()). The u flag makes
  // JavaScript match whole code points (an emoji is one piece), as Python does.
  const PIECE = /[a-z]+|[^\sa-z]/gu;
  function tokenize(text, vocab) {
    const pieces = text.toLowerCase().match(PIECE) || [];
    const words = pieces.filter(p => vocab.includes(p));
    return { pieces, words, ids: words.map(p => vocab.indexOf(p)), unknown: pieces.filter(p => !vocab.includes(p)) };
  }

  // ---- Byte pair encoding: how real tokenizers build a subword vocabulary ----
  // Merges every non-overlapping a+b in a word (a list of strings), left to right.
  function mergePair(word, a, b) {
    const out = [];
    for (let i = 0; i < word.length; i++) {
      if (i + 1 < word.length && word[i] === a && word[i + 1] === b) { out.push(a + b); i++; }
      else out.push(word[i]);
    }
    return out;
  }

  // Learns up to numMerges merges from a list of words, starting from single characters.
  // Each step merges the most frequent neighbouring pair; ties go to the pair seen first
  // (Map and Python's dict both keep insertion order, and both pick the first maximum).
  function bpeTrain(words, numMerges) {
    let split = words.map(w => [...w]);
    const merges = [];
    for (let k = 0; k < numMerges; k++) {
      const counts = new Map();
      for (const w of split) {
        for (let i = 0; i + 1 < w.length; i++) {
          const key = JSON.stringify([w[i], w[i + 1]]);
          counts.set(key, (counts.get(key) || 0) + 1);
        }
      }
      if (!counts.size) break;   // every word is a single token already
      let best = null, count = 0;
      for (const [key, c] of counts) if (c > count) { best = key; count = c; }
      const [a, b] = JSON.parse(best);
      merges.push({ a, b, count });
      split = split.map(w => mergePair(w, a, b));
    }
    return merges;
  }

  // Splits one word into subword tokens by replaying the learned merges in order.
  const bpeEncode = (word, merges) => merges.reduce((parts, m) => mergePair(parts, m.a, m.b), [...word]);

  // The BPE step of the page: train on the corpus plus the user's text, then split the user's text.
  function bpe(text, corpus, numMerges) {
    const pieces = text.toLowerCase().match(PIECE) || [];
    const corpusWords = [...corpus, text].flatMap(line => line.toLowerCase().match(PIECE) || []);
    const merges = bpeTrain(corpusWords, numMerges);
    const tokens = pieces.map(p => bpeEncode(p, merges));
    return { pieces, merges, tokens, chars: pieces.reduce((s, p) => s + [...p].length, 0), count: tokens.reduce((s, t) => s + t.length, 0) };
  }

  // Cosine similarity between every pair of embedding rows (NaN for an all-zero row, as in NumPy).
  function cosine(E) {
    const unit = E.map(r => r.map(v => v / norm(r)));
    return matmul(unit, transpose(unit));
  }

  // One forward pass: a single attention head, a residual connection, and tied input/output embeddings.
  function run(ids, w, temperature = 1) {
    const n = ids.length, d = w.E[0].length;
    const tok = ids.map(i => w.E[i].slice());
    const pe = positional(n, d);
    const X = add(tok, pe);
    const Q = matmul(X, w.Wq), K = matmul(X, w.Wk), V = matmul(X, w.Wv);
    const raw = matmul(Q, transpose(K)).map(row => row.map(v => v / Math.sqrt(d)));
    const scores = raw.map((row, i) => row.map((v, j) => (j <= i ? v : null)));  // causal mask
    const A = raw.map((row, i) => softmax(row, i + 1));
    const C = matmul(A, V);
    const O = matmul(C, w.Wo);
    const H = add(X, O);
    const logits = matmul([H[n - 1]], transpose(w.E))[0];
    const probs = softmax(logits, logits.length, temperature);
    return { n, d, tok, pe, X, Q, K, V, scores, A, C, O, H, logits, probs };
  }

  // ---- Python output formatting (must match what the generated code prints) ----
  const pyFloat = v => (Number.isNaN(v) ? 'nan' : (Object.is(v, -0) ? '-' : '') + v.toFixed(3));
  const showLines = (M, labels) => M.map((row, i) => labels[i].padStart(5) + ' ' + row.map(v => pyFloat(v).padStart(7)).join(' '));
  const pyNum = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  const pyMatrix = (name, M, labels, comment) =>
    (comment ? `# ${comment}\n` : '') +
    `${name} = np.array([\n` + M.map((r, i) => `    [${r.map(pyNum).join(', ')}],` + (labels ? `  # ${labels[i]}` : '')).join('\n') + '\n])';
  const ranked = (vocab, probs) => vocab.map((word, i) => ({ word, p: probs[i] })).sort((a, b) => b.p - a.p);
  // Python's repr() of a str, for the strings tokenize() can produce (no control characters).
  const pyRepr = s => (s.includes("'") && !s.includes('"') ? `"${s.replace(/\\/g, '\\\\')}"` : `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`);

  // The NumPy version of the page, one cell per section, each with the exact output it prints.
  // With no known words there is nothing to run, so only the tokens, BPE and embeddings cells are returned.
  function pythonCells(text, w, { temperature = 1, merges = 10 } = {}) {
    const { words, ids, unknown } = tokenize(text, w.vocab);
    const b = bpe(text, w.corpus, merges);
    const vocabLit = '[' + w.vocab.map(t => JSON.stringify(t)).join(', ') + ']';
    const cells = [
      {
        id: 'tokens',
        code: [
          'import re',
          'import numpy as np',
          '',
          `vocab = ${vocabLit}`,
          `text = ${JSON.stringify(text)}`,
          'pieces = re.findall(r"[a-z]+|[^\\sa-z]", text.lower())   # words and punctuation marks',
          'words = [p for p in pieces if p in vocab]',
          'ids = [vocab.index(p) for p in words]',
          'print(ids)',
          'print("not in vocabulary:", [p for p in pieces if p not in vocab])',
          '',
          'def show(matrix, labels):',
          '    for label, row in zip(labels, matrix):',
          '        print(f"{label:>5} " + " ".join(f"{v:7.3f}" for v in row))',
        ].join('\n'),
        out: '[' + ids.join(', ') + ']\nnot in vocabulary: [' + unknown.map(pyRepr).join(', ') + ']',
      },
      {
        id: 'bpe',
        code: [
          '# Byte pair encoding: learn merges by counting pairs, then split the text with them.',
          '# Real tokenizers start from the 256 byte values; letters are used here so the pieces stay readable.',
          `corpus = [${w.corpus.map(c => JSON.stringify(c)).join(', ')}] + [text]`,
          'corpus_words = [p for line in corpus for p in re.findall(r"[a-z]+|[^\sa-z]", line.lower())]',
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
          'def bpe_train(words, num_merges):',
          '    words = [list(w) for w in words]',
          '    merges = []',
          '    for _ in range(num_merges):',
          '        counts = {}',
          '        for w in words:',
          '            for pair in zip(w, w[1:]):',
          '                counts[pair] = counts.get(pair, 0) + 1',
          '        if not counts:',
          '            break',
          '        (a, b), count = max(counts.items(), key=lambda kv: kv[1])   # ties: the pair seen first',
          '        merges.append((a, b, count))',
          '        words = [merge(w, a, b) for w in words]',
          '    return merges',
          '',
          'def bpe_encode(word, merges):',
          '    parts = list(word)',
          '    for a, b, _ in merges:',
          '        parts = merge(parts, a, b)',
          '    return parts',
          '',
          `merges = bpe_train(corpus_words, ${merges})`,
          'for k, (a, b, count) in enumerate(merges, 1):',
          '    print(f"{k:>2}. {a} + {b} -> {a + b}  ({count}x)")',
          'subwords = [bpe_encode(p, merges) for p in pieces]',
          'print(" ".join("+".join(parts) for parts in subwords))',
          'print(f"{sum(map(len, pieces))} characters -> {sum(map(len, subwords))} tokens")',
        ].join('\n'),
        out: [
          ...b.merges.map((m, k) => `${String(k + 1).padStart(2)}. ${m.a} + ${m.b} -> ${m.a + m.b}  (${m.count}x)`),
          b.tokens.map(t => t.join('+')).join(' '),
          `${b.chars} characters -> ${b.count} tokens`,
        ].join('\n'),
      },
      {
        id: 'embeddings',
        code: [
          pyMatrix('E', w.E, w.vocab, `Embedding table: one row per token. Columns: ${w.dims.join(', ')}`),
          'show(E[ids], words)              # look up one row per token',
          '',
          '# Cosine similarity: 1 = same direction, 0 = unrelated, -1 = opposite',
          'unit = E / np.linalg.norm(E, axis=1, keepdims=True)',
          'show(unit @ unit.T, vocab)',
        ].join('\n'),
        out: [...showLines(ids.map(i => w.E[i]), words), ...showLines(cosine(w.E), w.vocab)].join('\n'),
      },
    ];
    if (!ids.length) return cells;
    const r = run(ids, w, temperature);
    return cells.concat([
      {
        id: 'positions',
        code: [
          'def positional(n, d):',
          '    """Sinusoidal position encoding, as in \'Attention Is All You Need\'."""',
          '    pos = np.arange(n)[:, None]',
          '    i = np.arange(d // 2)[None, :]',
          '    angle = pos / 10000 ** (2 * i / d)',
          '    pe = np.zeros((n, d))',
          '    pe[:, 0::2] = np.sin(angle)',
          '    pe[:, 1::2] = np.cos(angle)',
          '    return pe',
          '',
          'd = E.shape[1]',
          'X = E[ids] + positional(len(ids), d)   # token meaning + position',
          'show(X, words)',
        ].join('\n'),
        out: showLines(r.X, words).join('\n'),
      },
      {
        id: 'attention',
        code: [
          pyMatrix('Wq', w.Wq, null, 'Learned projections: query (what am I looking for?), key (what do I contain?), value (what do I pass on?)'),
          pyMatrix('Wk', w.Wk),
          pyMatrix('Wv', w.Wv),
          'Q, K, V = X @ Wq, X @ Wk, X @ Wv',
          'scores = Q @ K.T / np.sqrt(d)',
          '',
          '# Causal mask: each word may only look at itself and earlier words',
          'n = len(ids)',
          'scores = np.where(np.tril(np.ones((n, n))) == 1, scores, -np.inf)',
          'A = np.exp(scores - scores.max(axis=1, keepdims=True))',
          'A = A / A.sum(axis=1, keepdims=True)   # softmax per row',
          'show(A, words)',
        ].join('\n'),
        out: showLines(r.A, words).join('\n'),
      },
      {
        id: 'next',
        code: [
          pyMatrix('Wo', w.Wo, null, 'Learned output projection'),
          'H = X + (A @ V) @ Wo       # residual: add what attention found to each position',
          'logits = H[-1] @ E.T       # compare the last position with every token embedding',
          '',
          `temperature = ${pyNum(temperature)}`,
          'p = np.exp((logits - logits.max()) / temperature)',
          'p = p / p.sum()',
          'for word, prob in sorted(zip(vocab, p), key=lambda t: -t[1]):',
          '    print(f"{word:>5} {prob:.3f}")',
        ].join('\n'),
        out: ranked(w.vocab, r.probs).map(t => t.word.padStart(5) + ' ' + pyFloat(t.p)).join('\n'),
      },
    ]);
  }

  const fullScript = cells =>
    '# Tiny LLM walkthrough: one forward pass of a single-head transformer, in NumPy.\n\n' +
    cells.map(c => c.code).join('\n\n') + '\n';

  return { matmul, transpose, softmax, positional, tokenize, PIECE, bpe, bpeTrain, bpeEncode, cosine, run, ranked, pythonCells, fullScript, pyFloat };
});

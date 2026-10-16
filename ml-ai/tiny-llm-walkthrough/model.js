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

  // Splits on spaces and looks each word up in the vocabulary. Unknown words are reported, not guessed.
  function tokenize(text, vocab) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    return { words, ids: words.map(w => vocab.indexOf(w)), unknown: words.filter(w => !vocab.includes(w)) };
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

  // The NumPy version of the page, one cell per section, each with the exact output it prints.
  function pythonCells(text, w, temperature = 1) {
    const { words, ids } = tokenize(text, w.vocab);
    const r = run(ids, w, temperature);
    const vocabLit = '[' + w.vocab.map(t => JSON.stringify(t)).join(', ') + ']';
    return [
      {
        id: 'tokens',
        code: [
          'import numpy as np',
          '',
          `vocab = ${vocabLit}`,
          `text = ${JSON.stringify(words.join(' '))}`,
          'words = text.split()',
          'ids = [vocab.index(word) for word in words]',
          'print(ids)',
          '',
          'def show(matrix, labels):',
          '    for label, row in zip(labels, matrix):',
          '        print(f"{label:>5} " + " ".join(f"{v:7.3f}" for v in row))',
        ].join('\n'),
        out: '[' + ids.join(', ') + ']',
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
        out: [...showLines(r.tok, words), ...showLines(cosine(w.E), w.vocab)].join('\n'),
      },
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
    ];
  }

  const fullScript = cells =>
    '# Tiny LLM walkthrough: one forward pass of a single-head transformer, in NumPy.\n\n' +
    cells.map(c => c.code).join('\n\n') + '\n';

  return { matmul, transpose, softmax, positional, tokenize, cosine, run, ranked, pythonCells, fullScript, pyFloat };
});

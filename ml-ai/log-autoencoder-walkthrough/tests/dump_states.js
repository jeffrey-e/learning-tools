// Runs the model in a range of page states and writes, for each one, the Keras cells the page shows
// (with the output it claims they print) and some raw arrays, to tests/states.json. Also checks that
// model.js reproduces every precomputed score in data.js and every rule label.
// tests/check_outputs.py then runs the code in real TensorFlow and compares.
const fs = require('fs');
const path = require('path');
require('../weights.js');
require('../data.js');
const M = require('../model.js');
const { DEFAULT_LINE, LINES, BATCH } = require('./states_config.js');

const W = globalThis.AE_WEIGHTS, D = globalThis.AE_DATA;
const fail = msg => { console.error('dump_states failed: ' + msg); process.exit(1); };

// One source for the maths: the JavaScript forward pass must give the scores Keras computed, for every tokenizer.
for (const [k, tok] of D.tokenizers.entries()) {
  const w = W[tok];
  let worst = 0;
  for (const [msg, lab, , scores] of D.lines) {
    if (k === 0 && M.label(msg) !== lab) fail(`label of "${msg}" is ${M.label(msg)} in model.js, ${lab} in data.js`);
    worst = Math.max(worst, Math.abs(M.run(msg, w).score - scores[k]));
  }
  if (worst > 1e-3) fail(`${tok}: model.js scores differ from data.js by up to ${worst}`);
  console.log(`${tok}: model.js reproduces all ${D.lines.length} scores in data.js (largest difference ${worst.toExponential(1)})`);
}

const states = [];
for (const [k, tok] of D.tokenizers.entries()) {
  const w = W[tok];
  const thresholds = [w.threshold, 0.0001, 0.0123, 3.5, 12];
  LINES.forEach((line, j) => {
    const r = M.run(line, w);
    const pos = [0, Math.max(0, r.n - 1), 7, 3][j % 4];
    const threshold = thresholds[j % thresholds.length];
    states.push({
      name: `${tok} | ${JSON.stringify(line.slice(0, 50))} | pos ${pos} | threshold ${threshold}`,
      first: j === 0 ? k : undefined,   // check the split and every score against data.js here
      cells: M.pythonCells(line, w, D, { threshold, pos, batch: j < 2 ? BATCH : '' }),   // the batch cell, at two thresholds
      arrays: r.n ? { bottleneck: r.z, output: r.probs.slice(0, r.n) } : {},
    });
  });
  states.push({ name: `${tok} | empty line`, cells: M.pythonCells('', w, D), arrays: {} });
}

fs.writeFileSync(path.join(__dirname, 'states.json'), JSON.stringify({
  states,
  script: M.fullScript(M.pythonCells(DEFAULT_LINE, W.chars, D)),
  splits: Object.fromEntries(['train', 'val', 'test'].map(s => [s, D.lines.flatMap((r, i) => (r[1] === 'legit' && r[2] === s ? [i] : []))])),
  scores: D.tokenizers.map((_, k) => D.lines.map(r => r[3][k])),
}));
console.log(`dumped ${states.length} states`);

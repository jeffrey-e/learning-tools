// Runs the model in a range of page states and writes, for each one, the Keras cells the page shows
// (with the output it claims they print) and some raw arrays, to tests/states.json. Also checks that
// model.js reproduces every precomputed score in data.js and every rule label.
// tests/check_outputs.py then runs the code in real TensorFlow and compares.
const fs = require('fs');
const path = require('path');
require('../weights.js');
require('../data.js');
const M = require('../model.js');
const { DEFAULT_LINE, LINES } = require('./states_config.js');

const W = globalThis.AE_WEIGHTS, D = globalThis.AE_DATA;
const fail = msg => { console.error('dump_states failed: ' + msg); process.exit(1); };

// One source for the maths: the JavaScript forward pass must give the scores Keras computed.
let worst = 0;
for (const [msg, lab, , score] of D.lines) {
  if (M.label(msg) !== lab) fail(`label of "${msg}" is ${M.label(msg)} in model.js, ${lab} in data.js`);
  worst = Math.max(worst, Math.abs(M.run(msg, W).score - score));
}
if (worst > 1e-3) fail(`model.js scores differ from data.js by up to ${worst}`);
console.log(`model.js reproduces all ${D.lines.length} scores in data.js (largest difference ${worst.toExponential(1)})`);

const THRESHOLDS = [W.threshold, 1, 3.5, 12];
const states = LINES.map((line, k) => {
  const r = M.run(line, W);
  const pos = [0, Math.max(0, r.n - 1), 7, 3][k % 4];
  return {
    name: `${JSON.stringify(line.slice(0, 50))} | pos ${pos} | threshold ${THRESHOLDS[k % THRESHOLDS.length]}`,
    cells: M.pythonCells(line, W, D, { threshold: THRESHOLDS[k % THRESHOLDS.length], pos }),
    arrays: r.n ? { bottleneck: r.z, output: r.probs.slice(0, r.n) } : {},
  };
});
states.push({ name: 'empty line', cells: M.pythonCells('', W, D), arrays: {} });

fs.writeFileSync(path.join(__dirname, 'states.json'), JSON.stringify({
  states,
  script: M.fullScript(M.pythonCells(DEFAULT_LINE, W, D)),
  splits: Object.fromEntries(['train', 'val', 'test'].map(s => [s, D.lines.flatMap((r, i) => (r[1] === 'legit' && r[2] === s ? [i] : []))])),
  scores: D.lines.map(r => r[3]),
}));
console.log(`dumped ${states.length} states`);

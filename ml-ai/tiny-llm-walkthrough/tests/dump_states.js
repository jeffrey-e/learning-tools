// Runs the model in a range of page states and writes, for each one, the NumPy cells the page
// shows (with the output it claims they print) and the raw arrays, to tests/states.json.
// tests/check_outputs.py then runs that code in real NumPy and compares.
const fs = require('fs');
const path = require('path');
require('../weights.js');
const M = require('../model.js');

const DEFAULTS = globalThis.LLM_WEIGHTS;
const TEXTS = ['the', 'the cat', 'the cat sat on the', 'the dog ran on the mat .', 'mat mat mat', 'on the cat'];
const TEMPS = [0.5, 1, 5];

// Weight variants: the defaults, plus the kinds of edits a user can make on the page.
function variants() {
  const edited = structuredClone(DEFAULTS);
  edited.E[1] = [0.3, -0.7, 1.25, 0];          // decimals, negatives and an integer
  edited.Wq[0][0] = -2;
  edited.Wo[3][1] = 0.001;
  const zeroRow = structuredClone(DEFAULTS);
  zeroRow.E[6] = [0, 0, 0, 0];                 // cosine similarity is undefined (nan) for this row
  return { default: DEFAULTS, edited, zeroRow };
}

const states = [];
for (const [name, w] of Object.entries(variants())) {
  for (const text of TEXTS) {
    for (const temperature of TEMPS) {
      const ids = M.tokenize(text, w.vocab).ids;
      const r = M.run(ids, w, temperature);
      states.push({
        name: `${name} | "${text}" | T=${temperature}`,
        cells: M.pythonCells(text, w, temperature),
        arrays: { X: r.X, A: r.A, H: r.H, logits: r.logits, p: r.probs },
      });
    }
  }
}

const out = path.join(__dirname, 'states.json');
fs.writeFileSync(out, JSON.stringify({ states, script: M.fullScript(M.pythonCells('the cat sat on the', DEFAULTS)) }));
console.log(`dumped ${states.length} states`);

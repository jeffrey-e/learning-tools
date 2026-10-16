// Writes python/llm_numpy.py: the page's NumPy code for the default sentence and weights.
// Run after changing model.js or weights.js (npm run generate).
const fs = require('fs');
const path = require('path');
require('../weights.js');
const M = require('../model.js');

const out = path.join(__dirname, '..', 'python', 'llm_numpy.py');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, M.fullScript(M.pythonCells('the cat sat on the', globalThis.LLM_WEIGHTS)));
console.log('wrote python/llm_numpy.py');

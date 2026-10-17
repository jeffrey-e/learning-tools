// Writes python/autoencoder_tf.py: the page's TensorFlow/Keras code for the default line.
// Run after changing model.js, weights.js or data.js (npm run generate).
const fs = require('fs');
const path = require('path');
require('../weights.js');
require('../data.js');
const M = require('../model.js');
const { DEFAULT_LINE } = require('../tests/states_config.js');

const out = path.join(__dirname, '..', 'python', 'autoencoder_tf.py');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, M.fullScript(M.pythonCells(DEFAULT_LINE, globalThis.AE_WEIGHTS, globalThis.AE_DATA)));
console.log('wrote python/autoencoder_tf.py');

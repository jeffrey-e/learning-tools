// Writes python/walkthrough.py: the page's Keras code for the default photo (training cells left out).
// Run after changing page.js or retraining (npm run generate).
const fs = require('fs');
const path = require('path');
require('../data.js');
const P = require('../page.js');
const { defaultPhoto } = require('../tests/states_config.js');

const out = path.join(__dirname, '..', 'python', 'walkthrough.py');
fs.writeFileSync(out, P.fullScript(P.pythonCells(defaultPhoto(globalThis.IMG_DATA), globalThis.IMG_DATA)));
console.log('wrote python/walkthrough.py');

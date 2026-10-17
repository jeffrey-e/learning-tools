// Writes, for a few test photos, the Keras cells the page shows and the output it claims they print,
// to tests/states.json. tests/check_outputs.py then runs that code in real TensorFlow and compares.
const fs = require('fs');
const path = require('path');
require('../data.js');
const P = require('../page.js');
const { defaultPhoto, photos } = require('./states_config.js');

const D = globalThis.IMG_DATA;
const states = photos(D).map(p => ({ name: `${p.id} (${p.label})`, cells: P.pythonCells(p, D) }));
fs.writeFileSync(path.join(__dirname, 'states.json'), JSON.stringify({ states, script: P.fullScript(P.pythonCells(defaultPhoto(D), D)) }));
console.log(`dumped ${states.length} states`);

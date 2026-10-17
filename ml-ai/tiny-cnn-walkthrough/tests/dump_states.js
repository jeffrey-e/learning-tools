// Loads index.html in a headless DOM (with TensorFlow.js from node_modules instead of the CDN),
// drives the page into a series of states, and writes every Python cell + printed output the page
// shows, plus the copyable scripts, to tests/states.json. check_outputs.py then runs that code in
// real NumPy / TensorFlow and compares. Also fails on any runtime error in the page.
const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

// The page's Subresource Integrity hash must be the hash of the pinned TF.js (the same bytes as the CDN)
{
  const tfFile = require.resolve('@tensorflow/tfjs/dist/tf.min.js');
  const sri = 'sha384-' + require('crypto').createHash('sha384').update(require('fs').readFileSync(tfFile)).digest('base64');
  if (!require('fs').readFileSync(require('path').join(__dirname, '..', 'index.html'), 'utf8').includes(`integrity="${sri}"`)) {
    console.error(`dump_states failed: the TF.js script tag needs integrity="${sri}"`);
    process.exit(1);
  }
}

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
  .replace(/<script src="https:\/\/cdn\.jsdelivr[^>]*><\/script>/, '');
const errors = [];
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  beforeParse(w) {
    w.tf = tf;
    w.HTMLCanvasElement.prototype.getContext = () => null; // jsdom has no canvas; thumbnails skip drawing
    w.addEventListener('error', e => errors.push(e.message));
  },
});
const w = dom.window, d = w.document, $ = id => d.getElementById(id);
const click = sel => (sel.startsWith('#') ? $(sel.slice(1)) : d.querySelector(sel)).click();
const set = (id, v) => { const e = $(id); e.value = v; e.dispatchEvent(new w.Event('input')); };
const setSelect = (id, v) => { const e = $(id); e.value = v; e.dispatchEvent(new w.Event('change')); };

const grab = (label, mode) => ({
  label, mode,
  cells: ['py1', 'py4', 'py5', 'py6'].flatMap(id => [...d.querySelectorAll(`#${id} .pycell`)].map(c => ({
    code: c.querySelector('.code').textContent,
    out: c.querySelector('.out') ? c.querySelector('.out').textContent : null,
  }))),
  full: w.eval('fullScript()'),
});

const states = [], training = [];
for (const mode of ['np', 'tf']) {
  click('#reset');
  if ($('relu-btn').getAttribute('aria-pressed') === 'false') click('#relu-btn');
  click('#k-v'); click('#mode-' + mode);
  click('[data-shape="vertical"]'); states.push(grab('default, vertical line', mode));
  click('[data-shape="square"]'); click('#next'); click('#next'); states.push(grab('square, window moved', mode));
  set('k-0-0', '0.3'); set('k-1-2', '-0.7'); set('w-2-3', '-1.25'); set('w-0-1', '0.1'); set('b-3', '0.3');
  states.push(grab('awkward decimals', mode));
  click('#relu-btn'); click('[data-shape="cross"]'); click('#k-h'); set('k-2-1', '2.5');
  states.push(grab('ReLU off, cross, horizontal kernel edited', mode));
  // trained weights (messy 4-decimal values)
  click('#relu-btn'); click('#t-rand');
  for (let i = 0; i < 40; i++) click('#t-epoch');
  for (const sh of ['vertical', 'horizontal', 'cross', 'square', 'blank']) {
    click(`[data-shape="${sh}"]`); states.push(grab('trained weights, ' + sh, mode));
  }
}
for (const mode of ['np', 'tf']) for (const opt of ['sgd', 'momentum', 'adam']) {
  click('#mode-' + mode); setSelect('t-opt', opt);
  training.push({ mode, opt, code: w.eval('pyTrain()') });
}
fs.writeFileSync(path.join(__dirname, 'states.json'), JSON.stringify({ states, training }));
if (errors.length) { console.error('Runtime errors in the page:', errors); process.exit(1); }
console.log(`dumped ${states.length} page states and ${training.length} training scripts`);

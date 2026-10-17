// Loads index.html headlessly (jsdom), clicks through the controls and checks that the page
// renders without errors and shows the same outputs as model.js.
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const root = path.join(__dirname, '..');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => errors.push(e.message));
virtualConsole.on('error', e => errors.push(String(e)));

// Inline the local scripts so the test doesn't depend on file:// loading.
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8')
  .replace(/<script src="(weights|data|model)\.js"><\/script>/g, (_, f) => `<script>${fs.readFileSync(path.join(root, f + '.js'), 'utf8')}</script>`);

const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole });
const { window } = dom;
const doc = window.document;
const fail = msg => { console.error('page smoke test failed: ' + msg); process.exit(1); };
const click = el => el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const outText = id => doc.querySelector(`.pycell[data-id="${id}"] pre.out`).textContent;
const M = window.LogAE, W = window.AE_WEIGHTS, D = window.AE_DATA;
const verdict = () => doc.querySelector('#verdict .verdict').textContent;
const type = text => { const t = doc.getElementById('text'); t.value = text; t.dispatchEvent(new window.Event('input')); };
const pos = () => [...doc.querySelectorAll('#chips .chip')].findIndex(c => c.getAttribute('aria-pressed') === 'true');
const threshold = () => Number(doc.getElementById('thr').value);
const checkCells = (text, what) => {
  for (const c of M.pythonCells(text, W, D, { threshold: threshold(), pos: pos() })) {
    if (outText(c.id) !== c.out) fail(`code output for ${what} / ${c.id} differs from model.js`);
  }
};

if (errors.length) fail(errors.join('\n'));
if (verdict() !== 'Normal') fail(`the default line is "${verdict()}", expected Normal`);

// Every preset renders, its code panels match model.js, and its verdict matches its score.
const select = doc.getElementById('preset');
const presets = [...select.options].filter(o => !o.disabled);
for (const o of presets) {
  select.value = o.value;
  select.dispatchEvent(new window.Event('change'));
  const text = doc.getElementById('text').value;
  const row = D.lines.find(r => r[0] === text);
  if (!row) fail(`preset "${o.textContent}" is not a line of the log`);
  const expect = row[3] > W.threshold ? 'Anomaly' : 'Normal';
  if (verdict() !== expect) fail(`preset "${o.textContent}" shows ${verdict()}, its score says ${expect}`);
  const kind = o.textContent.split(':')[0];
  if ((kind === 'Attack' || kind === 'False alarm' || kind === 'Other') !== (expect === 'Anomaly')) fail(`preset "${o.textContent}" is named for the wrong verdict`);
  checkCells(text, `preset "${o.textContent}"`);
}

// Clicking a character selects it everywhere and updates the decoder code.
select.value = '0';
select.dispatchEvent(new window.Event('change'));
click(doc.querySelectorAll('#chips .chip')[5]);
if (pos() !== 5) fail('clicking a character did not select it');
if (!outText('decoder').startsWith("true character: 'v'")) fail('the decoder cell does not follow the selected character');
click(doc.querySelectorAll('#pair .c')[9]);
if (pos() !== 9) fail('clicking the reconstruction did not select the character');

// Free text: unknown characters are marked, the model still runs.
type('Accepted password for mallory from 203.0.113.9 port 2222 ssh2 é');
if (doc.querySelectorAll('#chips .chip.unk').length !== 1) fail('the unknown character was not marked');
if (select.value !== 'custom') fail('the toolbar does not show "your own text"');
checkCells(doc.getElementById('text').value, 'free text');

// Threshold: moving it changes the counts and the lists, the button restores it.
const counts = () => doc.getElementById('counts').textContent;
const before = counts();
const slider = doc.getElementById('thr');
slider.value = '4';
slider.dispatchEvent(new window.Event('input'));
if (counts() === before) fail('moving the threshold did not change the counts');
if (doc.querySelectorAll('#l-missed .item').length === 0) fail('a threshold of 4 should miss some attacks');
checkCells(doc.getElementById('text').value, 'threshold 4');
click(doc.getElementById('thr-reset'));
if (threshold() !== W.threshold || counts() !== before) fail('the 99th percentile button did not restore the threshold');

// Lists render, with a reconstruction under every line, and "show more" adds lines.
for (const id of ['l-flagged', 'l-legit', 'l-false', 'l-other', 'l-brute']) {
  if (!doc.querySelector(`#${id} .item .pair`)) fail(`list ${id} is empty`);
}
const n = doc.querySelectorAll('#l-flagged .item').length;
click(doc.querySelector('[data-more="flagged"]'));
if (doc.querySelectorAll('#l-flagged .item').length <= n) fail('show more did not add lines');
click(doc.querySelector('#l-legit .item button'));
const inspected = D.lines.find(r => r[0] === doc.getElementById('text').value);
if (!inspected || inspected[1] !== 'legit' || verdict() !== 'Normal') fail('inspect did not load a legitimate line');

// An empty line: the model steps are hidden and nothing throws.
type('');
if (doc.getElementById('empty-note').hidden || !doc.getElementById('steps').hidden) fail('the empty line is not handled');
type('Failed password for root from 1.2.3.4 port 22 ssh2');
if (doc.getElementById('steps').hidden || verdict() !== 'Anomaly') fail('the steps did not come back');

if (errors.length) fail(errors.join('\n'));
console.log('page smoke test: passed');

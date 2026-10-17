// Loads index.html headlessly (jsdom), clicks through the controls with every tokenizer and checks
// that the page renders without errors and shows the same outputs as model.js.
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
const choose = (select, value) => { select.value = value; select.dispatchEvent(new window.Event('change')); };
const outText = id => doc.querySelector(`.pycell[data-id="${id}"] pre.out`).textContent;
const M = window.LogAE, MODELS = window.AE_WEIGHTS, D = window.AE_DATA;
const verdict = () => doc.querySelector('#verdict .verdict').textContent;
const type = text => { const t = doc.getElementById('text'); t.value = text; t.dispatchEvent(new window.Event('input')); };
const pos = () => [...doc.querySelectorAll('#chips .chip')].findIndex(c => c.getAttribute('aria-pressed') === 'true');
const threshold = () => Number(doc.getElementById('thr-out').textContent);
const tokSelect = doc.getElementById('tok'), lineSelect = doc.getElementById('preset');
const tok = () => tokSelect.value;
const checkCells = what => {
  for (const c of M.pythonCells(doc.getElementById('text').value, MODELS[tok()], D, { threshold: threshold(), pos: pos() })) {
    if (outText(c.id) !== c.out) fail(`code output for ${what} / ${c.id} (${tok()}) differs from model.js`);
  }
};

if (errors.length) fail(errors.join('\n'));
if (tok() !== 'chars') fail(`the page starts with the ${tok()} tokenizer, expected chars`);
if (verdict() !== 'Normal') fail(`the default line is "${verdict()}", expected Normal`);
if (doc.querySelectorAll('#cmp tbody tr').length !== D.tokenizers.length) fail('the comparison table needs one row per tokenizer');

const presets = [...lineSelect.options].filter(o => !o.disabled);
for (const t of D.tokenizers) {
  choose(tokSelect, t);
  const k = D.tokenizers.indexOf(t), w = MODELS[t];
  if (threshold() !== w.threshold) fail(`${t}: switching did not reset the threshold to the model's default`);
  if (!doc.querySelector('#cmp tr.on') || doc.querySelector('#cmp tr.on').dataset.tok !== t) fail(`${t}: the comparison table does not mark the selected tokenizer`);

  // Every preset renders, is a line of the log with the label its name says, and its verdict matches its score.
  for (const o of presets) {
    choose(lineSelect, o.value);
    const text = doc.getElementById('text').value;
    const row = D.lines.find(r => r[0] === text);
    if (!row) fail(`preset "${o.textContent}" is not a line of the log`);
    const kind = { Legitimate: 'legit', Attack: 'attack', Other: 'other' }[o.textContent.split(':')[0]];
    if (row[1] !== kind) fail(`preset "${o.textContent}" is labelled ${row[1]}`);
    const expect = row[3][k] > w.threshold ? 'Anomaly' : 'Normal';
    if (verdict() !== expect) fail(`${t}: preset "${o.textContent}" shows ${verdict()}, its score says ${expect}`);
    if (kind === 'attack' && expect !== 'Anomaly') fail(`${t}: attack preset "${o.textContent}" is not flagged`);
    checkCells(`preset "${o.textContent}"`);
  }

  // Clicking a token selects it everywhere and updates the decoder code.
  choose(lineSelect, '0');
  click(doc.querySelectorAll('#chips .chip')[2]);
  if (pos() !== 2) fail(`${t}: clicking a token did not select it`);
  if (!doc.querySelector('.pycell[data-id="decoder"] pre.code').textContent.includes('pos = 2')) fail(`${t}: the decoder cell does not follow the selected token`);
  click(doc.querySelectorAll('#pair .c')[1]);
  if (pos() !== 1) fail(`${t}: clicking the reconstruction did not select the token`);

  // Free text: an unknown character is marked, the model still runs.
  type('Accepted password for mallory from 203.0.113.9 port 2222 ssh2 é');
  if (!doc.querySelector('#chips .chip.unk')) fail(`${t}: the unknown token was not marked`);
  if (lineSelect.value !== 'custom') fail('the toolbar does not show "your own text"');
  checkCells('free text');

  // Threshold: moving it changes the counts and the lists, the button restores it.
  const counts = () => doc.getElementById('counts').textContent;
  const before = counts();
  const slider = doc.getElementById('thr');
  slider.value = '1.1';   // log10: a threshold of about 12.6
  slider.dispatchEvent(new window.Event('input'));
  if (threshold() !== 12.6) fail(`${t}: the logarithmic slider gave ${threshold()}, expected 12.6`);
  if (counts() === before) fail(`${t}: moving the threshold did not change the counts`);
  if (doc.querySelectorAll('#l-missed .item').length === 0) fail(`${t}: a threshold of 12.6 should miss some attacks`);
  checkCells('threshold 12.6');
  click(doc.getElementById('thr-reset'));
  if (threshold() !== w.threshold || counts() !== before) fail(`${t}: the 99th percentile button did not restore the threshold`);

  // Lists render, with a reconstruction under every line.
  for (const id of ['l-flagged', 'l-legit', 'l-other', 'l-brute']) {
    if (!doc.querySelector(`#${id} .item .pair`)) fail(`${t}: list ${id} is empty`);
  }
}

// Show more adds lines; inspect loads a line.
const n = doc.querySelectorAll('#l-flagged .item').length;
click(doc.querySelector('[data-more="flagged"]'));
if (doc.querySelectorAll('#l-flagged .item').length <= n) fail('show more did not add lines');
click(doc.querySelector('#l-legit .item button'));
const inspected = D.lines.find(r => r[0] === doc.getElementById('text').value);
if (!inspected || inspected[1] !== 'legit' || verdict() !== 'Normal') fail('inspect did not load a legitimate line');

// The comparison table switches tokenizer.
click(doc.querySelector('#cmp tbody tr[data-tok="words"]'));
if (tok() !== 'words') fail('clicking a comparison row did not switch the tokenizer');

// An empty line: the model steps are hidden and nothing throws.
type('');
if (doc.getElementById('empty-note').hidden || !doc.getElementById('steps').hidden) fail('the empty line is not handled');
type('Failed password for root from 1.2.3.4 port 22 ssh2');
if (doc.getElementById('steps').hidden || verdict() !== 'Anomaly') fail('the steps did not come back');

if (errors.length) fail(errors.join('\n'));
console.log(`page smoke test: passed (${D.tokenizers.length} tokenizers)`);

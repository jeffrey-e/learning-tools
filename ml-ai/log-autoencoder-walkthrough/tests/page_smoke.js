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
  for (const c of M.pythonCells(doc.getElementById('text').value, MODELS[tok()], D, { threshold: threshold(), pos: pos(), batch: doc.getElementById('batch').value })) {
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

// 10. Filter your own lines.
const { BATCH } = require('./states_config.js');
const paste = text => { const t = doc.getElementById('batch'); t.value = text; t.dispatchEvent(new window.Event('input')); };
const metricV = i => Number(doc.querySelectorAll('#b-counts .metric .v')[i].textContent);
choose(tokSelect, 'chars');
paste(BATCH);
if (metricV(0) !== 9 || metricV(1) !== 2) fail(`the batch should have 9 lines and 2 skipped, got ${metricV(0)} and ${metricV(1)}`);
if (metricV(2) + metricV(3) !== 7) fail('removed + remaining should be the 7 sshd lines');
const f = M.filterBatch(BATCH, MODELS, { tok: 'chars', threshold: MODELS.chars.threshold, mode: 'selected' });
const keptMsgs = [...doc.querySelectorAll('#b-kept .item .orig')].map(e => e.textContent);
const expected = f.items.filter(x => x.scores.chars > MODELS.chars.threshold).map(x => x.msg);
if (keptMsgs.length !== expected.length || !expected.every(m => keptMsgs.includes(m))) fail('the remaining lines are not the lines scoring above the threshold');
if (!keptMsgs.includes('Invalid user oracle from 203.0.113.50')) fail('an attack line not in the log was not kept');
if (keptMsgs.some(m => m.startsWith('Mar 30'))) fail('the sshd prefix was not stripped');
if (doc.querySelectorAll('#b-kept .vote').length !== 4 * keptMsgs.length) fail('every remaining line needs four vote chips');
if (doc.getElementById('b-check').hidden) fail('lines from the log should be checked against their labels');
checkCells('batch');
const modeTo = v => { const r = doc.querySelector(`input[name="b-mode"][value="${v}"]`); r.checked = true; r.dispatchEvent(new window.Event('change', { bubbles: true })); };
modeTo('all');
const allN = metricV(3);
modeTo('any');
const anyN = metricV(3);
if (anyN < allN) fail('"any" must keep at least as many lines as "all"');
if (!doc.querySelector('.pycell[data-id="batch"]').parentElement.querySelector('.py-note')) fail('the vote modes need the note on the code panel');
modeTo('selected');
click(doc.getElementById('b-sample'));
if (metricV(0) !== 20 || doc.getElementById('b-lists').hidden) fail('the sample did not load 20 lines');
checkCells('sample batch');
click(doc.getElementById('b-clear'));
if (!doc.getElementById('b-lists').hidden || doc.querySelector('.pycell[data-id="batch"]')) fail('clearing did not empty the batch');

// An empty line: the model steps are hidden and nothing throws.
type('');
if (doc.getElementById('empty-note').hidden || !doc.getElementById('steps').hidden) fail('the empty line is not handled');
type('Failed password for root from 1.2.3.4 port 22 ssh2');
if (doc.getElementById('steps').hidden || verdict() !== 'Anomaly') fail('the steps did not come back');

if (errors.length) fail(errors.join('\n'));
console.log(`page smoke test: passed (${D.tokenizers.length} tokenizers)`);

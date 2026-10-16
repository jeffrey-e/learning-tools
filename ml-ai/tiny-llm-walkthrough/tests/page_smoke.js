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
  .replace(/<script src="(weights|model)\.js"><\/script>/g, (_, f) => `<script>${fs.readFileSync(path.join(root, f + '.js'), 'utf8')}</script>`);

const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole });
const { window } = dom;
const doc = window.document;
const fail = msg => { console.error('page smoke test failed: ' + msg); process.exit(1); };
const click = el => el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const outText = id => doc.querySelector(`details[data-cell="${id}"] pre.out`).textContent;
const topWord = () => doc.querySelector('#bars .win .n').textContent.split(' ')[0];

if (errors.length) fail(errors.join('\n'));
if (topWord() !== 'mat') fail(`default prediction is "${topWord()}", expected "mat"`);

// Every preset and temperature renders, and the code panels match model.js.
const M = window.TinyLLM;
for (const b of doc.querySelectorAll('#presets button')) {
  click(b);
  const cells = M.pythonCells(b.dataset.text, window.LLM_WEIGHTS, { merges: 10 });
  for (const c of cells) if (outText(c.id) !== c.out) fail(`code output for "${b.dataset.text}" / ${c.id} differs from model.js`);
}
for (const b of doc.querySelectorAll('#temps button')) click(b);

// Editing a weight changes the output, and reset brings it back.
click(doc.querySelector('#presets button[data-text="the cat sat on the"]'));
const before = outText('next');
const input = doc.querySelector('#t-Wo input');
input.value = '9';
input.dispatchEvent(new window.Event('change'));
if (outText('next') === before) fail('editing Wo did not change the prediction');
click(doc.getElementById('reset'));
if (outText('next') !== before) fail('reset did not restore the prediction');

// Generating appends a word; removing takes it off again.
click(doc.getElementById('add-top'));
if (doc.querySelectorAll('#chips .chip').length !== 6) fail('add the top word did not add a token');
click(doc.getElementById('pop'));
if (doc.querySelectorAll('#chips .chip').length !== 5) fail('remove last word did not remove a token');

// Free text: unknown words are skipped, the model still runs, and the code matches model.js.
const type = text => { const t = doc.getElementById('text'); t.value = text; t.dispatchEvent(new window.Event('input')); };
click(doc.querySelector('#temps button[data-t="1"]'));
type('The big cat sat on the old mat.');
if (doc.querySelectorAll('#chips .chip.unk').length !== 2) fail('unknown words were not marked as skipped');
if (doc.querySelectorAll('#chips button.chip').length !== 7) fail('known words were not turned into tokens');
if (doc.getElementById('unknown-note').hidden) fail('the unknown-word note is not shown');
for (const c of M.pythonCells('The big cat sat on the old mat.', window.LLM_WEIGHTS, { merges: 10 })) {
  if (outText(c.id) !== c.out) fail(`code output for free text / ${c.id} differs from model.js`);
}
click(doc.getElementById('pop'));
if (doc.getElementById('text').value !== 'The big cat sat on the old mat') fail('remove last word did not remove the final piece');

// BPE: unknown words still get subword tokens, and the merge count is adjustable.
if (doc.querySelectorAll('#subwords .subword').length !== 8) fail('BPE did not split every word of the free text');
const slider = doc.getElementById('merges');
slider.value = '0';
slider.dispatchEvent(new window.Event('input'));
if (doc.querySelectorAll('#t-merges tr').length !== 1) fail('with 0 merges the merge table should be empty');
if (outText('bpe') !== M.pythonCells('The big cat sat on the old mat', window.LLM_WEIGHTS, { merges: 0 }).find(c => c.id === 'bpe').out) fail('BPE code output differs from model.js');
slider.value = '10';
slider.dispatchEvent(new window.Event('input'));

// No known words: steps 3–5 are hidden and nothing throws.
type('hello world');
if (doc.getElementById('empty-note').hidden) fail('the no-known-words message is not shown');
if (!doc.getElementById('s5').closest('section').hidden) fail('step 5 is shown without any tokens');
type('the cat');
if (doc.getElementById('s5').closest('section').hidden) fail('step 5 did not come back');

if (errors.length) fail(errors.join('\n'));
console.log('page smoke test: passed');

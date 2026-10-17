// Opens the page in headless Chromium (Playwright), lets TensorFlow.js load both models, and clicks
// through it: photo selection, feature maps and heatmaps, the mistakes gallery, the code panels and
// your own photo. A real browser is needed: the page decodes images and runs TF.js on canvases.
// TensorFlow.js is served from node_modules (the same pinned version as the CDN), so no network is needed.
const fs = require('fs');
const http = require('http');
const path = require('path');
const { chromium } = require('playwright');

const site = path.join(__dirname, '..', '..', '..');   // the repository root, so ../../assets/site.css resolves
const project = '/ml-ai/image-recognition-walkthrough/';
const TFJS = require.resolve('@tensorflow/tfjs/dist/tf.min.js');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.jpg': 'image/jpeg', '.bin': 'application/octet-stream' };
const fail = msg => { console.error('page smoke test failed: ' + msg); process.exit(1); };

const server = http.createServer((req, res) => {
  const file = path.join(site, decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/\/$/, '/index.html'));
  if (!file.startsWith(site) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});

(async () => {
  // The Subresource Integrity hash in the page must be the hash of the pinned TF.js (same bytes as the CDN)
  const sri = 'sha384-' + require('crypto').createHash('sha384').update(fs.readFileSync(TFJS)).digest('base64');
  if (!fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8').includes(`integrity="${sri}"`)) fail(`the TF.js script tag needs integrity="${sri}"`);
  await new Promise(ok => server.listen(0, '127.0.0.1', ok));
  const base = `http://127.0.0.1:${server.address().port}${project}`;
  const browser = await chromium.launch();
  const errors = [];
  try {
    for (const width of [400, 1360]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      await page.route('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js', r => r.fulfill({ path: TFJS, contentType: 'text/javascript' }));
      await page.goto(base);
      await page.waitForFunction(() => document.getElementById('engine').textContent.startsWith('Models loaded'), null, { timeout: 120000 });
      await page.waitForFunction(() => document.querySelectorAll('#maps3 canvas').length === 8, null, { timeout: 60000 });
      if (errors.length) fail(errors.join('\n'));
      if (await page.evaluate(() => document.documentElement.scrollWidth) > width) fail(`horizontal scroll at ${width}px`);
      if (width === 1360) continue;

      // The page's numbers come from data.js; the code panels must match page.js for the selected photo.
      const check = async what => {
        const bad = await page.evaluate(() => {
          const id = document.querySelector('#pick b').textContent;
          const photo = IMG_DATA.photos.find(p => p.id === id);
          return ImgPage.pythonCells(photo, IMG_DATA).filter(c => document.querySelector(`.pycell[data-id="${c.id}"] pre.out`)?.textContent !== c.out).map(c => c.id);
        });
        if (bad.length) fail(`${what}: code panels differ from page.js: ${bad.join(', ')}`);
      };
      await check('default photo');
      // Live predictions agree with Keras (model B is float16 in the browser): "in your browser; Keras: 87%"
      const live = parseInt(await page.textContent('#pred-b .metric:nth-child(2) .v'), 10);
      const keras = parseInt((await page.textContent('#pred-b .metric:nth-child(2) .s')).match(/Keras: (\d+)%/)[1], 10);
      if (Math.abs(live - keras) > 2) fail(`the browser says ${live}% malamute, Keras ${keras}%`);

      // The heatmap is drawn (not a blank canvas)
      const lit = await page.evaluate(() => { const d = document.getElementById('cam-b').getContext('2d').getImageData(0, 0, 224, 224).data; let s = 0; for (let i = 0; i < d.length; i += 4) s += d[i]; return s; });
      if (!lit) fail('the class activation map is empty');

      // Selecting another photo updates everything
      await page.click('#next');
      await page.waitForTimeout(1500);
      await check('next photo');
      await page.click('#g-test .ph:nth-child(5)');
      await page.waitForTimeout(1500);
      await check('gallery photo');

      // The mistakes gallery has a filter for each model
      const nB = await page.locator('#g-mistakes .ph').count();
      await page.click('#g-filter button[data-f="a"]');
      const nA = await page.locator('#g-mistakes .ph').count();
      await page.click('#g-filter button[data-f="all"]');
      if (await page.locator('#g-mistakes .ph').count() !== await page.locator('#g-test .ph').count()) fail('"all test photos" does not show every test photo');
      if (nA <= nB) fail(`model A should make more mistakes than model B (${nA} vs ${nB})`);

      // Tables and charts
      if (await page.locator('#exp-a tr').count() < 3 || await page.locator('#exp-b tr').count() < 3) fail('the experiment tables are empty');
      if (await page.locator('#exp-a tr.chosen').count() !== 1 || await page.locator('#exp-b tr.chosen').count() !== 1) fail('each experiment table must mark one chosen run');
      if (await page.locator('#credits li').count() < 400) fail('the credits list is incomplete');

      // Your own photo, through the file input
      const ownFile = path.join(__dirname, '..', JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'photos.json')))[0].file);
      await page.setInputFiles('#file', ownFile);
      await page.waitForFunction(() => !document.getElementById('up-out').hidden, null, { timeout: 30000 });
      const said = await page.textContent('#up-b');
      if (!/^Model B: (husky|malamute) \(\d+%\)$/.test(said)) fail(`unexpected result for an uploaded photo: ${said}`);
      await page.setInputFiles('#file', { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('not an image') });
      await page.waitForFunction(() => !document.getElementById('up-err').hidden, null, { timeout: 10000 });

      // Show all code opens every panel
      await page.click('#code-all');
      if (await page.locator('details.py:not([open])').count()) fail('show all code left a panel closed');
      if (errors.length) fail(errors.join('\n'));
    }
  } finally {
    await browser.close();
    server.close();
  }
  console.log('page smoke test: passed (400px and 1360px)');
})().catch(e => fail(e.stack || String(e)));

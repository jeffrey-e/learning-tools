// Loads the TensorFlow.js models the page uses (model/scratch, model/transfer) in Node and checks them
// against Keras on the same decoded pixels (tests/tfjs_ref.json, written by check_outputs.py).
// Model A is float32 and must match closely; model B is stored in float16, so it may be about 0.01 off.
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

const root = path.join(__dirname, '..');
const fail = msg => { console.error('check_tfjs failed: ' + msg); process.exit(1); };
function handler(dir) {
  return {
    load: async () => {
      const mj = JSON.parse(fs.readFileSync(path.join(dir, 'model.json')));
      const all = Buffer.concat(mj.weightsManifest.flatMap(g => g.paths.map(p => fs.readFileSync(path.join(dir, p)))));
      return { modelTopology: mj.modelTopology, weightSpecs: mj.weightsManifest.flatMap(g => g.weights), weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.length) };
    },
  };
}
(async () => {
  const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'tfjs_ref.json')));
  const a = await tf.loadLayersModel(handler(path.join(root, 'model', 'scratch')));
  const b = await tf.loadLayersModel(handler(path.join(root, 'model', 'transfer')));
  let da = 0, db = 0;
  for (const r of ref) {
    const x = tf.tensor(r.pixels, [1, 224, 224, 3], 'float32');
    const pa = (await a.predict(x)[2].data())[1], pb = (await b.predict(x)[3].data())[1];
    da = Math.max(da, Math.abs(pa - r.pa)); db = Math.max(db, Math.abs(pb - r.pb));
  }
  if (da > 1e-3) fail(`model A in TF.js differs from Keras by ${da}`);
  if (db > 0.02) fail(`model B in TF.js differs from Keras by ${db}`);
  console.log(`TF.js models match Keras on ${ref.length} photos (largest difference: A ${da.toExponential(1)}, B ${db.toExponential(1)})`);
})();

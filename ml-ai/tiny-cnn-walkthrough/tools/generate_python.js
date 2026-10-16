// Regenerates python/*.py from index.html's default state, and writes the notebook's cell
// sources to tools/nbcells.json for build_notebook.py. Run via: npm run generate
const {JSDOM}=require('jsdom');const fs=require('fs');const tf=require('@tensorflow/tfjs');
const path=require('path');const R=path.join(__dirname,'..');
const html=fs.readFileSync(R+'/index.html','utf8').replace(/<script src="https:\/\/cdn\.jsdelivr[^>]*><\/script>/,'');
const dom=new JSDOM(html,{runScripts:'dangerously',beforeParse(w){w.tf=tf;w.HTMLCanvasElement.prototype.getContext=()=>null;}});
const w=dom.window,d=w.document;const $=id=>d.getElementById(id);
const cells=()=>['py1','py4','py5','py6'].map(id=>[...d.querySelectorAll(`#${id} .pycell`)].map(c=>({code:c.querySelector('.code').textContent,out:c.querySelector('.out')?.textContent??null,skip:!!c.dataset.skip})));
const hdr=t=>`"""${t}

Generated from the Tiny CNN Walkthrough page (index.html) with its default settings:
the vertical-line image and the hand-set weights. The page's copy buttons give a
version with your own drawing, weights or training settings.
"""

`;
const out={};
$('mode-np').click();
fs.writeFileSync(R+'/python/cnn_numpy.py',hdr('Forward pass of the tiny CNN in NumPy: convolution, ReLU, max pooling, flatten,\ndense layer and softmax on a 6x6 image.')+w.eval('fullScript()'));
fs.writeFileSync(R+'/python/train_numpy.py',hdr('Train the tiny CNN from random weights with hand-written backpropagation in NumPy\n(plain SGD). The page shows the update() function for SGD + momentum and Adam.')+w.eval('pyTrain()'));
out.np=cells();out.trainNp=w.eval('pyTrain()');
$('mode-tf').click();
fs.writeFileSync(R+'/python/cnn_tensorflow.py',hdr('Forward pass of the tiny CNN with TensorFlow ops and a Keras Dense layer.')+w.eval('fullScript()'));
out.tf=cells();
$('t-opt').value='adam';$('t-opt').dispatchEvent(new w.Event('change'));
fs.writeFileSync(R+'/python/train_tensorflow.py',hdr('Train the tiny CNN with Keras model.fit and the Adam optimizer.')+w.eval('pyTrain()'));
out.trainTf=w.eval('pyTrain()');
fs.writeFileSync(path.join(__dirname,'nbcells.json'),JSON.stringify(out));
console.log('generated 4 scripts');

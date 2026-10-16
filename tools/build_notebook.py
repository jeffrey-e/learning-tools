"""Builds and executes notebooks/tiny_cnn_walkthrough.ipynb from tools/nbcells.json
(written by generate_python.js). Needs: pip install nbformat nbclient ipykernel"""
import json
import os
import pathlib

import nbformat as nbf
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((HERE / "nbcells.json").read_text())
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
stages = [
    "### Steps 1–3: convolution, ReLU and max pooling\nA 3×3 kernel slides over the 6×6 image (6 − 3 + 1 = 4 positions per axis), "
    "ReLU sets negatives to 0, and 2×2 max pooling keeps the largest value of each block.",
    "### Step 4: flatten\nBoth kernels run on the same image; their pooled 2×2 maps are laid end to end as one vector of 8 numbers.",
    "### Step 5: dense layer\nOne row of 8 weights plus a bias per class. Each logit is the dot product of its row with the flattened vector, plus the bias.",
    "### Step 6: softmax prediction\nSoftmax turns the logits into probabilities; the highest one is the prediction.",
]
cells = [
    md("# Tiny CNN Walkthrough: notebook\n\nThe same network as the interactive page (`index.html`): a 6×6 image, two 3×3 kernels, ReLU, "
       "2×2 max pooling, flatten, a dense layer and softmax, classifying a drawing as a vertical line, horizontal line, cross/box or empty.\n\n"
       "Part 1 does the forward pass in NumPy, part 2 does it with TensorFlow, and part 3 trains the weights from random values. "
       "Every printed output in parts 1 and 2 matches the interactive page.\n\n"
       "In Google Colab everything is preinstalled. Locally, run `pip install -r requirements.txt` first."),
    md("## Part 1: forward pass in NumPy\nThe weights are the hand-set values from the page, so the output is predictable."),
]
for s, group in zip(stages, d["np"]):
    cells.append(md(s))
    cells += [code(c["code"]) for c in group]
cells.append(md("## Part 2: the same forward pass with TensorFlow\n`tf.nn.conv2d`, `tf.nn.relu` and `tf.nn.max_pool2d` on a "
                "`(1, 6, 6, 1)` tensor, then a real Keras `Dense` layer. TensorFlow computes in float32, so the printouts are limited to 4 decimals."))
for s, group in zip(stages, d["tf"]):
    cells.append(md(s.split("\n")[0]))
    cells += [code(c["code"]) for c in group]
cells.append(md("## Part 3: train the weights from scratch\nStarting from random weights, mini-batch gradient descent with hand-written "
                "backpropagation. The numbers differ from run to run; accuracy normally climbs to 0.9–1.0. The page's Train mode does the "
                "same thing interactively and also offers SGD + momentum and Adam."))
cells.append(code(d["trainNp"]))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
NotebookClient(nb, timeout=300, kernel_name="python3").execute()
for c in nb.cells:  # drop TensorFlow's start-up log noise from the saved outputs
    if c.cell_type == "code":
        c.outputs = [o for o in c.outputs if o.get("name") != "stderr"]
errors = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error")
nbf.write(nb, HERE.parent / "notebooks" / "tiny_cnn_walkthrough.ipynb")
print(f"notebook written: {len(nb.cells)} cells, {errors} errors")

"""The pretrained MobileNetV2 weights, verified before use.

Keras downloads these files from Google's keras-applications bucket without a checksum. This puts
them in the Keras cache first, with the SHA-256 they had when this walkthrough was built: a file
that doesn't match is downloaded again, and if it still doesn't match, Keras stops with an error.
The files are HDF5 weights (arrays only, no code), Apache 2.0, trained on ImageNet.

Works with Keras 3 and with tf_keras (tools/export_tfjs.py): pass the module in use.
"""

BASE = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
WEIGHTS = {
    # include_top=False (the base of model B)
    "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5":
        "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c",
    # include_top=True (the original 1000-class model)
    "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5":
        "3e195a2857356cfc092cbbb460beb2a5bce279015d7792598b8d3d9e451902e3",
}


def verify(keras):
    """Downloads (or checks the cached copy of) both weight files against their pinned SHA-256."""
    for name, sha in WEIGHTS.items():
        keras.utils.get_file(name, BASE + name, file_hash=sha, hash_algorithm="sha256", cache_subdir="models")

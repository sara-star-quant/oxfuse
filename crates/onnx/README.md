# oxfuse-onnx

ONNX import for [oxfuse](https://github.com/sara-star-quant/oxfuse): reads a float32 model made of a small operator subset and lowers it into the oxfuse IR. Anything outside the subset fails with an error that names the node and the reason.

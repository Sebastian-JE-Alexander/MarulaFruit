"""
-------------------------- model_export.py --------------------------------
Exports the trained pytorch model to ONNX format for deployment outside
PyTorch (e.g. TensorRT on a Jetson, ONNX Runtime on any platform).

"dummy input": torch.onnx.export() doesn't translate the Python code directly
Rather it runs one real forward pass through the model with a sample tensor
and traces every operation that actually executes during that pass to build
the ONNX computation graph.

The dummy input's values are irrelevant (random numbers are fine),
the shape and DTYPE are what matters most. These determine
which code path gets traced (which layer sizes, which branches).
Get the shape wrong, and you can get a subtly wrong graph, which is an error
you wouldn't immediately notice.

To ensure against that, this script also verifies against export against
the original pytorch model afterwards rather than assuming it was correct.

Usage: python model_export.py
--------------------------------------------------------------------------
"""

import numpy as np
import onnxruntime as ort
import torch

import config
from detect_and_classify import load_model


def export_to_onnx(output_path=config.ONNX_MODEL_PATH):
    model, device, classes = load_model()
    model.eval()  # disables dropout - required before export, otherwise dropout's
    # random behaviour gets traced into the graph as if it were
    # part of the model's actual computation

    # Shape must exactly match what the real model expects at inference:
    # (batch, channels, height, width) = (1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE).
    # Batch size 1 here, but dynamic_axes below still allows any batch
    # size at actual inference time - this is just what gets traced.
    dummy_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,
                              device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        # Names for the graph's input/output tensors - arbitrary strings,
        # but whatever loads this .onnx file later needs to reference
        # them by these exact names (see benchmark_onnx.py and
        # detect_and_classify.py's ONNX branch, both use "input").
        input_names=["input"],
        output_names=["output"],
        # Marks the batch dimension (index 0) as variable rather than
        # fixed at 1 - lets the exported model accept any batch size at
        # actual inference time, even though only a single dummy image
        # was used for tracing. Every other dimension (channels, height,
        # width) stays fixed, since those don't change between calls.
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,  # widely supported by ONNX Runtime and TensorRT
    )
    print(f"Exported ONNX model -> {output_path}")

    # Saved to a SEPARATE file from the PyTorch classes.txt
    # (config.CLASSES_PATH) rather than reused directly - keeps the ONNX
    # artifact self-contained, so anything loading it later (a different
    # machine, a non-Python deployment, TensorRT tooling) doesn't need to
    # know anything about the PyTorch side of this project at all, just
    # this one .onnx file plus this one small text file.
    with open(config.ONNX_CLASSES_PATH, "w") as f:
        f.write("\n".join(classes))
    print(f"Saved class order -> {config.ONNX_CLASSES_PATH} (classes = {classes})")
    print("IMPORTANT: the ONNX model outputs raw scores in this class order - "
          "whatever loads this model elsewhere needs to know this order to "
          "interpret index 0 / index 1 correctly.")

    return output_path, classes


def verify_export(onnx_path=config.ONNX_MODEL_PATH, n_test_cases=5):
    """Loads the exported ONNX model back and compares its output against
    the original PyTorch model on several random inputs - proves the
    export is numerically correct, not just that it didn't crash. Small
    floating-point differences (< 1e-4 or so) are normal and expected
    between PyTorch and ONNX Runtime's execution; anything larger is
    worth investigating before trusting this export.

    Multiple random test cases (not just one) - a single lucky/unlucky
    input could pass or fail by chance if there's a subtle, input-
    dependent bug (e.g. an op that only misbehaves near certain value
    ranges). A few different random inputs make that far less likely to
    slip through undetected.
    """
    model, device, classes = load_model()
    model.eval()

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_diff_overall = 0.0
    for i in range(n_test_cases):
        # Created directly on the model's device (matches
        # export_to_onnx()'s dummy_input) - plain torch.randn() with no
        # device specified always defaults to CPU, which would mismatch
        # a GPU-loaded model exactly the way it did before this fix.
        test_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,
                                 device=device)

        with torch.no_grad():
            # .cpu() before .numpy() - if the model is on a GPU, its
            # output tensor is too, and numpy() only works on CPU tensors.
            torch_output = model(test_input).cpu().numpy()

        # ONNX Runtime's session.run() always wants a plain CPU numpy
        # array regardless of what device PyTorch used - .cpu() here is
        # a no-op if test_input was already on CPU, and required if it
        # was moved to a GPU above.
        onnx_output = session.run(None, {input_name: test_input.cpu().numpy()})[0]

        max_diff = np.abs(torch_output - onnx_output).max()
        max_diff_overall = max(max_diff_overall, max_diff)
        print(f"  Test case {i + 1}: max difference = {max_diff:.8f}")

    print(f"\nLargest difference across {n_test_cases} test cases: {max_diff_overall:.8f}")
    if max_diff_overall < 1e-4:
        print("PASS - ONNX export matches the PyTorch model.")
    else:
        print("WARNING - difference larger than expected. Do not trust this "
              "export until this is investigated (check opset_version, or "
              "whether the model has any op ONNX handles differently).")
    return max_diff_overall


if __name__ == "__main__":
    # Always verify right after exporting, in the same run - catches a
    # bad export immediately, rather than only discovering a problem
    # later when something downstream (camera_gui.py with USE_ONNX=True,
    # benchmark_onnx.py) starts producing wrong predictions with no
    # obvious cause.
    onnx_path, classes = export_to_onnx()
    print("\nVerifying export against the original PyTorch model...")
    verify_export(onnx_path)
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
    # disables dropout, which is required before export
    # Otherwise dropout's random behaviour gets traced into the graph as if it were
    # part of the models actual computation.
    model.eval()

    # Shape must exactly match what the real model expects at inference:
    # (batch, channels, height, width)
    # Batch size 1 here, but dynamic_axes below still allows any batch size
    # at actual inference time - this is just what gets traced.
    dummy_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)


    torch.onnx.export(
        model,
        dummy_input,
        output_path,

        # Names for the graphs input/output tensors
        # These are arbitrary strings, but whatever loads this
        # ONNX file later needs to reference them by these exact names
        # (see benchmark_onnx.py and detect_and_classify.py ONNX branch, both use "input").
        input_names=["input"],
        output_names=["output"],

        # Marks the batch dimension (index 0) as variable rather than fixed at 1
        # lets the exported model accept any batch size at actual inference time,
        # even though only a single dummy image was used for tracing.
        # Every other dimension stays fixed since those don't change between calls.
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,   # supported by ONNX Runtime and TensorRT for deployment modes
    )
    print(f"Exported ONNX model to {output_path}")

    # Saved to a separate file from the pytorch classes.txt rather than reused directly.
    # Keeps ONNX self-contained, so anything loading it later doesn't need to know anything
    # about the pytorch side of the project at all, it will just use the one .onnx file and
    # .txt file
    with open(config.ONNX_CLASSES_PATH, "w") as f:
        f.write("\n".join(classes))

    print(f"Saved class order to {config.ONNX_CLASSES_PATH} (classes = {classes})")
    print("IMPORTANT: the ONNX model outputs raw scores in this class order "
          "whatever loads this model needs to know this order to"
          "interpret index 0/ index 1 correctly.")

    return output_path, classes

def verify_export(onnx_path=config.ONNX_MODEL_PATH, n_test_cases=5):
    """
    Loads the exported ONNX model and compares its output against the original
    pytorch model on several random inputs to prove the export was numerically correct.
    Small floating-point differences (< 1e-4) are normal and expected between pytorch
    and ONNX Runtime. Anything larger requires further investigation before you can trust
    the ONNX export.

    Multiple random test cases
    a single lucky/unlucky input could pass or fail by change if there's
    a subtle, input dependant bug. A few different random inputs make that
    far less likely to slip through undetected.
    """
    model, device, classes = load_model()
    model.eval()

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    max_diff_overall = 0.0
    for i in range(n_test_cases):
        test_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,)

        with torch.no_grad():
            torch_output = model(test_input).numpy()

        onnx_output = session.run(None, {input_name: test_input.numpy()})[0]

        max_diff = np.abs(torch_output - onnx_output).max()
        max_diff_overall = max(max_diff_overall, max_diff)
        print(f" Test case {i+1}: max difference = {max_diff:.8f}")

    print(f"\nLargest difference across {n_test_cases} test cases is {max_diff_overall:.8f}")
    if max_diff_overall < 1e-4:
        print("PASS - ONNX export matches the pytorch model.")
    else:
        print("WARNING - ONNX export does not match the pytorch model."
              "Difference larger than 1e-4, investigate ONNX export.")
    return max_diff_overall

if __name__ == "__main__":
    # Always verify right after exporting, in the same run.
    # Catches a bad export immediately, rather than discovering a problem
    # later when something downstream ( GUI with USE_ONNX=TRUE, benchmark_onnx.py)
    # starts producing wrong predications with no obvious cause.
    onnx_path, classes = export_to_onnx()
    print("\nVerifying exported ONNX model against the original pytorch model...")
    verify_export(onnx_path)
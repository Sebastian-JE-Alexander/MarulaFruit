"""
-------------------------- model_export.py --------------------------------
Exports the pretrained model into ONNX (Open Neural Network Exchange) format
so that we don't deploy the entire model onto our edge device.
This means we are pulling our model out of our dynamic python training
environment and into a stable production environment.

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
    model.eval()  #disables dropout, required before export otherwise its random behaviour gets into the ONNX graph.

    dummy_input = torch.randn(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=device)


    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,   # supported by ONNX Runtime and TensorRT
    )
    print(f"Exported ONNX model to {output_path}")

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
    and ONNX Runtime.
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
    onnx_path, classes = export_to_onnx()
    print("\nVerifying exported ONNX model against the original pytorch model...")
    verify_export(onnx_path)
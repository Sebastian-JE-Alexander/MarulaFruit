"""
--------------------------- benchmark_onnx.py ----------------------------
Compares PyTorch vs ONNX Runtime inference time on the same real
images, using IDENTICAL preprocessing for both (build_transform() -
same function the normal pipeline uses) so the timing comparison is
real. Only the "run the model" step actually differs
between the two everything else is shared.

Usage: python benchmark_onnx.py path/to/image.png [path/to/another.png ...]
----------------------------------------------------------------------------
"""

import argparse
import time

import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

import config
from detect_and_classify import load_model
from dataset import build_transform


def load_onnx_session(onnx_path=config.ONNX_MODEL_PATH):
    """
    starts the ONNX session using ONNX runtime and points to the model files location.
    """
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    return session, input_name


def benchmark_single_image(image_path, model, device, session, input_name, classes, n_repeats=20):
    img = Image.open(image_path)
    x = build_transform(augment=False)(img).unsqueeze(0)
    x_torch = x.to(device)
    x_np = x.numpy()

    # Warmup each separately (excluded from timing) - first call includes
    # one-time cost (lazy init, kernel compilation) that isn't
    # representative of steady-state speed, same reasoning as infer.py's
    # first-call handling.
    with torch.no_grad():
        model(x_torch)
    session.run(None, {input_name: x_np})

    torch_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        with torch.no_grad():
            logits = model(x_torch)
            probs = F.softmax(logits, dim=1)[0]
        torch_times.append((time.perf_counter() - start) * 1000)
    torch_pred = classes[int(probs.argmax())]
    torch_conf = float(probs.max())

    onnx_times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        onnx_logits = session.run(None, {input_name: x_np})[0]
        onnx_times.append((time.perf_counter() - start) * 1000)
    onnx_probs = torch.softmax(torch.tensor(onnx_logits), dim=1)[0]
    onnx_pred = classes[int(onnx_probs.argmax())]
    onnx_conf = float(onnx_probs.max())

    return {
        "image": image_path,
        "torch_pred": torch_pred, "torch_conf": torch_conf,
        "torch_avg_ms": sum(torch_times) / len(torch_times), "torch_min_ms": min(torch_times),
        "onnx_pred": onnx_pred, "onnx_conf": onnx_conf,
        "onnx_avg_ms": sum(onnx_times) / len(onnx_times), "onnx_min_ms": min(onnx_times),
    }


def main(image_paths, n_repeats=20):
    model, device, classes = load_model()
    model.eval()
    session, input_name = load_onnx_session()

    print(f"Benchmarking {len(image_paths)} image(s), {n_repeats} repeats each "
          f"(warmup call excluded from timing)\n")

    results = []
    for path in image_paths:
        r = benchmark_single_image(path, model, device, session, input_name, classes, n_repeats)
        results.append(r)
        match = "same prediction" if r["torch_pred"] == r["onnx_pred"] else "MISMATCH - investigate"
        print(f"{path}")
        print(f"  PyTorch: {r['torch_pred']} ({r['torch_conf']:.1%})  "
              f"avg={r['torch_avg_ms']:.3f}ms  min={r['torch_min_ms']:.3f}ms")
        print(f"  ONNX:    {r['onnx_pred']} ({r['onnx_conf']:.1%})  "
              f"avg={r['onnx_avg_ms']:.3f}ms  min={r['onnx_min_ms']:.3f}ms  [{match}]")
        print()

    avg_torch = sum(r["torch_avg_ms"] for r in results) / len(results)
    avg_onnx = sum(r["onnx_avg_ms"] for r in results) / len(results)
    ratio = avg_torch / avg_onnx if avg_onnx else float("nan")
    print(f"Overall average: PyTorch {avg_torch:.3f}ms vs ONNX {avg_onnx:.3f}ms "
          f"({ratio:.2f}x {'faster' if ratio > 1 else 'slower'} with ONNX)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--n_repeats", type=int, default=20)
    args = parser.parse_args()
    main(args.images, args.n_repeats)
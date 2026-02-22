"""Export Cellpose-SAM (cpsam) model to ONNX.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --output models/cellpose-cpsam/model.onnx

The exported model:
  Input:  pixel_values  float32  (1, 3, 256, 256) — one tile at a time
  Output: output        float32  (1, 3, 256, 256) — (dY, dX, cellprob) flows

For large models (>1GB), PyTorch saves weights as external data next to the
.onnx file (model.onnx.data). ONNX Runtime loads this automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def export(output: Path) -> None:
    # Ensure cellpose is importable from a sibling clone
    cellpose_src = Path(__file__).parent.parent.parent / "cellpose"
    if cellpose_src.exists() and str(cellpose_src) not in sys.path:
        sys.path.insert(0, str(cellpose_src))

    try:
        from cellpose.vit_sam import Transformer
        from cellpose.models import cache_CPSAM_model_path
    except ImportError as e:
        raise SystemExit(
            f"Cannot import cellpose: {e}\n"
            "Make sure you have cloned the cellpose repo at ../cellpose\n"
            "or install it with:  pip install cellpose"
        ) from e

    print("Loading cpsam weights (downloads if not cached)...")
    model_path = cache_CPSAM_model_path()
    model = Transformer(dtype=torch.float32)
    model.load_model(model_path, device=torch.device("cpu"), strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {n_params:,} parameters.")

    dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output} ...")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["pixel_values"],
        output_names=["output"],
        opset_version=14,
        do_constant_folding=False,
    )

    onnx_size = output.stat().st_size / 1024**2
    data_path = output.with_name(f"{output.name}.data")
    if data_path.exists():
        data_size = data_path.stat().st_size / 1024**2
        print(f"Saved: {output} ({onnx_size:.1f} MB) + {data_path.name} ({data_size:.1f} MB)")
    else:
        print(f"Saved: {output} ({onnx_size:.1f} MB)")

    # Sanity check
    print("Running sanity check with onnxruntime...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy.numpy()})
        print(f"Output shape: {out[0].shape}")
        assert out[0].shape == (1, 3, 256, 256), f"Unexpected shape: {out[0].shape}"
        print("Sanity check passed.")
    except ImportError:
        print("onnxruntime not available, skipping sanity check.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/cellpose-cpsam/model.onnx"),
        help="Output ONNX file path",
    )
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()

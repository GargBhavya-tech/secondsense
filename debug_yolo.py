"""
Offline YOLO bug-hunting tool — no phone, no rebuild, no install.

Runs the EXACT same yolov11_det.tflite the app bundles, on a still image, and prints
the raw tensor shapes/ranges + every decoded box BEFORE any coordinate-space
assumption is applied. This is the fastest way to answer "is the box tensor
normalized 0..1 or model-pixels 0..640?" and "which output is scores vs classes?" —
the two most likely causes of a tiny/wrong box with a wrong label.

USAGE:
    venv\\Scripts\\python.exe debug_yolo.py path\\to\\bottle.jpg

If you don't have a laptop photo of the object handy, take one on your phone,
AirDrop/transfer/USB it to the laptop (or just download any bottle photo off the
web), and point this script at it. Add --webcam to grab a live frame from a
laptop webcam instead of a file.
"""
import sys
import argparse
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = "export_assets/tflite_models/yolov11_det/yolov11_det.tflite"


def letterbox(img: Image.Image, size: int):
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to a test image (jpg/png)")
    ap.add_argument("--webcam", action="store_true", help="grab a frame from the laptop webcam instead")
    ap.add_argument("--conf", type=float, default=0.15, help="score threshold (lower = show more, default 0.15)")
    args = ap.parse_args()

    if args.webcam:
        import cv2
        cam = cv2.VideoCapture(0)
        ok, frame = cam.read()
        cam.release()
        if not ok:
            print("Could not read from webcam."); sys.exit(1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    elif args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        print("Pass an image path or --webcam. See the docstring at the top of this file.")
        sys.exit(1)

    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    size = in_detail["shape"][1]  # assume square NHWC
    print(f"model input: {in_detail['shape']} dtype={in_detail['dtype']}")
    for i, d in enumerate(out_details):
        print(f"model output[{i}]: {d['shape']} dtype={d['dtype']}")

    lb_img, scale, pad_x, pad_y, orig_w, orig_h = letterbox(img, size)
    arr = (np.asarray(lb_img).astype(np.float32) / 255.0)[None, ...]

    interp.set_tensor(in_detail["index"], arr)
    interp.invoke()
    outs = [interp.get_tensor(d["index"]) for d in out_details]

    print("\n--- RAW OUTPUT STATS (before any coordinate-space assumption) ---")
    for i, o in enumerate(outs):
        flat = o.astype(np.float32).flatten()
        print(f"out[{i}] shape={o.shape} dtype={o.dtype} "
              f"min={flat.min():.3f} max={flat.max():.3f} mean={flat.mean():.3f}")

    # Identify boxes tensor (rank 3, last dim 4)
    boxes_t = next((o for o in outs if o.ndim == 3 and o.shape[-1] == 4), None)
    if boxes_t is None:
        print("\nCould not find a [1,N,4] boxes tensor — layout may be the single-head [1,84,N] form.")
        return

    n = boxes_t.shape[1]
    others = [o for o in outs if o is not boxes_t and o.size == n]
    scores_t = next((o for o in others if 0.0 <= o.min() and o.max() <= 1.0), others[0] if others else None)
    classes_t = next((o for o in others if o is not scores_t), None)

    print(f"\nboxes tensor:   shape={boxes_t.shape}  sample[0]={boxes_t[0,0].tolist()}")
    if scores_t is not None:
        print(f"scores tensor:  shape={scores_t.shape}  sample[0:5]={scores_t.flatten()[:5].tolist()}")
    if classes_t is not None:
        print(f"classes tensor: shape={classes_t.shape}  sample[0:5]={classes_t.flatten()[:5].tolist()}")

    # THE KEY QUESTION: are box values in 0..1 (already normalized) or 0..size (pixels)?
    box_max = boxes_t.max()
    print(f"\n>>> box value range: 0..{box_max:.3f}  (model input size = {size})")
    if box_max <= 1.5:
        print(">>> LOOKS NORMALIZED (0..1) — the app's YoloDecoder currently assumes MODEL-PIXEL")
        print(">>> coords and divides by `scale` after subtracting padding. THIS IS LIKELY THE BUG:")
        print(">>> treating a ~0.5 value as if it were 0.5 pixels (out of 640) collapses every box")
        print(">>> to a sliver near the top-left corner — matches 'small box somewhere'.")
    else:
        print(">>> LOOKS LIKE MODEL-PIXEL coords (0..~640) — matches the app's current assumption.")

    print("\n--- TOP DETECTIONS BY SCORE (raw, no NMS) ---")
    if scores_t is not None:
        flat_scores = scores_t.flatten()
        order = np.argsort(-flat_scores)[:10]
        for idx in order:
            s = flat_scores[idx]
            if s < args.conf:
                continue
            b = boxes_t[0, idx]
            c = int(classes_t.flatten()[idx]) if classes_t is not None else -1
            print(f"  idx={idx:5d}  score={s:.3f}  class_idx={c}  box_raw={b.tolist()}")


if __name__ == "__main__":
    main()

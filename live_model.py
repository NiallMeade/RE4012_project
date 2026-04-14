import time
import cv2
import numpy as np
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/yolo26n_int8_256.tflite"
CONF_THRESH = 0.35
INPUT_SIZE  = 256
# ──────────────────────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# ─── MODEL ────────────────────────────────────────────────────────────────────
def load_model(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def preprocess(frame, size, inp_detail):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dtype = inp_detail['dtype']

    if dtype == np.float32:
        print("In float32")
        # float32 and float16 models (float16 is upcast to float32 by TFLite)
        img = (img / 255.0).astype(np.float32)

    elif dtype in (np.int8, np.uint8):
        print("In int8")
        # int8 / uint8 quantized models — apply input quantization
        scale, zero_point = inp_detail['quantization']
        if scale > 0:
            img = (img / 255.0 / scale + zero_point)
        img = np.clip(img, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

    return np.expand_dims(img, axis=0)

def postprocess(output, out_detail, orig_h, orig_w, conf_thresh):
    # ─── Dequantize if needed ─────────────────
    scale, zero_point = out_detail['quantization']
    if scale > 0:
        output = (output.astype(np.float32) - zero_point) * scale
    else:
        output = output.astype(np.float32)

    detections = output[0]  # shape: (300, 6)

    results = []

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det

        if conf < conf_thresh:
            continue

        # Clamp normalized coords
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))

        # Scale to pixel coords
        x1 = int(x1 * orig_w)
        y1 = int(y1 * orig_h)
        x2 = int(x2 * orig_w)
        y2 = int(y2 * orig_h)

        if x2 <= x1 or y2 <= y1:
            continue

        results.append((x1, y1, x2, y2, float(conf), int(cls_id)))

    return results

def draw_results(frame, boxes):
    for x1, y1, x2, y2, conf, cls_id in boxes:
        label = f"{COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else cls_id} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)

        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

# ─── INIT ─────────────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
interp = load_model(MODEL_PATH)
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print(f"Input shape: {inp_detail['shape']} dtype: {inp_detail['dtype']}")
print(f"Output shape: {out_detail['shape']} dtype: {out_detail['dtype']}")
print(f"Quantization: {out_detail['quantization']}")

# ─── CAMERA ───────────────────────────────────────────────────────────────────
print("[INFO] Starting camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}
))
picam2.start()
time.sleep(1)

# ─── FPS ──────────────────────────────────────────────────────────────────────
frame_count = 0
start_time = time.time()
fps = 0.0

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    orig_h, orig_w = frame.shape[:2]

    inp_data = preprocess(frame, INPUT_SIZE, inp_detail)

    t0 = time.perf_counter()
    interp.set_tensor(inp_detail['index'], inp_data)
    interp.invoke()
    output = interp.get_tensor(out_detail['index'])
    infer_ms = (time.perf_counter() - t0) * 1000

    boxes = postprocess(output, out_detail, orig_h, orig_w, CONF_THRESH)
    frame = draw_results(frame, boxes)

    # FPS calc
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}  Infer: {infer_ms:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO TFLite (Fixed)", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()

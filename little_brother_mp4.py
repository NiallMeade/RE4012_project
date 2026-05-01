import time
import cv2
import numpy as np
import pickle
import face_recognition

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/yolo26n_float32_320.tflite"
ENCODINGS_PATH  = "encodings.pickle"
VIDEO_PATH      = "capture_320x320.mp4"
OUTPUT_PATH     = "output.mp4"

CONF_THRESH     = 0.35
INPUT_SIZE      = 320
FACE_EVERY_N    = 5
FACE_SCALE      = 0.5
IOU_THRESH      = 0.3
PERSON_CLASS_ID = 0
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

def preprocess(frame, size, inp_detail):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dtype = inp_detail['dtype']
    if dtype == np.float32:
        img = (img / 255.0).astype(np.float32)
    elif dtype in (np.int8, np.uint8):
        scale, zero_point = inp_detail['quantization']
        if scale > 0:
            img = (img / 255.0 / scale + zero_point)
        img = np.clip(img, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return np.expand_dims(img, axis=0)

def postprocess(output, out_detail, orig_h, orig_w, conf_thresh):
    scale, zero_point = out_detail['quantization']
    if scale > 0:
        output = (output.astype(np.float32) - zero_point) * scale
    else:
        output = output.astype(np.float32)

    results = []
    for det in output[0]:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_thresh:
            continue
        x1 = int(max(0, min(1, x1)) * orig_w)
        y1 = int(max(0, min(1, y1)) * orig_h)
        x2 = int(max(0, min(1, x2)) * orig_w)
        y2 = int(max(0, min(1, y2)) * orig_h)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append((x1, y1, x2, y2, float(conf), int(cls_id)))
    return results

def recognize_in_crop(crop_bgr, known_encodings, known_names):
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    small = cv2.resize(crop_bgr, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        return None
    enc = encodings[0]
    distances = face_recognition.face_distance(known_encodings, enc)
    best_idx  = int(np.argmin(distances))
    matches = face_recognition.compare_faces(known_encodings, enc)
    if matches[best_idx]:
        return known_names[best_idx]
    return "Unknown"

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

def match_names_to_boxes(current_person_boxes, cached_labels):
    names = []
    for box in current_person_boxes:
        best_name = None
        best_iou  = IOU_THRESH
        for cached_box, cached_name in cached_labels:
            iou = compute_iou(box, cached_box)
            if iou > best_iou:
                best_iou  = iou
                best_name = cached_name
        names.append(best_name)
    return names

def draw_detections(frame, boxes, person_names):
    person_idx = 0
    for x1, y1, x2, y2, conf, cls_id in boxes:
        if cls_id == PERSON_CLASS_ID:
            name = person_names[person_idx] if person_idx < len(person_names) else None
            person_idx += 1
            color = (0, 200, 255) if (name and name != "Unknown") else (0, 255, 255)
            label = name if name else f"person {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = f"{COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else cls_id} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# ─── INIT ─────────────────────────────────────────────────────────────────────
print("[INFO] Loading face encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    enc_data = pickle.loads(f.read())
known_face_encodings = enc_data["encodings"]
known_face_names     = enc_data["names"]

print("[INFO] Loading YOLO model...")
interp = Interpreter(model_path=MODEL_PATH, num_threads=4)
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print("[INFO] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video file")

fps_input = cap.get(cv2.CAP_PROP_FPS)
width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ─── STATE ────────────────────────────────────────────────────────────────────
frame_count   = 0
start_time    = time.time()
fps           = 0.0
total_frames  = 0
cached_labels = []
processed_frames = []   # ← collect annotated frames here

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    t0       = time.perf_counter()
    inp_data = preprocess(frame, INPUT_SIZE, inp_detail)
    interp.set_tensor(inp_detail['index'], inp_data)
    interp.invoke()
    output   = interp.get_tensor(out_detail['index'])

    all_boxes    = postprocess(output, out_detail, orig_h, orig_w, CONF_THRESH)
    person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, cls in all_boxes
                    if cls == PERSON_CLASS_ID]

    if total_frames % FACE_EVERY_N == 0:
        new_labels = []
        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[y1:y2, x1:x2]
            name = recognize_in_crop(crop, known_face_encodings, known_face_names)
            new_labels.append(((x1, y1, x2, y2), name))
        cached_labels = new_labels

    person_names = match_names_to_boxes(person_boxes, cached_labels)
    infer_ms     = (time.perf_counter() - t0) * 1000

    frame = draw_detections(frame, all_boxes, person_names)

    frame_count  += 1
    total_frames += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps        = frame_count / elapsed
        frame_count = 0
        start_time  = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}  YOLO: {infer_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLO + Face Recognition (Video)", frame)

    processed_frames.append(frame)  # ← store, don't write yet

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ─── POST-RUN SAVE ────────────────────────────────────────────────────────────
if processed_frames:
    print(f"[INFO] Saving {len(processed_frames)} frames to {OUTPUT_PATH}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))
    for f in processed_frames:
        out.write(f)
    out.release()
    print("[INFO] Done.")

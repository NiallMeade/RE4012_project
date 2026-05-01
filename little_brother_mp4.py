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
MODEL_PATH      = "models/yolo26n_float32_480.tflite"
ENCODINGS_PATH  = "encodings.pickle"
VIDEO_PATH      = "input.mp4"   # ← your video file here
OUTPUT_PATH     = "output.mp4"  # optional output video

CONF_THRESH     = 0.35
INPUT_SIZE      = 480
FACE_EVERY_N    = 5
FACE_SCALE      = 0.5
IOU_THRESH      = 0.3
PERSON_CLASS_ID = 0
# ──────────────────────────────────────────────────────────────────────────────

# (Keep COCO_CLASSES, YOLO, face recognition, IoU, drawing functions EXACTLY the same)
# ──────────────────────────────────────────────────────────────────────────────
# ⬆️ NO CHANGES ABOVE THIS LINE (reuse your existing functions)
# ──────────────────────────────────────────────────────────────────────────────

# ─── INIT ─────────────────────────────────────────────────────────────────────
print("[INFO] Loading face encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    enc_data = pickle.loads(f.read())
known_face_encodings = enc_data["encodings"]
known_face_names     = enc_data["names"]

print("[INFO] Loading YOLO model...")
interp     = Interpreter(model_path=MODEL_PATH, num_threads=4)
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print("[INFO] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video file")

# Get video properties
fps_input = cap.get(cv2.CAP_PROP_FPS)
width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

# ─── STATE ────────────────────────────────────────────────────────────────────
frame_count  = 0
start_time   = time.time()
fps          = 0.0
total_frames = 0
cached_labels = []

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    orig_h, orig_w = frame.shape[:2]

    # ── YOLO inference ────────────────────────────────────────────────────────
    t0       = time.perf_counter()
    inp_data = preprocess(frame, INPUT_SIZE, inp_detail)
    interp.set_tensor(inp_detail['index'], inp_data)
    interp.invoke()
    output    = interp.get_tensor(out_detail['index'])
    infer_ms  = (time.perf_counter() - t0) * 1000

    all_boxes    = postprocess(output, out_detail, orig_h, orig_w, CONF_THRESH)
    person_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, cls in all_boxes
                    if cls == PERSON_CLASS_ID]

    # ── Face recognition ──────────────────────────────────────────────────────
    if total_frames % FACE_EVERY_N == 0:
        new_labels = []
        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[y1:y2, x1:x2]
            name = recognize_in_crop(crop, known_face_encodings, known_face_names)
            new_labels.append(((x1, y1, x2, y2), name))
        cached_labels = new_labels

    person_names = match_names_to_boxes(person_boxes, cached_labels)

    # ── Draw ──────────────────────────────────────────────────────────────────
    frame = draw_detections(frame, all_boxes, person_names)

    # ── FPS overlay ───────────────────────────────────────────────────────────
    frame_count  += 1
    total_frames += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps        = frame_count / elapsed
        frame_count = 0
        start_time  = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}  YOLO: {infer_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLO + Face Recognition (Video)", frame)
    out.write(frame)  # save frame

    if cv2.waitKey(1) == ord("q"):
        break

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()

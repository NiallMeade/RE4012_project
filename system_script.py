import time
import json
import cv2
import numpy as np
import pickle
import face_recognition
import matplotlib
matplotlib.use("Agg")          # headless – no display needed on Pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

# Config
MODEL_PATH = "models/yolo26n_float32_480.tflite"
ENCODINGS_PATH = "encodings.pickle"
VIDEO_PATH = "capture_480x480.mp4"
OUTPUT_PATH = "output.mp4"
LOG_PATH = "detections.json"

# Scenario label – change this string for each lighting condition run so that
# the saved plot filenames and quality-report titles reflect the scenario.
SCENARIO_LABEL = "test4"

CONF_THRESH = 0.35
INPUT_SIZE = 320
FACE_EVERY_N = 5 # Run face recognition every N frames
cv_scalar = 2 # Downsample person crop before face recognition
IOU_THRESH = 0.3 # Minimum IoU to match a cached face name to a current detection
PERSON_CLASS_ID = 0 # COCO class index for "person"

# Quality-analysis thresholds
LOW_CONF_WARN_THRESH = 0.45   # avg confidence below this → warn
HIGH_UNKNOWN_RATIO = 0.30   # >30 % of person detections as Unknown → warn
MIN_DETECTION_FRAMES = 3      # entity seen for fewer frames → treat as noise

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
    """
    Run face recognition inside a single person crop.
    Returns a name string, or None if no face found.
    """
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None, None
    resized_frame = cv2.resize(crop_bgr, (0, 0), fx=1/cv_scalar, fy=1/cv_scalar)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame, model="hog")
    if not face_locations:
        return None, None
    
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)
    if not face_encodings:
        return None, None

    # Take the first (most prominent) face in the crop
    enc = face_encodings[0]
    face_distances = face_recognition.face_distance(known_encodings, enc)
    best_match_idx = int(np.argmin(face_distances))

    matches = face_recognition.compare_faces(known_encodings, enc)
    confidence = float(1.0 - face_distances[best_match_idx])
    if matches[best_match_idx]:
        return known_names[best_match_idx], confidence
    return "Unknown", confidence

def compute_iou(boxA, boxB):
    """Compute Intersection over Union for two (x1,y1,x2,y2) boxes."""
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
    """
    For each current person box, find the best matching cached label via IoU.
    cached_labels: list of ((x1,y1,x2,y2), name)
    Returns: list of name strings aligned with current_person_boxes.
    """
    names = []
    for box in current_person_boxes:
        best_name = None
        best_conf = None
        best_iou = IOU_THRESH # minimum threshold to accept a match
        for cached_box, cached_name, cached_conf in cached_labels:
            iou = compute_iou(box, cached_box)
            if iou > best_iou:
                best_iou = iou
                best_name = cached_name
                best_conf = cached_conf
        names.append((best_name, best_conf))
    return names


def draw_detections(frame, boxes, person_name_confs):
    """
    Draw all YOLO detections. Person boxes get a name label (if recognised),
    all other objects get the standard class label.
    """
    person_idx = 0
    for x1, y1, x2, y2, conf, cls_id in boxes:
        if cls_id == PERSON_CLASS_ID:
            name, face_conf = (person_name_confs[person_idx] if person_idx < len(person_name_confs) else (None, None))
            person_idx += 1

            color = (0, 200, 255) if (name and name != "Unknown") else (0, 255, 255)
            if name and face_conf is not None:
                label = f"{name} {face_conf:.0%}"
            else:
                label = f"person {conf:.2f}"
        else:
            color = (0, 255, 0)
            label = (f"{COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else cls_id}"
                     f" {conf:.2f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


# Post processing

def build_entity_timelines(log):
    """
    Returns a dict keyed by entity label:
      {
        label: {
            "timestamps":   [t, ...],      # seconds at which entity was seen
            "confidences":  [c, ...],      # matching confidence values
            "intervals":    [(t_enter, t_exit), ...]  # contiguous presence spans
        }
      }
    Persons are tracked by their recognised name (not just "person").
    Objects are tracked by COCO class label.
    """
    raw = defaultdict(lambda: {"timestamps": [], "confidences": []})

    for frame in log["frames"]:
        t = frame["timestamp_s"]
        seen_this_frame = set()
        for det in frame["detections"]:
            # For persons: use their name if known, else "Unknown_<box>"
            if det["type"] == "person":
                label = det["label"] if det["label"] not in (None, "person") else "Unknown"
            else:
                label = det["label"]

            if label in seen_this_frame:
                # Only suffix genuinely anonymous labels (Unknown).
                # Named individuals should never be duplicated in the same
                # frame – if the same name appears twice, keep the first hit.
                if label == "Unknown":
                    idx = sum(1 for k in seen_this_frame if k.startswith("Unknown"))
                    label = f"Unknown_{idx}"
                else:
                    # Named person already counted this frame – skip duplicate.
                    continue
            seen_this_frame.add(label)

            conf = det["confidence"] if det["confidence"] is not None else det["det_conf"]
            raw[label]["timestamps"].append(t)
            raw[label]["confidences"].append(conf)

    fps = log["metadata"]["fps"]
    gap_threshold = (FACE_EVERY_N + 1) / fps # gap > this to new interval

    entities = {}
    for label, data in raw.items():
        ts = data["timestamps"]
        confs = data["confidences"]
        if len(ts) < MIN_DETECTION_FRAMES:
            continue  # skip noise
        intervals = []
        seg_start = ts[0]
        prev_t = ts[0]
        for t in ts[1:]:
            if t - prev_t > gap_threshold:
                intervals.append((seg_start, prev_t))
                seg_start = t
            prev_t = t
        intervals.append((seg_start, prev_t))

        entities[label] = {
            "timestamps":  ts,
            "confidences": confs,
            "intervals":   intervals,
        }
    return entities


def plot_presence_timeline(entities, video_duration, scenario_label):
    """
    Gantt-style chart: one horizontal bar per entity showing when they were
    present. Separate panels for people vs objects.
    """
    people = {k: v for k, v in entities.items()
                if not any(c in k for c in COCO_CLASSES[1:])}
    objects = {k: v for k, v in entities.items() if k not in people}

    def _gantt(ax, subset, title, color_map):
        if not subset:
            ax.set_visible(False)
            return
        labels = sorted(subset.keys())
        for i, label in enumerate(labels):
            color = color_map.get(label, f"C{i}")
            for (t0, t1) in subset[label]["intervals"]:
                ax.barh(i, t1 - t0, left=t0, height=0.5,
                        color=color, edgecolor="white", linewidth=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, video_duration)
        ax.set_xlabel("Time (s)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    n_panels = (1 if not objects else 2)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(12, 3 + 1.2 * max(len(people), 1)
                                      + (1.2 * len(objects) if objects else 0)),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    person_colors = {k: f"C{i}" for i, k in enumerate(sorted(people.keys()))}
    _gantt(axes[0], people, f"People – {scenario_label}", person_colors)
    if objects:
        object_colors = {k: f"C{i+len(people)}" for i, k in enumerate(sorted(objects.keys()))}
        _gantt(axes[1], objects, f"Objects – {scenario_label}", object_colors)

    fig.suptitle(f"Presence Timeline  [{scenario_label}]", fontsize=13, fontweight="bold")
    fname = f"plot_timeline_{scenario_label}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Presence timeline saved → {fname}")


def plot_confidence_over_time(entities, video_duration, scenario_label):
    """
    One subplot per entity showing confidence vs time,
    with shaded bands for presence intervals.
    """
    if not entities:
        return

    n = len(entities)
    fig, axes = plt.subplots(n, 1,
                             figsize=(12, 2.2 * n),
                             constrained_layout=True,
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, sorted(entities.items())):
        ts    = data["timestamps"]
        confs = data["confidences"]

        # Shade the presence intervals
        for (t0, t1) in data["intervals"]:
            ax.axvspan(t0, t1, alpha=0.12, color="steelblue")

        ax.plot(ts, confs, "o-", markersize=3, linewidth=1.2, label=label)
        ax.axhline(LOW_CONF_WARN_THRESH, color="orange", linestyle="--",
                   linewidth=0.9, label=f"Warn threshold ({LOW_CONF_WARN_THRESH})")
        avg = np.mean(confs)
        ax.axhline(avg, color="green", linestyle=":", linewidth=0.9,
                   label=f"Mean = {avg:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Confidence")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(0, video_duration)
    fig.suptitle(f"Confidence Over Time  [{scenario_label}]", fontsize=13, fontweight="bold")
    fname = f"plot_confidence_{scenario_label}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confidence over time saved {fname}")


def analyse_quality(entities, log, scenario_label):
    """
    Prints a quality report and returns a dict of findings.
    Suggestions are drawn from ISP / image-sensor knowledge:
      - Low confidence  → blur, low-light noise, focus issues
      - High Unknown %  → face angle, low resolution, occlusion
      - Flickering       → temporal confidence variance
    """
    print(f"\n{'='*60}")
    print(f"QUALITY REPORT - {scenario_label}")
    print(f"{'='*60}")

    findings = {}
    total_person_dets = 0
    unknown_dets = 0
    all_confs = []

    for frame in log["frames"]:
        for det in frame["detections"]:
            if det["type"] == "person":
                total_person_dets += 1
                if det["label"] in (None, "person", "Unknown"):
                    unknown_dets += 1
            c = det["confidence"] if det["confidence"] is not None else det["det_conf"]
            all_confs.append(c)

    global_avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    unknown_ratio = unknown_dets / max(total_person_dets, 1)

    print(f"  Global mean confidence : {global_avg_conf:.3f}")
    print(f"  Unknown person ratio   : {unknown_ratio:.1%}  "
          f"({unknown_dets}/{total_person_dets} detections)")

    # Per-entity flicker (std-dev of confidence)
    flicker_entities = {}
    for label, data in entities.items():
        std = float(np.std(data["confidences"]))
        if std > 0.12:
            flicker_entities[label] = std

    issues = []
    suggestions = []

    if global_avg_conf < LOW_CONF_WARN_THRESH:
        issues.append(f"Low global confidence ({global_avg_conf:.2f} < {LOW_CONF_WARN_THRESH})")
        suggestions.append(
            "Confidence is low. Possible causes: insufficient scene illumination "
            "motion blur"
            " or camera defocus. Try increasing scene brightness, "
            "or adjusting camera focus."
        )

    if unknown_ratio > HIGH_UNKNOWN_RATIO:
        issues.append(f"High Unknown ratio ({unknown_ratio:.1%} > {HIGH_UNKNOWN_RATIO:.0%})")
        suggestions.append(
            "Many persons are unidentified. Likely causes: extreme face angle, "
            "partial occlusion, or the crop resolution being too small for the "
            "face_recognition HOG model. Try higher resolution face crop or more "
            "diverse face angles for training data."
        )

    if flicker_entities:
        ent_list = ", ".join(f"{k} (σ={v:.2f})" for k, v in flicker_entities.items())
        issues.append(f"Flickering confidence for: {ent_list}")
        suggestions.append(
            "Confidence flickering suggests intermittent occlusion or"
            "excessively lenient confidence threshold. Try increasing"
            " confidence threshold."
        )

    if not issues:
        print("No significant quality issues detected.")
    else:
        print("\nIssues detected:")
        for i, iss in enumerate(issues, 1):
            print(f"    {i}. {iss}")
        print("\nSuggestions:")
        for i, sug in enumerate(suggestions, 1):
            print(f"    {i}. {sug}")

    findings = {
        "global_avg_confidence": round(global_avg_conf, 4),
        "unknown_ratio":         round(unknown_ratio, 4),
        "flicker_entities":      {k: round(v, 4) for k, v in flicker_entities.items()},
        "issues":                issues,
        "suggestions":           suggestions,
    }
    print(f"{'='*60}\n")
    return findings


def save_analysis_json(entities, quality_findings, scenario_label):
    """Save a compact per-scenario analysis JSON alongside the main log."""
    out = {
        "scenario": scenario_label,
        "quality":  quality_findings,
        "entities": {}
    }
    for label, data in entities.items():
        out["entities"][label] = {
            "n_frames":        len(data["timestamps"]),
            "mean_confidence": round(float(np.mean(data["confidences"])), 4),
            "std_confidence":  round(float(np.std(data["confidences"])),  4),
            "intervals":       [(round(a, 3), round(b, 3))
                                for a, b in data["intervals"]],
        }
    fname = f"analysis_{scenario_label}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Analysis JSON saved → {fname}")


print("[INFO] Loading face encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    enc_data = pickle.loads(f.read())
known_face_encodings = enc_data["encodings"]
known_face_names = enc_data["names"]

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
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Video: {width}x{height} @ {fps_input:.1f}fps, {total_in} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

frame_count = 0
total_frames = 0
cached_labels = []
fps_arr = []

log = {
    "metadata": {
        "video":        VIDEO_PATH,
        "scenario":     SCENARIO_LABEL,
        "fps":          fps_input,
        "total_frames": total_in,
        "width":        width,
        "height":       height,
    },
    "frames": []
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    timestamp_s = total_frames / fps_input

    t_0 = time.perf_counter()
    inp_data = preprocess(frame, INPUT_SIZE, inp_detail)
    interp.set_tensor(inp_detail['index'], inp_data)
    interp.invoke()
    output = interp.get_tensor(out_detail['index'])

    all_boxes = postprocess(output, out_detail, orig_h, orig_w, CONF_THRESH)
    person_boxes = [(x1, y1, x2, y2)
                    for x1, y1, x2, y2, _, cls in all_boxes
                    if cls == PERSON_CLASS_ID]

    # Face recognition (every FACE_EVERY_N frames)
    if total_frames % FACE_EVERY_N == 0:
        new_labels = []
        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[y1:y2, x1:x2]
            name, face_conf = recognize_in_crop(
                crop, known_face_encodings, known_face_names)
            new_labels.append(((x1, y1, x2, y2), name, face_conf))
        cached_labels = new_labels

    person_name_confs = match_names_to_boxes(person_boxes, cached_labels)
    infer_ms = (time.perf_counter() - t_0) * 1000

    frame_detections = []
    person_idx = 0
    for x1, y1, x2, y2, conf, cls_id in all_boxes:
        if cls_id == PERSON_CLASS_ID:
            name, face_conf = (person_name_confs[person_idx]
                               if person_idx < len(person_name_confs) else (None, None))
            person_idx += 1
            # Use the actual recognised name, not just "person"
            frame_detections.append({
                "label":      name if name else "Unknown",
                "type":       "person",
                "confidence": round(face_conf, 4) if face_conf is not None else None,
                "det_conf":   round(conf, 4),
                "box":        [x1, y1, x2, y2],
            })
        else:
            label = (COCO_CLASSES[cls_id]
                     if cls_id < len(COCO_CLASSES) else str(cls_id))
            frame_detections.append({
                "label":      label,
                "type":       "object",
                "confidence": round(conf, 4),
                "det_conf":   round(conf, 4),
                "box":        [x1, y1, x2, y2],
            })

    log["frames"].append({
        "frame_idx":   total_frames,
        "timestamp_s": round(timestamp_s, 4),
        "detections":  frame_detections,
    })

    frame = draw_detections(frame, all_boxes, person_name_confs)
    fps_v = 1000 / infer_ms if infer_ms > 0 else 0.0
    fps_arr.append(fps_v)

    frame_count  += 1
    total_frames += 1
    print(f"[INFO] FPS: {fps_v:.1f}  INF: {infer_ms:.1f}ms  "
          f"Frame: {total_frames}/{total_in}")

    cv2.putText(frame, f"FPS: {fps_v:.1f}  INF: {infer_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(frame)

cap.release()
out.release()
print(f"[INFO] Video saved → {OUTPUT_PATH}")

log["metadata"]["total_frames_processed"] = total_frames

with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2)
print(f"[INFO] Detection log saved → {LOG_PATH}")

with open("fps_tracker.txt", "w") as f:
    for fps_val in fps_arr:
        f.write(f"{fps_val}\n")

# Analysis
print("\n[INFO] Running post-run analysis...")

video_duration = total_frames / fps_input

entities = build_entity_timelines(log)

plot_presence_timeline(entities, video_duration, SCENARIO_LABEL)
plot_confidence_over_time(entities, video_duration, SCENARIO_LABEL)

quality_findings = analyse_quality(entities, log, SCENARIO_LABEL)

save_analysis_json(entities, quality_findings, SCENARIO_LABEL)

print("[INFO] All done.")

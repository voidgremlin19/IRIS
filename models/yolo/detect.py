import os
import json
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"
# Lazily initialize YOLO model to prevent import errors during generic test suite runs
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

# ============================================================
# CLASS NORMALIZATION
# ============================================================

CLASS_MAPPING = {
    "person": "pedestrian",
    "traffic light": "traffic_light",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
    "rider": "cyclist",
    "stray_dog": "dog",
    "stray dog": "dog",
    "auto": "auto_rickshaw",
    "auto rickshaw": "auto_rickshaw",
    "stop sign": "stop_sign",
}

SUPPORTED_CLASSES = {
    "pedestrian",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "traffic_light",
    "bicycle",
    "auto_rickshaw",
    "tractor",
    "cyclist",
    "handcart",
    "cow",
    "dog",
    "animal_other",
    "infrastructure",
    "hazards",
    "stop_sign",
}

VRU_CLASSES = {"pedestrian", "motorcycle", "bicycle", "cyclist", "handcart"}

CLASS_THRESHOLDS = {
    "pedestrian": 0.20,
    "motorcycle": 0.20,
    "bicycle": 0.20,
    "car": 0.25,
    "truck": 0.25,
    "bus": 0.25,
    "traffic_light": 0.20,
    "auto_rickshaw": 0.25,
    "tractor": 0.25,
    "cyclist": 0.20,
    "handcart": 0.20,
    "cow": 0.20,
    "dog": 0.20,
    "animal_other": 0.20,
    "infrastructure": 0.20,
    "hazards": 0.20,
    "stop_sign": 0.20,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def normalize_class_name(name):
    return CLASS_MAPPING.get(name, name)


def get_position_zone(cx, frame_w):
    ratio = cx / frame_w

    if ratio < 0.33:
        return "left"
    elif ratio < 0.66:
        return "center"

    return "right"


def get_size_category(box_area, frame_area):
    ratio = box_area / frame_area

    if ratio < 0.05:
        return "small"
    elif ratio < 0.20:
        return "medium"

    return "large"


def is_large_box(area_ratio):
    return area_ratio > 0.25


def is_bottom_heavy(y2, frame_h):
    return y2 > frame_h * 0.85


def is_low_conf(conf):
    return conf < 0.65


def generate_qa_flags(x1, y1, x2, y2, w, h, conf, cls_name, area_ratio):
    flags = []

    # Edge cropped object
    if x1 <= 5 or y1 <= 5 or x2 >= (w - 5) or y2 >= (h - 5):
        flags.append("edge_crop")

    # Very large box
    if area_ratio > 0.25:
        flags.append("large_box")

    # Low confidence
    if conf < 0.60:
        flags.append("low_confidence")

    # Unsupported class
    if cls_name not in SUPPORTED_CLASSES:
        flags.append("unsupported_class")

    # Dashboard / ego vehicle false positive
    dashboard_fp = (
        is_large_box(area_ratio) and is_bottom_heavy(y2, h) and is_low_conf(conf)
    )

    if dashboard_fp:
        flags.append("possible_dashboard_fp")

    # Possible auto-rickshaw heuristic
    if cls_name == "truck" and area_ratio < 0.08:
        flags.append("possible_auto_rickshaw")

    return flags


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================


def detect(
    image_path,
    default_conf_threshold=0.50,
    output_dir="data/yolo_results",
    save=True,
    suppress_dashboard_fp=False,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    yolo_model = get_model()

    try:
        results = yolo_model.predict(image_path, conf=0.25, verbose=False)  # NOT 0.0
    except Exception as e:
        raise RuntimeError(f"Detection failed: {str(e)}")

    result = results[0]

    print("Raw detections:", len(result.boxes))

    h, w = result.orig_shape
    frame_area = h * w

    image_id = os.path.basename(image_path)

    detections = []

    for i, box in enumerate(result.boxes):
        raw_conf = float(box.conf[0])
        print(f"{yolo_model.names[int(box.cls[0])]} {raw_conf:.2f}")

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        raw_cls_name = yolo_model.names[cls_id]
        cls_name = normalize_class_name(raw_cls_name)

        threshold = CLASS_THRESHOLDS.get(cls_name, default_conf_threshold)
        if conf < threshold:
            continue

        cx = (x1 + x2) / 2
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        qa_flags = generate_qa_flags(x1, y1, x2, y2, w, h, conf, cls_name, area_ratio)

        if suppress_dashboard_fp and "possible_dashboard_fp" in qa_flags:
            continue

        detection = {
            "id": i + 1,
            "class": cls_name,
            "raw_class": raw_cls_name,
            "confidence": round(conf, 2),
            "bbox_pixels": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "bbox_normalized": {
                "x1": round(x1 / w, 4),
                "y1": round(y1 / h, 4),
                "x2": round(x2 / w, 4),
                "y2": round(y2 / h, 4),
            },
            "position_zone": get_position_zone(cx, w),
            "size_category": get_size_category(box_area, frame_area),
            "is_vru": cls_name in VRU_CLASSES,
            "is_supported_class": cls_name in SUPPORTED_CLASSES,
            "qa_flags": qa_flags,
        }

        detections.append(detection)

    summary_parts = []
    for d in detections:
        summary_parts.append(
            f"{d['class']} {d['position_zone']} ({d['size_category']})"
        )

    summary = ", ".join(summary_parts) if detections else "No objects"

    output = {
        "image_id": image_id,
        "frame_size": {"w": w, "h": h},
        "total_detections": len(detections),
        "detections": detections,
        "summary": summary,
    }

    if save:
        os.makedirs(output_dir, exist_ok=True)

        annotations_dir = os.path.join(BASE_DIR, "data", "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        base_name = image_id.rsplit(".", 1)[0]

        # Save JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        # Save visualization image
        image = cv2.imread(image_path)
        for d in detections:
            x1 = d["bbox_pixels"]["x1"]
            y1 = d["bbox_pixels"]["y1"]
            x2 = d["bbox_pixels"]["x2"]
            y2 = d["bbox_pixels"]["y2"]

            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        viz_path = os.path.join(annotations_dir, f"{base_name}_viz.jpg")
        cv2.imwrite(viz_path, image)

        print(f"Saved JSON: {json_path}")
        print(f"Saved Image: {viz_path}")

    return output


# ============================================================
# SIMPLE PIPELINE OUTPUT
# ============================================================


def detect_objects_simple(image_path):
    output = detect(image_path, save=False)
    objects = [d["class"] for d in output["detections"]]
    # Preserve order + remove duplicates
    return list(dict.fromkeys(objects))


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":

    IMG_FOLDER = r"D:\VsCode\cruvixai\scene-reasoning-autonomy\indian-road-intelligence\datasets\indian_road_dataset\images\test"

    image_extensions = (".jpg", ".jpeg", ".png")

    files = []

    if os.path.exists(IMG_FOLDER):
        for img in os.listdir(IMG_FOLDER):

            if not img.lower().endswith(image_extensions):
                continue

            # only use road images
            if "leftImg8bit" in img:
                files.append(img)

        files = sorted(files)

        print(f"\nFound {len(files)} images\n")

        for i, file in enumerate(files, 1):

            image_path = os.path.join(IMG_FOLDER, file)

            print(f"\n[{i}/{len(files)}] Processing: {file}")

            try:

                output = detect(image_path, save=True, suppress_dashboard_fp=False)

                print(f"Detections: {output['total_detections']}")

            except Exception as e:

                print(f"Error on {file}: {e}")

        print("\nDONE")
    else:
        print(f"IMG_FOLDER {IMG_FOLDER} does not exist. Skipping main execution block.")

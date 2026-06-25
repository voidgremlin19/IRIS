import os
from analyze import analyze

IMAGE_DIR = "data/yolo_results/sample_annotate_images"
YOLO_DIR = "data/yolo_results/split"
OUTPUT_DIR = "data/vlm_results"


def get_json_name(image_name):
    # remove _viz + extension
    base = image_name.replace("_viz", "")
    base = os.path.splitext(base)[0]
    return base + ".json"


def run_batch():

    images = os.listdir(IMAGE_DIR)

    for img in images:

        if not img.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(IMAGE_DIR, img)

        json_name = get_json_name(img)
        json_path = os.path.join(YOLO_DIR, json_name)

        if not os.path.exists(json_path):
            print(f"❌ JSON missing for {img}")
            continue

        print("\n==============================")
        print(f"Processing: {img}")

        analyze(
            image_path=image_path,
            yolo_json_path=json_path,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    run_batch()

import json
import os
import time
import argparse
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
import base64

# RELATIONSHIP LOGIC (NEW)

def bbox_iou(a, b):
    xA = max(a["x1"], b["x1"])
    yA = max(a["y1"], b["y1"])
    xB = min(a["x2"], b["x2"])
    yB = min(a["y2"], b["y2"])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (a["x2"]-a["x1"]) * (a["y2"]-a["y1"])
    areaB = (b["x2"]-b["x1"]) * (b["y2"]-b["y1"])

    return inter / (areaA + areaB - inter + 1e-6)

# Remove duplicate detections
def deduplicate_detections(yolo_data, iou_threshold=0.6):

    detections = yolo_data["detections"]
    keep = []

    for det in detections:
        duplicate = False
        for k in keep:
            if det["class"] == k["class"]:
                if bbox_iou(det["bbox_pixels"], k["bbox_pixels"]) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            keep.append(det)

    yolo_data["detections"] = keep
    return yolo_data


def attach_person_vehicle_relationships(yolo_data):

    persons = [d for d in yolo_data["detections"]
               if d["class"] == "person"]

    motorcycles = [d for d in yolo_data["detections"]
                   if d["class"] in ["motorcycle", "bicycle"]]

    for person in persons:
        for vehicle in motorcycles:

            iou = bbox_iou(
                person["bbox_pixels"],
                vehicle["bbox_pixels"]
            )

            # overlap threshold
            if iou > 0.15:
                person["relationship"] = "riding"
                person["vehicle_id"] = vehicle["id"]

    return yolo_data

def attach_ego_proximity(yolo_data, image_width=1280):

    center_x = image_width / 2

    for det in yolo_data["detections"]:
        box = det["bbox_pixels"]
        obj_center = (box["x1"] + box["x2"]) / 2

        distance_from_center = abs(obj_center - center_x)

        if distance_from_center < image_width * 0.2:
            det["ego_relevance"] = "high"
        elif distance_from_center < image_width * 0.4:
            det["ego_relevance"] = "medium"
        else:
            det["ego_relevance"] = "low"

    return yolo_data

# Load API Key
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter Client (Qwen)
# Wait: Only initialize OpenRouter client if API_KEY is available (to prevent crashes on import during test runs)
client = None
if API_KEY:
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

MODEL = "qwen/qwen2.5-vl-72b-instruct"


# Load Prompt
def load_prompt_template():
    prompt_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "scene_analysis.txt"
    )

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    return "Analyze this driving scene."


# Encode Image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Clean JSON Output
def parse_json_response(text):

    text = text.strip()

    if "```" in text:
        text = text.split("```")[1]

    start = text.find("{")
    end = text.rfind("}") + 1

    return json.loads(text[start:end])


# MAIN VLM FUNCTION
def analyze(image_path, yolo_json_path,
            output_dir="data/vlm_results",
            max_retries=3):

    global client, API_KEY
    if not API_KEY:
        API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found")
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    print("Loading image...")
    img_b64 = encode_image(image_path)

    print("Loading YOLO JSON...")
    with open(yolo_json_path) as f:
        yolo_data = json.load(f)
        yolo_data = deduplicate_detections(yolo_data)
        yolo_data = attach_person_vehicle_relationships(yolo_data)
        yolo_data = attach_ego_proximity(yolo_data)

    prompt_template = load_prompt_template()

    prompt = prompt_template.replace(
        "{yolo_json}",
        json.dumps(yolo_data, indent=2)
    )

    for attempt in range(max_retries):
        try:
            print(f"Calling Qwen VLM (Attempt {attempt+1})")

            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
            )

            result_text = response.choices[0].message.content
            result = parse_json_response(result_text)

            os.makedirs(output_dir, exist_ok=True)

            image_id = os.path.basename(image_path).split(".")[0]
            out_path = os.path.join(output_dir, f"{image_id}.json")

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print("✅ Saved:", out_path)
            return result

        except json.JSONDecodeError:
            print("Invalid JSON, retrying...")
            time.sleep(2)

        except Exception as e:
            print("Error:", e)
            time.sleep(3)

    print("❌ FAILED after retries")
    return None


# CLI RUN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo-json", required=True)
    parser.add_argument("--output", default="data/vlm_results")

    args = parser.parse_args()

    analyze(args.image, args.yolo_json, args.output)

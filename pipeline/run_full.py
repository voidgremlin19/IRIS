import argparse
import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm.decide import decide
#voice
from models.voice.speaker import speak_with_buffer


# NEW FUNCTION
def find_vlm_file(image_id):

    normal = f"data/vlm_results/{image_id}.json"
    viz = f"data/vlm_results/{image_id}_viz.json"

    if os.path.exists(normal):
        return normal

    if os.path.exists(viz):
        return viz

    return None


def run(image_path):

    image_id = os.path.basename(image_path).split(".")[0]

    vlm_json = find_vlm_file(image_id)

    if not vlm_json:
        print("Missing VLM file for:", image_id)
        return

    print("Running LLM decision...")
    result = decide(vlm_json)

    if not result:
        return

    print("VOICE OUTPUT...")
    speak_with_buffer(result["voice_message"])

    print("DONE:", image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    run(args.image)

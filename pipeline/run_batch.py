import os
import argparse
from run_full import run


def batch(folder):

    os.makedirs("data/llm_results", exist_ok=True)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    completed = set(os.listdir("data/llm_results"))

    for img in images:

        image_id = img.split(".")[0]
        json_name = image_id + ".json"

        # Skip already processed images
        if json_name in completed:
            print("Skipping:", img)
            continue

        print("\n======================")
        print("Processing:", img)

        run(os.path.join(folder, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    batch(args.scenario)

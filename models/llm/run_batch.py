import os
from decide import decide

VLM_DIR = "data/vlm_results"
OUTPUT_DIR = "data/llm_results"


def run_batch():

    files = os.listdir(VLM_DIR)

    for file in files:

        if not file.endswith(".json"):
            continue

        vlm_path = os.path.join(VLM_DIR, file)

        print("\n==============================")
        print(f"Processing: {file}")

        decide(
            vlm_json_path=vlm_path,
            output_dir=OUTPUT_DIR
        )


if __name__ == "__main__":
    run_batch()

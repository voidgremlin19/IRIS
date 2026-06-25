import json
import os
import time
import argparse
# pyrefly: ignore [missing-import]
from groq import Groq

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = None

def get_client():
    global client, api_key
    if client is None:
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
        client = Groq(api_key=api_key)
    return client

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = "Respond ONLY in valid JSON."

def load_prompt():
    path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "driving_decision.txt"
    )
    with open(path) as f:
        return f.read()

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n",1)[1]
        text = text.rsplit("```",1)[0]
    return json.loads(text)

def decide(vlm_json_path, output_dir="data/llm_results", retries=3):

    with open(vlm_json_path) as f:
        vlm_context = json.load(f)

    prompt = load_prompt().replace(
        "{vlm_context}",
        json.dumps(vlm_context, indent=2)
    )

    for attempt in range(retries):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt}
                ],
                temperature=0.2
            )

            result = parse_json(
                response.choices[0].message.content
            )

            # ✅ BUILD VOICE MESSAGE HERE
            result["voice_message"] = build_voice_message(result)

            os.makedirs(output_dir, exist_ok=True)

            name = os.path.basename(vlm_json_path)
            out_path = os.path.join(output_dir, name)

            with open(out_path,"w") as f:
                json.dump(result,f,indent=2)

            print("Saved:", out_path)

            return result

        except Exception as e:
            print("Retry:", e)
            time.sleep(2)

    return None

# voice message
def build_voice_message(result):

    plan = result["action_plan"]

    speed_action = plan["speed"]["action"]
    speed_value = plan["speed"]["target_kmh"]
    steering = plan["steering"]["action"]
    distance = plan["following_distance"]["action"]
    horn = plan["horn"]["action"]

    alerts = plan.get("alerts", [])
    reasoning = result.get("reasoning", "")

    message_parts = []

    # Speed instruction
    message_parts.append(
        f"{speed_action} speed to {speed_value} kilometers per hour."
    )

    # Steering
    message_parts.append(f"{steering}.")

    # Following distance
    message_parts.append(f"{distance} following distance.")

    # Horn
    if horn != "no horn":
        message_parts.append(f"{horn}.")

    # Alerts
    if alerts:
        message_parts.append("Alert: " + ", ".join(alerts) + ".")

    # Short explanation (VERY IMPORTANT)
    short_reason = reasoning.split(".")[0]
    message_parts.append(short_reason + ".")

    return " ".join(message_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-json", required=True)
    args = parser.parse_args()

    decide(args.vlm_json)

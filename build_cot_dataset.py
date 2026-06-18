import json, os, random

MANIFEST = r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\maram_groot_carla_frames 1\maram_groot_carla_frames\manifest.jsonl'
OUTPUT = r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model\data\danger_cot\cot_dataset.jsonl'

COT_TEMPLATES = {
    "red_light": [
        "The traffic light ahead is RED. I must stop before the line. Applying brake now.",
        "Red light detected. Decelerating to a full stop. Throttle off, brake engaged.",
        "Traffic signal is RED. Stopping the vehicle to comply with traffic rules.",
    ],
    "near_stop": [
        "A stop sign is nearby. Slowing down and preparing to stop completely.",
        "Stop sign detected ahead. Reducing speed gradually and coming to a halt.",
        "Approaching a stop zone. Braking carefully and checking surroundings.",
    ],
    "hard_brake": [
        "Sudden obstacle or hazard detected ahead. Emergency braking applied immediately.",
        "Unexpected event ahead. Hard brake engaged to avoid collision.",
        "Critical situation: rapid deceleration required. Brake applied with high force.",
    ],
    "traffic_jam": [
        "Traffic congestion detected. Switching to stop-and-go mode at low speed.",
        "Dense traffic ahead. Reducing throttle and maintaining safe following distance.",
        "Traffic jam ahead. Moving slowly and monitoring the vehicles in front.",
    ],
}

def classify(s):
    if s['red_light'] == 1:
        return 'red_light'
    if s['near_stop'] == 1:
        return 'near_stop'
    if s['brake'] > 0.7:
        return 'hard_brake'
    if s['speed_ratio'] < 0.05 and s['throttle'] < 0.1:
        return 'traffic_jam'
    return None

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

counts = {}
with open(MANIFEST) as f_in, open(OUTPUT, 'w') as f_out:
    for line in f_in:
        s = json.loads(line)
        scenario = classify(s)
        if scenario is None:
            continue
        cot = random.choice(COT_TEMPLATES[scenario])
        entry = {
            "episode_index": s["episode_index"],
            "frame_index": s["frame_index"],
            "image_path": s["image_path"],
            "scenario": scenario,
            "instruction": s["instruction"],
            "cot_reasoning": cot,
            "action": {
                "steering": s["steering"],
                "throttle": s["throttle"],
                "brake": s["brake"]
            }
        }
        f_out.write(json.dumps(entry) + '\n')
        counts[scenario] = counts.get(scenario, 0) + 1

print("CoT dataset built!")
print(counts)
print(f"Total: {sum(counts.values())} dangerous frames saved to:\n{OUTPUT}")
import json, os, random
from collections import Counter

MANIFEST = r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\maram_groot_carla_frames 1\maram_groot_carla_frames\manifest.jsonl'
OUTPUT = r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model\data\danger_cot\cot_dataset.jsonl'


JAM_SPEED_RATIO  = 0.10  
JAM_THROTTLE     = 0.20   


OVERSAMPLE = {
    "near_stop":   6,   
    "traffic_jam": 50,  
    "red_light":   1,
    "hard_brake":  1,
}

COT_TEMPLATES = {
    "red_light": [
        "The traffic light ahead is RED. I must stop before the line. Applying brake now.",
        "Red light detected. Decelerating to a full stop. Throttle off, brake engaged.",
        "Traffic signal is RED. Stopping the vehicle to comply with traffic rules.",
        "Red signal ahead. I cannot proceed. Braking firmly and waiting for green.",
        "The intersection shows a red light. Halting the vehicle and monitoring the signal.",
    ],
    "near_stop": [
        "A stop sign is nearby. Slowing down and preparing to stop completely.",
        "Stop sign detected ahead. Reducing speed gradually and coming to a halt.",
        "Approaching a stop zone. Braking carefully and checking surroundings.",
        "Stop sign visible. I need to decelerate fully and yield before proceeding.",
        "Stop marker ahead. Reducing throttle, applying brake, scanning for crossing traffic.",
        "Mandatory stop detected. Full brake engagement required before the line.",
        "Nearing a stop zone. Slowing progressively and verifying intersection is clear.",
        "Stop sign in range. Coming to a complete halt and checking all directions.",
    ],
    "hard_brake": [
        "Sudden obstacle or hazard detected ahead. Emergency braking applied immediately.",
        "Unexpected event ahead. Hard brake engaged to avoid collision.",
        "Critical situation: rapid deceleration required. Brake applied with high force.",
        "Hazard in path. Maximum braking force applied to prevent impact.",
        "Collision risk detected. Immediate full brake — no time to steer around.",
    ],
    "traffic_jam": [
        "Traffic congestion detected. Switching to stop-and-go mode at low speed.",
        "Dense traffic ahead. Reducing throttle and maintaining safe following distance.",
        "Traffic jam ahead. Moving slowly and monitoring the vehicles in front.",
        "Congestion zone. Crawling forward carefully, keeping distance from the vehicle ahead.",
        "Heavy traffic detected. Throttle minimal, ready to brake if the lead vehicle stops.",
        "Queue of vehicles detected. Entering slow-follow mode and staying alert.",
        "Stop-and-go conditions. Speed is near zero — gentle throttle pulses only.",
        "Traffic standstill ahead. Matching the pace of surrounding vehicles with caution.",
    ],
}

def classify(s):
    if s['red_light'] == 1:
        return 'red_light'
    if s['near_stop'] == 1:
        return 'near_stop'
    if s['brake'] > 0.7:
        return 'hard_brake'
    if s['speed_ratio'] < JAM_SPEED_RATIO and s['throttle'] < JAM_THROTTLE:
        return 'traffic_jam'
    return None

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)


all_entries = {k: [] for k in COT_TEMPLATES}

with open(MANIFEST) as f_in:
    for line in f_in:
        s = json.loads(line)
        scenario = classify(s)
        if scenario is None:
            continue
        entry = {
            "episode_index": s["episode_index"],
            "frame_index":   s["frame_index"],
            "image_path":    s["image_path"],
            "scenario":      scenario,
            "instruction":   s["instruction"],
            "action": {
                "steering": s["steering"],
                "throttle": s["throttle"],
                "brake":    s["brake"]
            }
        }
        all_entries[scenario].append(entry)

print("Raw counts before oversampling:")
for k, v in all_entries.items():
    print(f"  {k}: {len(v)}")


final = []
for scenario, entries in all_entries.items():
    multiplier = OVERSAMPLE[scenario]
    for entry in entries:
        for _ in range(multiplier):
            copy = dict(entry)
            copy["cot_reasoning"] = random.choice(COT_TEMPLATES[scenario])
            final.append(copy)

random.shuffle(final)

with open(OUTPUT, 'w') as f_out:
    for entry in final:
        f_out.write(json.dumps(entry) + '\n')

counts = Counter(e["scenario"] for e in final)
print("\nFinal counts after oversampling:")
for k, v in counts.items():
    print(f"  {k}: {v}")
print(f"\nTotal: {sum(counts.values())} frames → {OUTPUT}")
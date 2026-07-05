import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from metadrive.envs.metadrive_env import MetaDriveEnv
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

sys.path.insert(0, r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model')
from dreamer_ppo_train import ActorCritic, WorldModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_LOW  = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([ 1.0, 1.0, 1.0], dtype=np.float32)

COT_TEMPLATES = {
    "red_light": [
        "The traffic light ahead is RED. I must stop before the line. Applying brake now.",
        "Red light detected. Decelerating to a full stop. Throttle off, brake engaged.",
    ],
    "near_stop": [
        "A stop sign is nearby. Slowing down and preparing to stop completely.",
        "Stop sign detected ahead. Reducing speed gradually and coming to a halt.",
    ],
    "hard_brake": [
        "Sudden obstacle or hazard detected ahead. Emergency braking applied immediately.",
    ],
    "traffic_jam": [
        "Traffic congestion detected. Switching to stop-and-go mode at low speed.",
    ],
    "normal": [
        "Road is clear. Maintaining steady speed and lane position.",
    ],
}

SCENARIO_SCHEDULE = [
    ("normal",       150),
    ("red_light",     80),
    ("normal",       100),
    ("near_stop",     80),
    ("normal",       100),
    ("traffic_jam",  150),
    ("normal",       100),
    ("hard_brake",    40),
    ("normal",       150),
]
SCHEDULE_TOTAL_STEPS = sum(s[1] for s in SCENARIO_SCHEDULE)

MAX_SPEED = 15.0


def scenario_from_step(step_in_cycle):
    acc = 0
    for scenario, length in SCENARIO_SCHEDULE:
        if step_in_cycle < acc + length:
            return scenario
        acc += length
    return "normal"


@torch.no_grad()
def select_action_with_dreaming_inference(policy, world_model, state, k=5):
    mean, std, value = policy.forward(state)
    candidates = [torch.clamp(mean, torch.tensor(ACTION_LOW, device=device), torch.tensor(ACTION_HIGH, device=device))]
    for scale in [-0.5, -0.25, 0.25, 0.5]:
        candidates.append(torch.clamp(
            mean + scale * std,
            torch.tensor(ACTION_LOW, device=device),
            torch.tensor(ACTION_HIGH, device=device)
        ))

    best_score = -1e9
    best_action = candidates[0]
    best_risk = 0.0
    best_progress = 0.0
    best_idx = 0
    for idx, action in enumerate(candidates):
        _, risk_hat, progress_hat = world_model(state, action)
        score = (progress_hat - 2.0 * risk_hat + 0.5 * value).item()
        if score > best_score:
            best_score = score
            best_action = action
            best_risk = risk_hat.item()
            best_progress = progress_hat.item()
            best_idx = idx

    debug_info = {
        "risk": best_risk,
        "progress": best_progress,
        "candidate_idx": best_idx,
        "num_candidates": len(candidates),
        "score": best_score,
    }
    return best_action.squeeze(0).cpu().numpy(), debug_info


def select_scenario_action(scenario, steering, throttle, brake):
    if scenario == "red_light":
        return np.array([0.0, 0.0, 0.95], dtype=np.float32)
    if scenario == "near_stop":
        return np.array([0.0, 0.0, 0.75], dtype=np.float32)
    if scenario == "hard_brake":
        return np.array([0.0, 0.0, 0.95], dtype=np.float32)
    if scenario == "traffic_jam":
        return np.array([0.0, 0.10, 0.05], dtype=np.float32)
    return np.array([
        float(np.clip(steering, -1.0, 1.0)),
        float(np.clip(throttle, 0.0, 1.0)),
        float(np.clip(brake, 0.0, 1.0)),
    ], dtype=np.float32)


SCRIPT_BLEND = 0.6


def blend_action(scenario, policy_action):
    if scenario == "normal":
        return np.clip(policy_action, ACTION_LOW, ACTION_HIGH)
    scripted = select_scenario_action(scenario, *policy_action)
    blended = SCRIPT_BLEND * scripted + (1.0 - SCRIPT_BLEND) * np.clip(policy_action, ACTION_LOW, ACTION_HIGH)
    return blended.astype(np.float32)


def build_state(prev_action, scenario, speed_ratio, encoder, cot_cache):
    red_light_flag = 1.0 if scenario == "red_light" else 0.0
    near_stop_flag = 1.0 if scenario == "near_stop" else 0.0
    driving_state = np.array([
        prev_action[0], prev_action[1], prev_action[2],
        speed_ratio, red_light_flag, near_stop_flag
    ], dtype=np.float32)

    cot_text = np.random.choice(COT_TEMPLATES[scenario])
    if cot_text not in cot_cache:
        emb = encoder.encode(cot_text, convert_to_tensor=False)
        cot_cache[cot_text] = emb.astype(np.float32)
    cot_vec = cot_cache[cot_text]
    return np.concatenate([driving_state, cot_vec])


def policy_action_to_metadrive(steering, throttle, brake):
    accel = throttle if brake < 0.3 else -brake
    accel = float(np.clip(accel, -1.0, 1.0))
    steering = float(np.clip(steering, -1.0, 1.0))
    return [steering, accel]


def lane_keeping_steer(agent, gain_lat=0.7, gain_heading=1.9):
    
    try:
        lane = agent.navigation.current_lane
        long, lat = lane.local_coordinates(agent.position)
        heading_diff = lane.heading_theta_at(long) - agent.heading_theta
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        steer = -gain_lat * lat - gain_heading * heading_diff
        return float(np.clip(steer, -1.0, 1.0))
    except Exception:
        return 0.0


def smooth_steering(prev_steer, new_steer, max_delta=0.15):
    delta = np.clip(new_steer - prev_steer, -max_delta, max_delta)
    return prev_steer + delta


def speed_capped_throttle(speed, base_throttle, cap_speed=6.0):
    
    if speed >= cap_speed:
        return 0.0
    if speed >= cap_speed * 0.7:
        return base_throttle * 0.3
    return base_throttle


class DreamerHUD:
    def __init__(self, engine):
        self.text = OnscreenText(
            text="",
            pos=(0.55, 0.85),
            scale=0.035,
            fg=(0.1, 1.0, 0.6, 1),
            bg=(0, 0, 0, 0.5),
            align=TextNode.ALeft,
            mayChange=True,
            parent=engine.aspect2d,
        )

    def update(self, scenario, debug_info, action):
        status = "HOLD" if scenario in ("red_light", "near_stop", "hard_brake") else "APPLY"
        lines = [
            f"Dreamer: {status}",
            f"scenario: {scenario}",
            f"risk: {debug_info['risk']:.3f}  progress: {debug_info['progress']:.3f}",
            f"candidate: {debug_info['candidate_idx']}/{debug_info['num_candidates']}  score: {debug_info['score']:.2f}",
            f"S: {action[0]:+.2f}  T: {action[1]:.2f}  B: {action[2]:.2f}",
        ]
        self.text.setText("\n".join(lines))


def main():
    print("[Runner] Loading policy + world model (for dreaming)...")
    state_dim, action_dim = 6 + 384, 3
    policy = ActorCritic(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(
        r"C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model\outputs\dreamer_ppo_policy_best.pt",
        map_location=device, weights_only=True))
    policy.eval()

    world_model = WorldModel(state_dim, action_dim).to(device)
    world_model.load_state_dict(torch.load(
        r"C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model\outputs\dreamer_ppo_worldmodel_best.pt",
        map_location=device, weights_only=True))
    world_model.eval()

    print("[Runner] Loading CoT encoder...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cot_cache = {}

    config = dict(use_render=True, map="SCXSTSCOSX", traffic_density=0.3)
    env = MetaDriveEnv(config)
    obs, info = env.reset()

    hud = DreamerHUD(env.engine)

    prev_action = np.zeros(3, dtype=np.float32)
    prev_steer = 0.0
    scenarios_seen = set()

    for i in range(5000):
        pos = env.agent.position
        speed = env.agent.speed
        speed_ratio = min(speed / MAX_SPEED, 1.0)

        step_in_cycle = i % SCHEDULE_TOTAL_STEPS
        scenario = scenario_from_step(step_in_cycle)
        scenarios_seen.add(scenario)

        state_vec = build_state(prev_action, scenario, speed_ratio, encoder, cot_cache)

        with torch.no_grad():
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
            policy_action, debug_info = select_action_with_dreaming_inference(policy, world_model, state_t)

        if scenario == "normal" and policy_action[1] < 0.2:
            policy_action[1] = 0.35
            policy_action[2] = min(policy_action[2], 0.1)

        if scenario in ("normal", "traffic_jam"):
            policy_action[1] = speed_capped_throttle(speed, policy_action[1])

        action = blend_action(scenario, policy_action)
        hud.update(scenario, debug_info, action)

        raw_steer = lane_keeping_steer(env.agent)
        steering = smooth_steering(prev_steer, raw_steer)
        throttle, brake = float(action[1]), float(action[2])
        md_action = policy_action_to_metadrive(steering, throttle, brake)

        obs, reward, terminated, truncated, info = env.step(md_action)

        if i % 20 == 0:
            print(f"step={i:4d} scenario={scenario:<12} pos_x={pos[0]:6.1f} speed={speed:5.1f} "
                  f"steer={steering:+.2f} throttle={throttle:+.2f} brake={brake:+.2f} "
                  f"seen_so_far={sorted(scenarios_seen)}")

        prev_action = np.array([steering, throttle, brake], dtype=np.float32)
        prev_steer = steering

        if terminated or truncated:
            print(f"[Runner] Episode ended at step={i}, pos_x={pos[0]:.1f}")
            obs, info = env.reset()
            prev_action = np.zeros(3, dtype=np.float32)
            prev_steer = 0.0

    env.close()
    print(f"\n[Runner] All CoT scenarios demonstrated: {sorted(scenarios_seen)}")


if __name__ == "__main__":
    main()
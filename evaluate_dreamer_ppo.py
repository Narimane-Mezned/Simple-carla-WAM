import os
import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model')
from envs.carla_offline_cot_env import CarlaOfflineCotEnv
from dreamer_ppo_train import ActorCritic, WorldModel

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUTS_DIR = "outputs"

ACTION_LOW  = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([ 1.0, 1.0, 1.0], dtype=np.float32)


def get_checkpoint_paths():
    policy_best = f"{OUTPUTS_DIR}/dreamer_ppo_policy_best.pt"
    world_best = f"{OUTPUTS_DIR}/dreamer_ppo_worldmodel_best.pt"
    policy_final = f"{OUTPUTS_DIR}/dreamer_ppo_policy.pt"
    world_final = f"{OUTPUTS_DIR}/dreamer_ppo_worldmodel.pt"

    if os.path.exists(policy_best) and os.path.exists(world_best):
        return policy_best, world_best
    return policy_final, world_final


@torch.no_grad()
def select_action_deterministic(policy, world_model, state, k=5):
    mean, std, value = policy.forward(state)
    mean_clipped = torch.clamp(
        mean,
        torch.tensor(ACTION_LOW, device=device),
        torch.tensor(ACTION_HIGH, device=device)
    )
    candidates = [mean_clipped]
    for scale in [-0.5, -0.25, 0.25, 0.5]:
        candidates.append(torch.clamp(
            mean + scale * std,
            torch.tensor(ACTION_LOW, device=device),
            torch.tensor(ACTION_HIGH, device=device)
        ))

    best_score = -1e9
    best_action = mean_clipped
    for action in candidates:
        _, risk_hat, progress_hat = world_model(state, action)
        score = (progress_hat - 2.0 * risk_hat + 0.5 * value).item()
        if score > best_score:
            best_score = score
            best_action = action

    return best_action, value


def is_correct_response(scenario, action_np, gt_action=None):
    if gt_action is not None:
        pred = np.asarray(action_np, dtype=np.float32)
        gt   = np.asarray(gt_action, dtype=np.float32)
        return float(np.mean((pred - gt) ** 2)) < 0.05

    throttle = action_np[1]
    brake    = action_np[2]

    if scenario == "hard_brake":
        return brake > 0.8 and throttle < 0.2
    if scenario == "red_light":
        return brake > 0.8 and throttle < 0.2
    if scenario == "near_stop":
        return brake > 0.5 and throttle < 0.3
    if scenario == "traffic_jam":
        return throttle < 0.2 and brake < 0.3
    return False


def evaluate(num_episodes=20):
    env        = CarlaOfflineCotEnv(max_ep_len=64)
    state_dim  = env.state_dim
    action_dim = env.action_dim

    policy_path, world_model_path = get_checkpoint_paths()
    print(f"[Eval] Using checkpoints: {policy_path} and {world_model_path}")

    policy = ActorCritic(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()

    world_model = WorldModel(state_dim, action_dim).to(device)
    world_model.load_state_dict(torch.load(world_model_path, map_location=device, weights_only=True))
    world_model.eval()

    stats = {
        "red_light":   {"total": 0, "correct": 0, "mse_sum": 0.0},
        "hard_brake":  {"total": 0, "correct": 0, "mse_sum": 0.0},
        "near_stop":   {"total": 0, "correct": 0, "mse_sum": 0.0},
        "traffic_jam": {"total": 0, "correct": 0, "mse_sum": 0.0},
    }
    total_rewards = []

    for ep in range(num_episodes):
        obs       = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = select_action_deterministic(policy, world_model, state, k=5)
            # FIX: defensive clip (action is already clipped inside select_action_deterministic)
            action_np = np.clip(action.squeeze(0).cpu().numpy(), ACTION_LOW, ACTION_HIGH)

            obs, reward, done, info = env.step(action_np)
            ep_reward += reward

            scenario = info["scenario"]
            gt_action = info.get("gt_action")
            if scenario in stats:
                stats[scenario]["total"] += 1
                mse = float(np.mean((action_np - np.asarray(gt_action, dtype=np.float32)) ** 2)) if gt_action is not None else 0.0
                stats[scenario]["mse_sum"] += mse
                if is_correct_response(scenario, action_np, gt_action):
                    stats[scenario]["correct"] += 1

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1:02d} | reward={ep_reward:.3f} | scenario={info['scenario']}")

    print("\n EVALUATION RESULTS (ALL SCENARIOS, SEED=42) ")
    print(f"Avg reward per episode : {np.mean(total_rewards):.3f}\n")
    for scenario, s in stats.items():
        total   = s["total"]
        correct = s["correct"]
        pct     = 100 * correct / max(total, 1)
        avg_mse = s["mse_sum"] / max(total, 1)
        print(f"{scenario:12s} | frames={total:4d} | correct={correct:4d} ({pct:.1f}%) | avg_mse={avg_mse:.3f}")


if __name__ == "__main__":
    evaluate()
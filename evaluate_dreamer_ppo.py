import numpy as np
import torch
import sys
sys.path.insert(0, r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model')
from envs.carla_offline_cot_env import CarlaOfflineCotEnv
from dreamer_ppo_train import ActorCritic, WorldModel, select_action_with_dreaming

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(num_episodes=10):
    env        = CarlaOfflineCotEnv(max_ep_len=64)
    state_dim  = env.state_dim
    action_dim = env.action_dim

    policy = ActorCritic(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load("outputs/dreamer_ppo_policy.pt", map_location=device))
    policy.eval()

    world_model = WorldModel(state_dim, action_dim).to(device)
    world_model.load_state_dict(torch.load("outputs/dreamer_ppo_worldmodel.pt", map_location=device))
    world_model.eval()

    total_rewards     = []
    red_light_correct = 0
    red_light_total   = 0
    red_light_violations = 0

    for ep in range(num_episodes):
        obs       = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            state  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = select_action_with_dreaming(policy, world_model, state, k=5)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            ep_reward += reward

            if info["scenario"] == "red_light":
                red_light_total += 1
                if action_np[1] < 0.3:
                    red_light_correct += 1
                else:
                    red_light_violations += 1

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1:02d} | reward={ep_reward:.3f} | scenario={info['scenario']}")

    print("\nEVALUATION RESULTS : ")
    print(f"Avg reward per episode : {np.mean(total_rewards):.3f}")
    print(f"Red light frames total : {red_light_total}")
    print(f"Correct stops at red   : {red_light_correct} ({100*red_light_correct/max(red_light_total,1):.1f}%)")
    print(f"Red light violations   : {red_light_violations} ({100*red_light_violations/max(red_light_total,1):.1f}%)")

if __name__ == "__main__":
    evaluate()
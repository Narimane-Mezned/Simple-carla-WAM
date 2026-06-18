import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import sys
sys.path.insert(0, r'C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model')
from envs.carla_offline_cot_env import CarlaOfflineCotEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self, size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.size       = size
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.states     = np.zeros((size, state_dim),  dtype=np.float32)
        self.actions    = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards    = np.zeros(size, dtype=np.float32)
        self.dones      = np.zeros(size, dtype=np.float32)
        self.values     = np.zeros(size, dtype=np.float32)
        self.log_probs  = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns    = np.zeros(size, dtype=np.float32)
        self.ptr        = 0
        self.path_start_idx = 0

    def store(self, state, action, reward, done, value, log_prob):
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values  = np.append(self.values[path_slice],  last_value)
        dones   = np.append(self.dones[path_slice],   0.0)
        gae = 0.0
        for t in reversed(range(len(rewards) - 1)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * nonterminal - values[t]
            gae   = delta + self.gamma * self.gae_lambda * nonterminal * gae
            self.advantages[self.path_start_idx + t] = gae
            self.returns[self.path_start_idx + t]    = gae + values[t]
        self.path_start_idx = self.ptr

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0

    def get(self):
        adv = self.advantages[:self.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return dict(
            states     = torch.tensor(self.states[:self.ptr],    device=device),
            actions    = torch.tensor(self.actions[:self.ptr],   device=device),
            rewards    = torch.tensor(self.rewards[:self.ptr],   device=device),
            dones      = torch.tensor(self.dones[:self.ptr],     device=device),
            values     = torch.tensor(self.values[:self.ptr],    device=device),
            log_probs  = torch.tensor(self.log_probs[:self.ptr], device=device),
            advantages = torch.tensor(adv,                       device=device),
            returns    = torch.tensor(self.returns[:self.ptr],   device=device),
        )


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
        )
        self.actor_mean    = torch.nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.critic        = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s):
        z     = self.shared(s)
        mean  = self.actor_mean(z)
        std   = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(z).squeeze(-1)
        return mean, std, value

    def act(self, s):
        mean, std, value = self.forward(s)
        dist     = Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, s, a):
        mean, std, value = self.forward(s)
        dist     = Normal(mean, std)
        log_prob = dist.log_prob(a).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class WorldModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
        )
        self.next_state    = torch.nn.Linear(hidden_dim, state_dim)
        self.risk_head     = torch.nn.Linear(hidden_dim, 1)
        self.progress_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x            = torch.cat([s, a], dim=-1)
        z            = self.net(x)
        s_next_hat   = self.next_state(z)
        risk_hat     = self.risk_head(z).squeeze(-1)
        progress_hat = self.progress_head(z).squeeze(-1)
        return s_next_hat, risk_hat, progress_hat


@torch.no_grad()
def select_action_with_dreaming(policy, world_model, state, k=5):
    candidates = []
    for _ in range(k):
        action, logp, value = policy.act(state)
        _, risk_hat, progress_hat = world_model(state, action)
        score = progress_hat - 2.0 * risk_hat + 0.5 * value
        candidates.append((score.item(), action, logp, value))
    best = max(candidates, key=lambda x: x[0])
    return best[1], best[2], best[3]


def train(num_episodes=50, rollout_size=512):
    env        = CarlaOfflineCotEnv(max_ep_len=64)
    state_dim  = env.state_dim
    action_dim = env.action_dim

    policy      = ActorCritic(state_dim, action_dim).to(device)
    world_model = WorldModel(state_dim, action_dim).to(device)
    opt_pi      = torch.optim.Adam(policy.parameters(),      lr=3e-4)
    opt_wm      = torch.optim.Adam(world_model.parameters(), lr=3e-4)
    buffer      = RolloutBuffer(rollout_size, state_dim, action_dim)

    for episode in range(num_episodes):
        obs = env.reset()
        buffer.clear()

        while buffer.ptr < rollout_size:
            state  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, logp, value = select_action_with_dreaming(policy, world_model, state)
            action_np  = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)
            buffer.store(obs, action_np, reward, float(done), value.item(), logp.item())
            obs = next_obs
            if done:
                buffer.finish_path(last_value=0.0)
                obs = env.reset()

        if buffer.ptr > 0 and buffer.ptr == rollout_size:
            with torch.no_grad():
                ns     = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, _, last_value = policy.forward(ns)
            buffer.finish_path(last_value=last_value.item())

        batch = buffer.get()

        for _ in range(10):
            new_logp, entropy, values = policy.evaluate(batch["states"], batch["actions"])
            ratio   = torch.exp(new_logp - batch["log_probs"])
            clipped = torch.clamp(ratio, 0.8, 1.2)
            loss_pi = -torch.mean(torch.min(ratio * batch["advantages"], clipped * batch["advantages"]))
            loss_v  = F.mse_loss(values, batch["returns"])
            loss    = loss_pi + 0.5 * loss_v - 0.01 * entropy.mean()
            opt_pi.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_pi.step()

            s_next_hat, risk_hat, progress_hat = world_model(batch["states"], batch["actions"])
            loss_wm = F.mse_loss(s_next_hat, batch["states"])
            opt_wm.zero_grad()
            loss_wm.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 0.5)
            opt_wm.step()

        print(f"Episode {episode+1:03d} | PPO loss: {loss.item():.4f} | WM loss: {loss_wm.item():.4f}")

    torch.save(policy.state_dict(),      "outputs/dreamer_ppo_policy.pt")
    torch.save(world_model.state_dict(), "outputs/dreamer_ppo_worldmodel.pt")
    print("Training complete. Checkpoints saved to outputs/")


if __name__ == "__main__":
    train()
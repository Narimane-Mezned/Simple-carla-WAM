"""
CONFIRMED dataset structure (from manifest.jsonl):
  Each line contains:
    episode_index, source_run, frame_index, image_path,
    steering, throttle, brake, speed_ratio,
    red_light, near_stop, instruction, done

Observation : RGB image (64x64x3) uint8
Action      : [steering, throttle, brake] float32
"""

import os
import json
import numpy as np
from PIL import Image


class CARLADatasetEnv:

    def __init__(self, dataset_root, manifest_path, image_size=64, split="train", val_ratio=0.1):
        self.dataset_root = dataset_root
        self.image_size = image_size

        # Load and group frames by episode
        self.episodes = self._load_episodes(manifest_path)
        episode_names = sorted(self.episodes.keys())

        # Train / val split by episode
        n_val = max(1, int(len(episode_names) * val_ratio))
        if split == "val":
            episode_names = episode_names[:n_val]
        else:
            episode_names = episode_names[n_val:]

        self.episode_names = episode_names
        print(f"[CARLAEnv] {split}: {len(self.episode_names)} episodes")

        # State
        self.current_episode = None
        self.current_step = 0
        self.done = True

        # Spaces
        self.obs_space  = {"image": (image_size, image_size, 3)}
        self.act_space  = {"action": np.zeros(3, dtype=np.float32)}

    # ── Public API ───────────────────────────────────────────────────

    def reset(self):
        ep_name = np.random.choice(self.episode_names)
        self.current_episode = self.episodes[ep_name]
        self.current_step = 0
        self.done = False
        return self._get_obs(0)

    def step(self, action):
        frame = self.current_episode[self.current_step]

        # Dataset action
        dataset_action = np.array([
            float(frame["steering"]),
            float(frame["throttle"]),
            float(frame["brake"]),
        ], dtype=np.float32)

        # Reward: smooth driving
        # + speed  - swerving  - braking  - red light  - near stop sign
        reward = (
            + float(frame["speed_ratio"]) * 0.5
            - abs(float(frame["steering"])) * 0.3
            - float(frame["brake"]) * 0.5
            - float(frame["red_light"]) * 1.0
            - float(frame["near_stop"]) * 0.2
        )
        reward = float(np.clip(reward, -1.0, 1.0))

        # Advance
        self.current_step += 1
        is_last = (self.current_step >= len(self.current_episode) - 1)

        # Check if dataset says done
        dataset_done = str(frame.get("done", "False")).lower() == "true"
        self.done = is_last or dataset_done

        obs = self._get_obs(min(self.current_step, len(self.current_episode) - 1))

        info = {
            "dataset_action": dataset_action,
            "speed_ratio":    float(frame["speed_ratio"]),
            "red_light":      int(frame["red_light"]),
            "instruction":    frame.get("instruction", ""),
        }

        return obs, reward, self.done, info

    # ── Private helpers ──────────────────────────────────────────────

    def _get_obs(self, idx):
        frame = self.current_episode[idx]
        # image_path in manifest is relative, e.g. "runs/episode_000000/..."
        img_path = os.path.join(self.dataset_root, frame["image_path"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return {"image": np.array(img, dtype=np.uint8)}

    def _load_episodes(self, manifest_path):
        episodes = {}
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                ep = frame["source_run"]          # "episode_000000"
                if ep not in episodes:
                    episodes[ep] = []
                episodes[ep].append(frame)

        for ep in episodes:
            episodes[ep].sort(key=lambda x: x["frame_index"])

        total = sum(len(v) for v in episodes.values())
        print(f"[CARLAEnv] Loaded {len(episodes)} episodes, {total} frames")
        return episodes


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    DATASET_ROOT  = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    MANIFEST_PATH = sys.argv[2] if len(sys.argv) > 2 else "./dataset/manifest.jsonl"

    env = CARLADatasetEnv(DATASET_ROOT, MANIFEST_PATH, image_size=64, split="train")
    obs = env.reset()
    print(f"Image shape : {obs['image'].shape}")   # (64, 64, 3)
    print(f"Image dtype : {obs['image'].dtype}")   # uint8
    obs, reward, done, info = env.step(np.zeros(3))
    print(f"Reward      : {reward:.4f}")
    print(f"Action      : {info['dataset_action']}")
    print(f"Instruction : {info['instruction']}")
    print("carla_dataset.py OK")
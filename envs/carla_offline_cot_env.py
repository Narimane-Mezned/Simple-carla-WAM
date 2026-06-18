import json
import os
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

DATASET_ROOT = r"C:\Users\nerim\OneDrive\Bureau\Carla-world-model\maram_groot_carla_frames 1\maram_groot_carla_frames"
COT_PATH     = r"C:\Users\nerim\OneDrive\Bureau\Carla-world-model\world-model\data\danger_cot\cot_dataset.jsonl"

W_PROGRESS  =  1.0
W_BRAKE     = -0.5
W_RED_LIGHT = -2.0
W_SMOOTH    = -0.1

class CarlaOfflineCotEnv:
    def __init__(self, max_ep_len=64):
        self.max_ep_len = max_ep_len
        print("[CarlaOfflineCotEnv] Loading CoT dataset...")
        self._episodes = self._load_episodes()
        self._ep_keys  = list(self._episodes.keys())
        print(f"[CarlaOfflineCotEnv] {len(self._ep_keys)} episodes loaded")
        print("[CarlaOfflineCotEnv] Loading CoT encoder...")
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        print("[CarlaOfflineCotEnv] Encoder ready.")
        self._cot_cache     = {}
        self._current_ep    = []
        self._frame_idx     = 0
        self._prev_action   = np.zeros(3, dtype=np.float32)
        self._ep_step       = 0
        self.state_dim      = 6 + 384
        self.action_dim     = 3

    def reset(self):
        ep_key = self._ep_keys[np.random.randint(len(self._ep_keys))]
        self._current_ep  = self._episodes[ep_key]
        self._frame_idx   = 0
        self._ep_step     = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        return self._get_state()

    def step(self, action):
        frame = self._current_ep[self._frame_idx]
        gt_action = np.array([
            frame["action"]["steering"],
            frame["action"]["throttle"],
            frame["action"]["brake"],
        ], dtype=np.float32)
        reward = self._compute_reward(frame, action, gt_action)
        self._frame_idx  += 1
        self._ep_step    += 1
        self._prev_action = np.array(action, dtype=np.float32)
        done = (self._frame_idx >= len(self._current_ep) or
                self._ep_step  >= self.max_ep_len)
        if done:
            next_obs = self.reset()
        else:
            next_obs = self._get_state()
        info = {
            "scenario":  frame["scenario"],
            "vru_risk":  1.0 if frame["scenario"] in ("hard_brake", "near_stop") else 0.0,
            "progress":  1.0 - (self._frame_idx / len(self._current_ep)),
            "red_light": frame.get("red_light", 0),
        }
        return next_obs, reward, done, info

    def _get_state(self):
        frame = self._current_ep[self._frame_idx]
        driving_state = np.array([
            float(frame["action"]["steering"]),
            float(frame["action"]["throttle"]),
            float(frame["action"]["brake"]),
            float(frame.get("speed_ratio", 0.0)),
            float(frame.get("red_light",   0.0)),
            float(frame.get("near_stop",   0.0)),
        ], dtype=np.float32)
        cot_text = frame["cot_reasoning"]
        if cot_text not in self._cot_cache:
            emb = self._encoder.encode(cot_text, convert_to_tensor=False)
            self._cot_cache[cot_text] = emb.astype(np.float32)
        cot_vec = self._cot_cache[cot_text]
        return np.concatenate([driving_state, cot_vec])

    def _compute_reward(self, frame, pred_action, gt_action):
        pred = np.array(pred_action, dtype=np.float32)
        gt   = np.array(gt_action,   dtype=np.float32)
        action_match = -np.mean((pred - gt) ** 2)
        red_penalty  = W_RED_LIGHT if (frame.get("red_light", 0) == 1 and pred[1] > 0.3) else 0.0
        smooth_penalty = W_SMOOTH * np.sum(np.abs(pred - self._prev_action))
        return float(W_PROGRESS * action_match + red_penalty + smooth_penalty)

    def _load_episodes(self):
        episodes = defaultdict(list)
        with open(COT_PATH) as f:
            for line in f:
                entry = json.loads(line)
                episodes[entry["episode_index"]].append(entry)
        for k in episodes:
            episodes[k].sort(key=lambda x: x["frame_index"])
        return {k: v for k, v in episodes.items() if len(v) >= 4}
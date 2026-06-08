# Simple CARLA World Action Model (WAM)

A DreamerV3-style World Action Model implemented from scratch in PyTorch, trained on offline CARLA front-camera driving data.

---

## What it does

Given a sequence of past driving frames, the model:
- **Compresses** each frame into a compact latent representation
- **Models** how the world evolves over time given driving actions
- **Dreams** future frames without seeing real images
- **Predicts** future driving actions (steering, throttle, brake)

---

## Architecture

| Component | Type | Role |
|---|---|---|
| ConvEncoder | 4-layer CNN + SiLU | Image (64×64×3) → latent vector (256-dim) |
| RSSM | GRU (512) + prior/posterior | Temporal dynamics + dreaming |
| ConvDecoder | 4-layer transposed CNN | Latent vector → reconstructed image |
| ActionHead | 3-layer MLP + Tanh | World state → [steering, throttle, brake] |

**Total parameters: 8.2M**

---

## Dataset

The model trains on the CARLA driving dataset :
- 133,672 sequential front-camera frames
- 209 driving episodes
- Each frame linked to: steering, throttle, brake, speed_ratio, red_light, instruction

The dataset is not included in this repository. 

---

## Project Structure

```
Simple-carla-WAM/
├── envs/
│   └── carla_dataset.py     — dataset loader (manifest.jsonl → sequences)
├── train_carla.py           — full WAM training script
├── evaluate_carla.py        — dreaming evaluation → generates dream.gif
├── cosmos_transfer.py       — converts CARLA frames to photorealistic (SDU server)
├── test_setup.py            — sanity checks before training
└── configs/
    └── carla.yaml           — hyperparameters
```

---

## Installation

```bash
# Python 3.12 required (3.13+ not supported by PyTorch CUDA builds)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow pandas
```

---

## Usage

### Step 1 — Verify setup
```bash
python test_setup.py \
    --dataset_root /path/to/dataset \
    --manifest_path /path/to/dataset/manifest.jsonl
```

### Step 2 — Train
```bash
python train_carla.py \
    --dataset_root /path/to/dataset \
    --manifest_path /path/to/dataset/manifest.jsonl \
    --steps 500000 \
    --image_size 96 \
    --device cuda \
    --logdir ./logdir/carla_sdu
```

### Step 3 — Generate dreaming GIF
```bash
python evaluate_carla.py \
    --checkpoint ./logdir/carla_sdu/checkpoint_final.pt \
    --dataset_root /path/to/dataset \
    --manifest_path /path/to/dataset/manifest.jsonl \
    --output ./outputs/dream_final.gif \
    --device cuda
```

### Step 4 — (Optional) Apply Cosmos Transfer before training
```bash
python cosmos_transfer.py \
    --input_root  /path/to/dataset \
    --output_root /path/to/dataset_cosmos \
    --manifest_path /path/to/dataset/manifest.jsonl \
    --device cuda
```
Then point `--dataset_root` to the Cosmos output folder in Step 2.

---

## Training Results (proof of concept — 50,000 steps, RTX 2050)

| Step | Loss | Recon | Action |
|---|---|---|---|
| 500 | 1.054 | 0.036 | 0.0185 |
| 10,000 | 1.012 | 0.010 | 0.0024 |
| 50,000 | 1.007 | 0.005 | 0.0017 |

- Reconstruction loss: ↓ 86%
- Action prediction loss: ↓ 91%
- Training time: 236 minutes on RTX 2050


---

## Recommended hyperparameters for full run 

| Parameter | Value |
|---|---|
| Steps | 500,000 |
| Image size | 96×96 |
| Batch size | 16 |
| Sequence length | 16 |
| Latent dim | 256 |
| Deter dim | 512 |
| Device | CUDA (A100 / H100) |


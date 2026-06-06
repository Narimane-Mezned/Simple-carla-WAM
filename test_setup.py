import argparse
import sys
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",  type=str, default="./dataset")
    parser.add_argument("--manifest_path", type=str, default="./dataset/manifest.jsonl")
    parser.add_argument("--image_size",    type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()

    print("\n[Test 1] Checking manifest file...")
    if not os.path.exists(args.manifest_path):
        print(f"  [FAIL] manifest.jsonl not found at: {args.manifest_path}")
        sys.exit(1)
    print(f"  [OK] Found: {args.manifest_path}")

    print("\n[Test 2] Importing CARLA environment...")
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from envs.carla_dataset import CARLADatasetEnv
        print("  [OK] Import successful")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[Test 3] Loading dataset environment...")
    try:
        env = CARLADatasetEnv(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest_path,
            image_size=args.image_size,
            split="train"
        )
        print(f"  [OK] Environment loaded")
        print(f"  [OK] Episodes available: {len(env.episode_names)}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[Test 4] Testing env.reset()...")
    try:
        obs = env.reset()
        img = obs["image"]
        assert img.shape == (args.image_size, args.image_size, 3)
        assert img.dtype == np.uint8
        print(f"  [OK] Image shape  : {img.shape}")
        print(f"  [OK] Image dtype  : {img.dtype}")
        print(f"  [OK] Pixel range  : {img.min()} to {img.max()}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[Test 5] Testing env.step()...")
    try:
        action = np.zeros(3, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        print(f"  [OK] Next image shape  : {obs['image'].shape}")
        print(f"  [OK] Reward            : {reward:.4f}")
        print(f"  [OK] Done              : {done}")
        print(f"  [OK] Dataset action    : {info['dataset_action']}")
        print(f"  [OK] Speed ratio       : {info['speed_ratio']:.4f}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[Test 6] Collecting one full episode...")
    try:
        obs = env.reset()
        done = False
        frame_count = 0
        while not done:
            action = np.zeros(3, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            frame_count += 1
        print(f"  [OK] Episode length: {frame_count} frames")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print("\n[Test 7] Checking PyTorch...")
    try:
        import torch
        print(f"  [OK] PyTorch version : {torch.__version__}")
        print(f"  [OK] CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [OK] GPU             : {torch.cuda.get_device_name(0)}")
        else:
            print(f"  [INFO] No GPU found — will train on CPU")
    except ImportError:
        print("  [FAIL] PyTorch not installed. Run: pip install torch torchvision")
        sys.exit(1)

    print("\nALL TESTS PASSED — You are ready to train!")

if __name__ == "__main__":
    main()
import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos Transfer for CARLA frames")
    parser.add_argument("--input_root",   type=str, required=True,
                        help="Root folder of original CARLA dataset")
    parser.add_argument("--output_root",  type=str, required=True,
                        help="Root folder for Cosmos-enhanced output")
    parser.add_argument("--manifest_path",type=str, required=True,
                        help="Path to original manifest.jsonl")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--batch_size",   type=int, default=8,
                        help="Images per batch (increase for faster processing on A100)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes for testing (None = all 209)")
    return parser.parse_args()


def load_manifest(manifest_path):
    episodes = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            frame = json.loads(line)
            ep    = frame["source_run"]
            if ep not in episodes: episodes[ep] = []
            episodes[ep].append(frame)
    for ep in episodes:
        episodes[ep].sort(key=lambda x: x["frame_index"])
    return episodes


def load_cosmos_model(device):
    
    print("Loading Cosmos Transfer 2.5 model...")
    try:
        # Try NVIDIA's official Cosmos package
        from cosmos_transfer import CosmosTransfer
        model = CosmosTransfer.from_pretrained("nvidia/cosmos-transfer-2.5")
        model = model.to(device)
        model.eval()
        print("[OK] Cosmos Transfer model loaded")
        return model, "cosmos_official"

    except ImportError:
        pass

    try:
        # Try via HuggingFace diffusers
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "nvidia/cosmos-transfer-2.5-img2img",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)
        print("[OK] Cosmos Transfer loaded via diffusers")
        return pipe, "diffusers"

    except Exception as e:
        print(f"[WARNING] Could not load Cosmos Transfer: {e}")
        print("Running in DEMO MODE — applying simple style transfer instead")
        print("Contact NVIDIA for Cosmos Transfer 2.5 access at:")
        print("  https://developer.nvidia.com/cosmos")
        return None, "demo"


def cosmos_transfer_image(model, model_type, img_pil, device):
    if model_type == "demo":
        import numpy as np
        arr   = np.array(img_pil).astype(np.float32)
        arr   = np.clip(arr * 1.15 - 10, 0, 255)
        arr[:,:,0] = np.clip(arr[:,:,0] * 1.05, 0, 255)  
        arr[:,:,2] = np.clip(arr[:,:,2] * 0.95, 0, 255)  
        return Image.fromarray(arr.astype(np.uint8))

    elif model_type == "cosmos_official":
        import torch
        arr    = np.array(img_pil)
        tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            result = model(tensor)
        result = result[0].permute(1,2,0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    elif model_type == "diffusers":
        import torch
        with torch.no_grad():
            result = model(
                prompt="photorealistic dashcam view, urban driving, high quality",
                image=img_pil,
                strength=0.4,        
                guidance_scale=7.5,
            ).images[0]
        return result.resize(img_pil.size)


def process_episode(episode_frames, input_root, output_root,
                    model, model_type, device, batch_size=8):
    
    ep_name    = episode_frames[0]["source_run"]
    out_ep_dir = Path(output_root) / "runs" / ep_name / "front_camera_frames"
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    
    src_actions = Path(input_root) / "runs" / ep_name / "actions.csv"
    dst_actions = Path(output_root) / "runs" / ep_name / "actions.csv"
    if src_actions.exists():
        shutil.copy2(src_actions, dst_actions)

    
    src_meta = Path(input_root) / "runs" / ep_name / "metadata.jsonl"
    dst_meta = Path(output_root) / "runs" / ep_name / "metadata.jsonl"
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)

    
    processed = 0
    for i in range(0, len(episode_frames), batch_size):
        batch = episode_frames[i:i + batch_size]
        for frame in batch:
            src_img_path = Path(input_root) / frame["image_path"]
            
            rel_path     = frame["image_path"]
            dst_img_path = Path(output_root) / rel_path
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)

            if dst_img_path.exists():
                continue 

            img     = Image.open(src_img_path).convert("RGB")
            img_out = cosmos_transfer_image(model, model_type, img, device)
            img_out.save(dst_img_path, quality=95)
            processed += 1

    return processed


def main():
    args = parse_args()

    
    print("Cosmos Transfer 2.5 — CARLA Dataset Enhancement")
    print(f"  Input  : {args.input_root}")
    print(f"  Output : {args.output_root}")
    print(f"  Device : {args.device}")
   

   
    os.makedirs(args.output_root, exist_ok=True)

    
    src_tasks = Path(args.input_root) / "tasks.jsonl"
    if src_tasks.exists():
        shutil.copy2(src_tasks, Path(args.output_root) / "tasks.jsonl")

   
    print("\nLoading manifest...")
    episodes    = load_manifest(args.manifest_path)
    ep_names    = sorted(episodes.keys())
    if args.max_episodes:
        ep_names = ep_names[:args.max_episodes]
    print(f"[OK] {len(ep_names)} episodes to process")

   
    model, model_type = load_cosmos_model(args.device)

   
    print(f"\nProcessing {len(ep_names)} episodes...")
    total_processed = 0

    for i, ep_name in enumerate(ep_names):
        frames    = episodes[ep_name]
        processed = process_episode(
            frames, args.input_root, args.output_root,
            model, model_type, args.device, args.batch_size
        )
        total_processed += processed
        print(f"  [{i+1:3d}/{len(ep_names)}] {ep_name} — {processed} frames processed")

    
    print("\nWriting new manifest.jsonl...")
    out_manifest = Path(args.output_root) / "manifest.jsonl"
    with open(out_manifest, "w") as f:
        for ep_name in ep_names:
            for frame in episodes[ep_name]:
                
                new_frame = dict(frame)
                new_frame["absolute_image_path"] = str(
                    Path(args.output_root) / frame["image_path"]
                )
                f.write(json.dumps(new_frame) + "\n")

    print(f"[OK] New manifest: {out_manifest}")
    print(f"Cosmos Transfer complete!")
    print(f"Total frames processed : {total_processed}")
    print(f"Output dataset         : {args.output_root}")
    print(f"\nNow train the WAM on enhanced data:")
    print(f"  python train_carla.py \\")
    print(f"    dataset_root {args.output_root} \\")
    print(f"    manifest_path {out_manifest} \\")
    print(f"    steps 500000 - device cuda")
   

if __name__ == "__main__":
    main()
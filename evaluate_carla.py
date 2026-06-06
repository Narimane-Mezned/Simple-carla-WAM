import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))


class ConvEncoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=256, depth=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth,       4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth,   depth*2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth*2, depth*4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth*4, depth*8, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
        )
        flat = depth * 8 * (image_size // 16) * (image_size // 16)
        self.proj = nn.Linear(flat, latent_dim)

    def forward(self, x):
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.view(B*T, *x.shape[2:])
            out = self.proj(self.net(x))
            return out.view(B, T, -1)
        return self.proj(self.net(x))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=256, image_size=64, depth=32):
        super().__init__()
        self.depth = depth
        self.proj  = nn.Linear(latent_dim, depth*8*4*4)
        self.net   = nn.Sequential(
            nn.ConvTranspose2d(depth*8, depth*4, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth*4, depth*2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth*2, depth,   4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(depth,   3,       4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        if z.dim() == 3:
            B, T, D = z.shape
            x = self.proj(z.view(B*T, D)).view(B*T, self.depth*8, 4, 4)
            out = self.net(x)
            return out.view(B, T, *out.shape[1:])
        x = self.proj(z).view(z.shape[0], self.depth*8, 4, 4)
        return self.net(x)


class RSSM(nn.Module):
    def __init__(self, latent_dim=256, deter_dim=512, action_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.deter_dim  = deter_dim
        self.gru        = nn.GRUCell(latent_dim + action_dim, deter_dim)
        self.prior_net  = nn.Sequential(
            nn.Linear(deter_dim, deter_dim), nn.SiLU(),
            nn.Linear(deter_dim, latent_dim * 2)
        )
        self.post_net   = nn.Sequential(
            nn.Linear(deter_dim + latent_dim, deter_dim), nn.SiLU(),
            nn.Linear(deter_dim, latent_dim * 2)
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def forward(self, encoded_imgs, actions):
        B, T, _ = encoded_imgs.shape
        device   = encoded_imgs.device
        h, z     = self.initial_state(B, device)
        posts, priors, states = [], [], []
        for t in range(T):
            gru_input  = torch.cat([z, actions[:, t]], dim=-1)
            h          = self.gru(gru_input, h)
            prior      = self.prior_net(h)
            priors.append(prior)
            post_input = torch.cat([h, encoded_imgs[:, t]], dim=-1)
            post       = self.post_net(post_input)
            posts.append(post)
            mu, log_sigma = post.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma.clamp(-4, 4))
            z     = mu + sigma * torch.randn_like(sigma)
            states.append(torch.cat([h, z], dim=-1))
        return (
            torch.stack(posts,  dim=1),
            torch.stack(priors, dim=1),
            torch.stack(states, dim=1),
        )

    def imagine(self, h, z, action, steps):
        imagined = []
        for _ in range(steps):
            gru_input     = torch.cat([z, action], dim=-1)
            h             = self.gru(gru_input, h)
            prior         = self.prior_net(h)
            mu, log_sigma = prior.chunk(2, dim=-1)
            sigma         = torch.exp(log_sigma.clamp(-4, 4))
            z             = mu + sigma * torch.randn_like(sigma)
            imagined.append(torch.cat([h, z], dim=-1))
        return torch.stack(imagined, dim=0)


class ActionHead(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(),
            nn.Linear(256, 256),       nn.SiLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class WorldModel(nn.Module):
    def __init__(self, image_size=64, latent_dim=256, deter_dim=512, action_dim=3):
        super().__init__()
        self.encoder     = ConvEncoder(image_size, latent_dim)
        self.rssm        = RSSM(latent_dim, deter_dim, action_dim)
        self.decoder     = ConvDecoder(deter_dim + latent_dim, image_size)
        self.action_head = ActionHead(deter_dim + latent_dim, action_dim)
        self.latent_dim  = latent_dim
        self.deter_dim   = deter_dim

    def forward(self, images, actions):
        encoded = self.encoder(images)
        posts, priors, states = self.rssm(encoded, actions)
        recon        = self.decoder(states)
        pred_actions = self.action_head(states)
        return recon, posts, priors, pred_actions

    def dream(self, images, actions, dream_steps=15):
        encoded = self.encoder(images)
        _, _, states = self.rssm(encoded, actions)
        last_state   = states[:, -1]
        h = last_state[:, :self.deter_dim]
        z = last_state[:, self.deter_dim:]
        last_action     = actions[:, -1]
        imagined_states = self.rssm.imagine(h, z, last_action, dream_steps)
        imagined_states = imagined_states.permute(1, 0, 2)
        imagined_imgs   = self.decoder(imagined_states)
        pred_actions    = self.action_head(imagined_states)
        return imagined_imgs, pred_actions


def load_episode(manifest_path, dataset_root, episode_idx=0, image_size=64):
    episodes = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d  = json.loads(line)
            ep = d["source_run"]
            if ep not in episodes: episodes[ep] = []
            episodes[ep].append(d)
    for ep in episodes:
        episodes[ep].sort(key=lambda x: x["frame_index"])

    ep_name = sorted(episodes.keys())[episode_idx]
    frames  = episodes[ep_name]
    print(f"Using episode: {ep_name} ({len(frames)} frames)")

    images, actions = [], []
    for frame in frames[:30]:
        img_path = os.path.join(dataset_root, frame["image_path"])
        img      = Image.open(img_path).convert("RGB")
        img      = img.resize((image_size, image_size), Image.BILINEAR)
        img_t    = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        images.append(img_t.permute(2, 0, 1))
        actions.append(torch.tensor([
            float(frame["steering"]),
            float(frame["throttle"]),
            float(frame["brake"]),
        ]))

    return torch.stack(images), torch.stack(actions)


def tensor_to_pil(t):
    arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def add_label(img, text, color):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((2, 2), text, fill=color)
    return img


def make_gif(real_frames, dreamed_frames, pred_actions, output_path, context_len=5):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    frames_out = []
    total      = len(real_frames)

    for i in range(total):
        real_img = tensor_to_pil(real_frames[i])
        label    = "CONTEXT" if i < context_len else "REAL"
        real_img = add_label(real_img, f"{label} t={i}", (100, 255, 100))

        if i < len(dreamed_frames):
            dream_img = tensor_to_pil(dreamed_frames[i])
            if i < context_len:
                dream_img = add_label(dream_img, f"CONTEXT t={i}", (100, 255, 100))
            else:
                action    = pred_actions[i - context_len]
                label     = f"DREAM t={i} | st={action[0]:.2f} th={action[1]:.2f}"
                dream_img = add_label(dream_img, label, (255, 150, 50))
        else:
            dream_img = Image.new("RGB", real_img.size, (30, 30, 30))
            dream_img = add_label(dream_img, "---", (100, 100, 100))

        combined = Image.new("RGB", (real_img.width * 2 + 4, real_img.height), (50, 50, 50))
        combined.paste(real_img,  (0, 0))
        combined.paste(dream_img, (real_img.width + 4, 0))
        frames_out.append(combined)

    frames_out[0].save(
        output_path,
        save_all=True,
        append_images=frames_out[1:],
        duration=200,
        loop=0,
    )
    print(f"[OK] GIF saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--dataset_root",  type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--context_len",   type=int, default=5)
    parser.add_argument("--dream_steps",   type=int, default=15)
    parser.add_argument("--image_size",    type=int, default=64)
    parser.add_argument("--output",        type=str, default="./outputs/dream.gif")
    parser.add_argument("--device",        type=str, default="cpu")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Context     : {args.context_len} real frames")
    print(f"Dream steps : {args.dream_steps} imagined frames")
    print(f"Output      : {args.output}")

    print("\nLoading checkpoint...")
    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = WorldModel(
        image_size=ckpt_args.get("image_size", args.image_size),
        latent_dim=ckpt_args.get("latent_dim", 256),
        deter_dim =ckpt_args.get("deter_dim",  512),
        action_dim=3,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[OK] Loaded checkpoint at step {ckpt.get('step', '?')}")

    print("\nLoading real frames from dataset...")
    images, actions = load_episode(
        args.manifest_path, args.dataset_root,
        episode_idx=5,
        image_size=args.image_size,
    )

    context_imgs    = images[:args.context_len].unsqueeze(0).to(device)
    context_actions = actions[:args.context_len].unsqueeze(0).to(device)

    print(f"\nDreaming {args.dream_steps} future frames...")
    with torch.no_grad():
        dreamed_imgs, pred_actions = model.dream(
            context_imgs, context_actions, dream_steps=args.dream_steps
        )

    dreamed_imgs = dreamed_imgs[0]
    pred_actions = pred_actions[0].cpu().numpy()

    print(f"[OK] Dreamed {len(dreamed_imgs)} frames")
    print("\nPredicted future actions (steering | throttle | brake):")
    for i, act in enumerate(pred_actions):
        print(f"  t+{i+1:2d}: steer={act[0]:+.3f}  throttle={act[1]:.3f}  brake={act[2]:.3f}")

    print(f"\nGenerating GIF...")
    all_real    = list(images[:args.context_len + args.dream_steps])
    all_dreamed = list(context_imgs[0].cpu()) + list(dreamed_imgs.cpu())

    make_gif(all_real, all_dreamed, pred_actions, args.output, args.context_len)
    print("Done! Left = real frames | Right = dreamed frames")


if __name__ == "__main__":
    main()
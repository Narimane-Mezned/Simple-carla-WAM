import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from envs.carla_dataset import CARLADatasetEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",  type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--steps",         type=int, default=10000)
    parser.add_argument("--image_size",    type=int, default=64)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--seq_len",       type=int, default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--latent_dim",    type=int, default=256)
    parser.add_argument("--deter_dim",     type=int, default=512)
    parser.add_argument("--device",        type=str, default="cpu")
    parser.add_argument("--logdir",        type=str, default="./logdir/carla")
    parser.add_argument("--log_every",     type=int, default=500)
    parser.add_argument("--save_every",    type=int, default=2000)
    return parser.parse_args()


class CARLASequenceDataset(Dataset):
    def __init__(self, dataset_root, manifest_path, image_size=64, seq_len=16):
        import json
        self.dataset_root = dataset_root
        self.image_size   = image_size
        self.seq_len      = seq_len

        episodes = {}
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                frame = json.loads(line)
                ep = frame["source_run"]
                if ep not in episodes: episodes[ep] = []
                episodes[ep].append(frame)

        for ep in episodes:
            episodes[ep].sort(key=lambda x: x["frame_index"])

        self.windows = []
        for ep_name, frames in episodes.items():
            if len(frames) < seq_len: continue
            for start in range(0, len(frames) - seq_len, seq_len // 2):
                self.windows.append(frames[start:start + seq_len])

        total_frames = sum(len(v) for v in episodes.values())
        print(f"[Dataset] {len(episodes)} episodes, {total_frames} frames, "
              f"{len(self.windows)} windows (seq_len={seq_len})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        images, actions = [], []
        for frame in window:
            img_path = os.path.join(self.dataset_root, frame["image_path"])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            img_t = img_t.permute(2, 0, 1)
            images.append(img_t)
            actions.append(torch.tensor([
                float(frame["steering"]),
                float(frame["throttle"]),
                float(frame["brake"]),
            ], dtype=torch.float32))
        return torch.stack(images), torch.stack(actions)


class ConvEncoder(nn.Module):
    def __init__(self, image_size=64, latent_dim=256, depth=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth,       4, 2, 1), nn.SiLU(),
            nn.Conv2d(depth, depth*2, 4, 2, 1), nn.SiLU(),
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

        self.gru = nn.GRUCell(latent_dim + action_dim, deter_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, deter_dim), nn.SiLU(),
            nn.Linear(deter_dim, latent_dim * 2)
        )

        self.post_net = nn.Sequential(
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

        h, z = self.initial_state(B, device)
        posts, priors, states = [], [], []

        for t in range(T):
            gru_input = torch.cat([z, actions[:, t]], dim=-1)
            h = self.gru(gru_input, h)

            prior = self.prior_net(h)
            priors.append(prior)

            post_input = torch.cat([h, encoded_imgs[:, t]], dim=-1)
            post = self.post_net(post_input)
            posts.append(post)

            mu, log_sigma = post.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma.clamp(-4, 4))
            z = mu + sigma * torch.randn_like(sigma)

            states.append(torch.cat([h, z], dim=-1))

        return (
            torch.stack(posts,  dim=1),
            torch.stack(priors, dim=1),
            torch.stack(states, dim=1),
        )

    def imagine(self, h, z, action, steps):
        imagined = []
        for _ in range(steps):
            gru_input = torch.cat([z, action], dim=-1)
            h = self.gru(gru_input, h)
            prior = self.prior_net(h)
            mu, log_sigma = prior.chunk(2, dim=-1)
            sigma = torch.exp(log_sigma.clamp(-4, 4))
            z = mu + sigma * torch.randn_like(sigma)
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
        B, T = images.shape[:2]
        device = images.device

        encoded = self.encoder(images)
        _, _, states = self.rssm(encoded, actions)

        last_state  = states[:, -1]
        h = last_state[:, :self.deter_dim]
        z = last_state[:, self.deter_dim:]
        last_action = actions[:, -1]

        imagined_states = self.rssm.imagine(h, z, last_action, dream_steps)
        imagined_states = imagined_states.permute(1, 0, 2)
        imagined_imgs   = self.decoder(imagined_states)
        pred_actions    = self.action_head(imagined_states)
        return imagined_imgs, pred_actions

def kl_loss(posts, priors):
    post_mu,  post_ls  = posts.chunk(2,  dim=-1)
    prior_mu, prior_ls = priors.chunk(2, dim=-1)

    post_sigma  = torch.exp(post_ls.clamp(-4, 4))
    prior_sigma = torch.exp(prior_ls.clamp(-4, 4))

    kl = (
        torch.log(prior_sigma / post_sigma)
        + (post_sigma**2 + (post_mu - prior_mu)**2) / (2 * prior_sigma**2)
        - 0.5
    )
    return kl.mean()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.logdir, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Steps      : {args.steps}")
    print(f"Image size : {args.image_size}x{args.image_size}")
    print(f"Latent dim : {args.latent_dim}")
    print(f"Deter dim  : {args.deter_dim}")

    print("\nLoading dataset...")
    dataset = CARLASequenceDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        image_size=args.image_size,
        seq_len=args.seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    print(f"[OK] {len(dataset)} sequences ready for training")

    model = WorldModel(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        deter_dim=args.deter_dim,
        action_dim=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] World model: {n_params/1e6:.1f}M parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.steps} steps...")

    step       = 0
    epoch      = 0
    start_time = time.time()

    while step < args.steps:
        epoch += 1
        for images, actions in loader:
            if step >= args.steps:
                break

            images  = images.to(device)
            actions = actions.to(device)

            recon, posts, priors, pred_actions = model(images, actions)

            recon_loss  = ((recon - images) ** 2).mean()
            kl          = kl_loss(posts, priors)
            kl_scaled   = torch.clamp(kl, min=1.0)
            action_loss = ((pred_actions[:, :-1] - actions[:, 1:]) ** 2).mean()
            loss        = recon_loss + kl_scaled + action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()

            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                fps     = step * args.batch_size * args.seq_len / elapsed
                print(
                    f"  Step {step:6d}/{args.steps} | "
                    f"loss {loss.item():.4f} | "
                    f"recon {recon_loss.item():.4f} | "
                    f"kl {kl.item():.4f} | "
                    f"action {action_loss.item():.4f} | "
                    f"{fps:.0f} frames/s"
                )

            if step % args.save_every == 0:
                ckpt = os.path.join(args.logdir, f"checkpoint_{step:06d}.pt")
                torch.save({
                    "step":      step,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args":      vars(args),
                }, ckpt)
                print(f"  [Saved] {ckpt}")

    final_ckpt = os.path.join(args.logdir, "checkpoint_final.pt")
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args":      vars(args),
    }, final_ckpt)

    elapsed = time.time() - start_time
    print(f"\nTraining complete! {step} steps in {elapsed/60:.1f} min")
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
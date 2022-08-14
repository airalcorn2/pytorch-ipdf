import numpy as np
import torch

from ipdf import IPDF
from symmetric_solids_dataset import SymmetricSolidsDataset, SYMSOL_I
from torch import optim
from torch.utils.data import DataLoader

DATA_DIR = "symmetric_solids"
NEG_SAMPLES = 4095
PARAMS_F = "best_params.pth"
DEVICE = "cuda:0"
BATCH_SIZE = 64  # Paper is 128, but that was too large for my GPU.
NUM_WORKERS = 2
# See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
# and Section S8.
LR = 1e-4
WARMUP_STEPS = 1000
ITERATIONS = 100000


def main():
    train_dataset = SymmetricSolidsDataset(DATA_DIR, "train", SYMSOL_I, NEG_SAMPLES)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    valid_dataset = SymmetricSolidsDataset(DATA_DIR, "test", SYMSOL_I, NEG_SAMPLES)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    device = "cuda:0"
    model = IPDF().to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    step = 0
    best_valid_loss = float("inf")
    while True:
        if step > ITERATIONS:
            break

        model.train()
        for (imgs, Rs_fake_Rs) in train_loader:
            # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/train.py#L160.
            step += 1
            warmup_factor = min(step, WARMUP_STEPS) / WARMUP_STEPS
            decay_step = max(step - WARMUP_STEPS, 0) / (ITERATIONS - WARMUP_STEPS)
            new_lr = LR * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2
            for g in optimizer.param_groups:
                g["lr"] = new_lr

            probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
            loss = -torch.log(probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step > ITERATIONS:
                break

        model.eval()
        valid_loss = 0.0
        n_valid = 0
        with torch.no_grad():
            for (imgs, Rs_fake_Rs) in valid_loader:
                probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
                loss = -torch.log(probs).mean()
                valid_loss += loss.item()
                n_valid += 1

        valid_loss /= n_valid
        print(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PARAMS_F)


if __name__ == "__main__":
    main()

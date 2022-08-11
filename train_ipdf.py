import numpy as np
import torch

from ipdf import IPDF
from symmetric_solids_dataset import SymmetricSolidsDataset, SYMSOL_I
from torch import optim
from torch.utils.data import DataLoader


def main():
    data_dir = "symmetric_solids"
    neg_samples = 4095
    batch_size = 64  # Paper is 128, but that was too large for my GPU.
    train_dataset = SymmetricSolidsDataset(data_dir, "train", SYMSOL_I, neg_samples)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valid_dataset = SymmetricSolidsDataset(data_dir, "test", SYMSOL_I, neg_samples)
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, num_workers=2
    )

    # See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
    # and Section S8.
    lr = 1e-4
    warmup_steps = 1000
    iterations = 100000
    step = 0

    device = "cuda:0"
    model = IPDF().to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float("inf")

    while True:
        if step > iterations:
            break

        model.train()
        for (imgs, Rs_fake_Rs, labels) in train_loader:
            probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
            loss = -torch.log(probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step > iterations:
                break

            # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/train.py#L160.
            warmup_factor = min(step, warmup_steps) / warmup_steps
            decay_step = max(step - warmup_steps, 0) / (iterations - warmup_steps)
            new_lr = lr * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2
            for g in optimizer.param_groups:
                g["lr"] = new_lr

        model.eval()
        valid_loss = 0.0
        n_valid = 0
        with torch.no_grad():
            for (imgs, Rs_fake_Rs, labels) in valid_loader:
                probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
                loss = -torch.log(probs).mean()
                valid_loss += loss.item()
                n_valid += 1

        valid_loss /= n_valid
        print(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_params.pth")


if __name__ == "__main__":
    main()

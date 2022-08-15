import healpy as hp
import numpy as np
import torch

from ipdf import IPDF
from scipy.spatial.transform import Rotation
from symmetric_solids_dataset import SymmetricSolidsDataset, SYMSOL_I
from torch.utils.data import DataLoader
from train import DATA_DIR, DEVICE, PARAMS_F

NUMBER_QUERIES = 2000000
BATCH_SIZE = 2**18


def generate_healpix_grid(recursion_level=None, size=None):
    # See: # https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L380.
    # I replaced TensorFlow functions with functions from SciPy and NumPy.
    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)

    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    polars = np.arccos(s2_points[:, 2])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)

    R1s = Rotation.from_euler("X", azimuths).as_matrix()
    R2s = Rotation.from_euler("Z", polars).as_matrix()
    R3s = Rotation.from_euler("X", tilts).as_matrix()

    Rs = np.einsum("bij,tjk->tbik", R1s @ R2s, R3s).reshape(-1, 3, 3)
    return Rs


def main():
    # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L272
    # and Section 3.4/Figure 2.
    grid_sizes = 72 * 8 ** np.arange(7)
    size = grid_sizes[np.argmin(np.abs(np.log(NUMBER_QUERIES) - np.log(grid_sizes)))]
    R_grid = generate_healpix_grid(size=size)
    V = np.pi**2 / len(R_grid)

    model = IPDF().to(DEVICE)
    model.load_state_dict(torch.load(PARAMS_F))
    model.eval()

    valid_dataset = SymmetricSolidsDataset(DATA_DIR, "test", SYMSOL_I, 1)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1)

    valid_loss = 0.0
    n_batches = int(np.ceil(len(R_grid) / BATCH_SIZE))
    with torch.no_grad():
        for (imgs, Rs_fake_Rs) in valid_loader:
            # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L154.
            R = Rs_fake_Rs[0, 0].reshape(3, 3).float().to(DEVICE)
            R_delta = R_grid[0].T @ R
            R_grid_new = R_grid @ R_delta
            scores = []
            for batch_idx in range(n_batches):
                start = batch_idx * BATCH_SIZE
                end = start + BATCH_SIZE
                R_batch = R_grid_new[start:end].reshape(1, -1, 9)
                scores.append(model.get_scores(imgs.to(DEVICE), R_batch.to(DEVICE)))

            scores = torch.cat(scores).flatten()
            prob = 1 / V * torch.softmax(scores, 0)[0]
            loss = -torch.log(prob)
            valid_loss += loss.item()

    valid_loss /= len(valid_dataset)
    print(valid_loss)


if __name__ == "__main__":
    main()

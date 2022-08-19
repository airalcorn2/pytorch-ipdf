import healpy as hp
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg
import torch

from ipdf import IPDF
from scipy.spatial.transform import Rotation
from symmetric_solids_dataset import SymmetricSolidsDataset, SYMSOL_I
from torch.utils.data import DataLoader
from train import DATA_DIR, DEVICE, PARAMS_F

NUMBER_QUERIES = 2000000


def compare_rotations():
    # Replicating code here: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L380
    # with SciPy and NumPy functions.
    recursion_level = 1
    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    polars = np.arccos(s2_points[:, 2])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)

    # IPDF.
    zeros = np.zeros(number_pix)
    R1s_tf = tfg.rotation_matrix_3d.from_euler(np.stack([azimuths, zeros, zeros], 1))
    R2s_tf = tfg.rotation_matrix_3d.from_euler(np.stack([zeros, zeros, polars], 1))
    R1_R2s_tf = R1s_tf @ R2s_tf
    Rs_tf = []
    for tilt in tilts:
        R3_tf = tfg.rotation_matrix_3d.from_euler([[tilt, 0, 0]])
        R1_R2s_R3_tf = R1_R2s_tf @ R3_tf
        Rs_tf.append(R1_R2s_R3_tf)

    Rs_tf = np.concatenate(Rs_tf)

    # Mine.
    R1s_sp = Rotation.from_euler("X", azimuths).as_matrix()
    R2s_sp = Rotation.from_euler("Z", polars).as_matrix()
    R3s_sp = Rotation.from_euler("X", tilts).as_matrix()
    Rs_sp = np.einsum("bij,tjk->tbik", R1s_sp @ R2s_sp, R3s_sp).reshape(-1, 3, 3)

    assert np.all(np.isclose(Rs_tf, Rs_sp))


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

    model = IPDF().to(DEVICE)
    model.load_state_dict(torch.load(PARAMS_F))
    model.eval()

    valid_dataset = SymmetricSolidsDataset(DATA_DIR, "test", SYMSOL_I, 1)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1)

    valid_loss = 0.0
    with torch.no_grad():
        for (imgs, Rs_fake_Rs) in valid_loader:
            # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L154.
            R = Rs_fake_Rs[0, 0].reshape(3, 3).float().to(DEVICE)
            R_delta = R_grid[0].T @ R
            R_grid_new = (R_grid @ R_delta).reshape(1, -1, 9)
            probs = model(imgs.to(DEVICE), R_grid_new.to(DEVICE))
            loss = -torch.log(probs).mean()
            valid_loss += loss.item()

    valid_loss /= len(valid_dataset)
    print(valid_loss)


if __name__ == "__main__":
    main()

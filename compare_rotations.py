# Replicating code here: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L380
# with SciPy and NumPy functions.

import healpy as hp
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg

from scipy.spatial.transform import Rotation

recursion_level = 1
number_per_side = 2**recursion_level
number_pix = hp.nside2npix(number_per_side)
s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
s2_points = np.stack([*s2_points], 1)

azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
polars = np.arccos(s2_points[:, 2])
tilts = np.linspace(0, 2 * np.pi, 10, endpoint=False)

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

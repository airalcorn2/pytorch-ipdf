import numpy as np
import os
import shutil
import tensorflow_datasets as tfds

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms

SYMSOL_I = {"tet", "cube", "icosa", "cone", "cyl"}
SYMSOL_II = {"tetX", "cylO", "sphereX"}


class SymmetricSolidsDataset(Dataset):
    # Adapted from: https://www.tensorflow.org/datasets/catalog/symmetric_solids.
    def __init__(self, data_dir, dataset, subset, neg_samples):
        assert neg_samples > 0

        self.data_dir = data_dir
        self.dataset = dataset
        try:
            rotations = np.load(f"{data_dir}/{dataset}/rotations.npz")
        except FileNotFoundError:
            temp_dir = f"{data_dir}_temp"
            _ = tfds.load("symmetric_solids", data_dir=temp_dir)
            zip_dir = os.listdir(f"{temp_dir}/downloads/extracted")[0]
            shutil.move(f"{temp_dir}/downloads/extracted/{zip_dir}", data_dir)
            shutil.rmtree(temp_dir)
            rotations = np.load(f"{data_dir}/{dataset}/rotations.npz")

        Rs = {}
        for shape in rotations:
            Rs[shape] = rotations[shape][:, 0]

        self.Rs = Rs

        img_fs = os.listdir(f"{data_dir}/{dataset}/images")
        img_fs = [img_f for img_f in img_fs if img_f.split("_")[0] in subset]
        self.img_fs = img_fs

        self.neg_samples = neg_samples
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.img_fs)

    def __getitem__(self, idx):
        # See: https://github.com/tensorflow/datasets/blob/3db2bf78a2b25fd55c2a9c1e4d59db8af05bb94a/tensorflow_datasets/image/symmetric_solids/symmetric_solids.py#L141-L150.
        img_f = self.img_fs[idx]
        (shape_name, image_index) = img_f.split(".")[0].split("_")
        image_index = int(image_index.split(".")[0])
        R = self.Rs[shape_name][image_index]
        img = Image.open(f"{self.data_dir}/{self.dataset}/images/{img_f}")
        img = self.preprocess(img)

        fake_Rs = Rotation.random(self.neg_samples).as_matrix()
        R_fake_Rs = np.concatenate([R[None], fake_Rs])

        return (img, R_fake_Rs.reshape(-1, 9))

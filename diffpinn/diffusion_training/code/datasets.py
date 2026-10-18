from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import random

def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    dataset = BasicDataset(
        all_files,
        class_cond=class_cond,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def normalizer_vel(x, dmin=1.5, dmax=4.5):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def denormalizer_vel(x, dmin=1.5, dmax=4.5):
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


class BasicDataset(Dataset):
    def __init__(self, paths, class_cond=False):
        super().__init__()
        self.local_dataset = paths
        self.class_cond = class_cond

    def __len__(self):
        return len(self.local_dataset)

    def __getitem__(self, idx):
        path = self.local_dataset[idx]

        dict = sio.loadmat(path)
        v = dict['v']
        latent_vector = dict['latent_vector']
        x_coords = np.squeeze(dict['x'])
        z_coords = np.squeeze(dict['z'])
        sx_coords = np.squeeze(dict['sx'])

        selected_v = np.squeeze(normalizer_vel(v))
        latent_vector = np.array(latent_vector, dtype=np.float32)
        x_coords = np.array(0.1*x_coords, dtype=np.float32)
        z_coords = np.array(0.1*z_coords, dtype=np.float32)
        sx_coords = np.array(0.1*sx_coords, dtype=np.float32)
        selected_v = np.array(selected_v, dtype=np.float32)

        cls_label = np.array(np.zeros((1,)), dtype=np.int64)
        out_dict = {}
        return latent_vector, x_coords, z_coords, sx_coords, selected_v, cls_label, out_dict

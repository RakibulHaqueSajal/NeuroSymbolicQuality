from collections.abc import Hashable, Mapping
import copy

import nrrd
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform

from monai.transforms import NormalizeIntensity
from skimage.transform import resize

def load_nrrd_data(file_path):
    """Load the NRRD file and return the data array."""
    data, _ = nrrd.read(file_path)
    return np.expand_dims(data, axis=0)  # Add channel dimension


def find_bounding_box(label_data):
    """Find the bounding box for the blood pool regions."""
    xs, ys, zs = np.where(label_data == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    return x_min, x_max, y_min, y_max, z_min, z_max


def crop_patches(volume, patch_size, stride):
    """Crop overlapping patches from the volume."""
    patches = []

    for x in range(0, volume.shape[0] - patch_size[0] + 1, stride[0]):  # 0, 64, 32
        for y in range(0, volume.shape[1] - patch_size[1] + 1, stride[1]):  # 0, 64, 32
            # 0, 64, 32
            for z in range(0, volume.shape[2] - patch_size[2] + 1, stride[2]):
                # 0:64, 0:64, 0:64
                patch = volume[
                    x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]
                ]
                patch = np.expand_dims(patch, axis=0)  # Add channel dimension
                patches.append(patch)

    if len(patches) == 0:
        msg = "No patches were generated"
        raise ValueError(msg)

    return np.stack(patches, axis=0)  # (n_patches, 64, 64, 7)


def extract_and_crop_patches(
    mri_data, label_data, patch_size=(64, 64, 7), stride=(32, 32, 4), enlarge_xy=20
):
    # shape of label_data:
    # Find the bounding box for the blood pool
    x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(label_data.squeeze(0))

    squeezed_mri_data = mri_data.squeeze(0)
    # Expand x, y dimensions to make the bounding box larger
    x_min = max(0, x_min - enlarge_xy)
    x_max = min(squeezed_mri_data.shape[0] - 1, x_max + enlarge_xy)

    y_min = max(0, y_min - enlarge_xy)
    y_max = min(squeezed_mri_data.shape[1] - 1, y_max + enlarge_xy)

    # Extract the sub-volume containing the blood pool
    sub_mri_data = squeezed_mri_data[
        x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1
    ]

    # Crop patches from the sub-volume
    patches = crop_patches(sub_mri_data, patch_size, stride)

    return patches, sub_mri_data


class LoadNrrd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for keys in self.keys:
            d[keys] = load_nrrd_data(d[keys])

        return d

class LoadAxialViewLAMedianSlices(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        mri_data = d['image'] # (1, 254, 190, 36)
        label_data = d['la_label']
        
        axial_images, axial_labels = [], []

        for i in range(mri_data.shape[3]):
            if label_data[0, :, :, i].sum() == 0:
                continue

            axial_images.append(mri_data[0, :, :, i].transpose(1, 0))
            axial_labels.append(label_data[0, :, :, i].transpose(1, 0))

        axial_images = np.stack(axial_images, axis=0) # (N, 256, 190)
        axial_labels = np.stack(axial_labels, axis=0) # (N, 256, 190)

        # Get the middle slice index
        middle_idx = len(axial_images) // 2

        # Calculate the range of slices to include (5 slices before and after the middle slice)
        start_idx = max(0, middle_idx - 3)
        end_idx = min(len(axial_images), middle_idx + 3)

        # Select the slices
        axial_images = axial_images[start_idx:end_idx]
        axial_labels = axial_labels[start_idx:end_idx]

        d['image'] = axial_images # (10, 256, 190)
        d['la_label'] = axial_labels # (10, 256, 190)
        
        return d

class ResizeAxialViewImages(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None, spatial_size=(256, 256)
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spatial_size = spatial_size

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:

        d = dict(data)

        axial_images = d['image'] # (30, 256, 190)
        axial_labels = d['la_label']

        axial_images = np.stack([resize(image, self.spatial_size) for image in axial_images])
        axial_labels = np.stack([resize(label, self.spatial_size, order=0) for label in axial_labels])

        # add channel dimension
        axial_images = np.expand_dims(axial_images, axis=1) # (30, 1, 256, 256)
        axial_labels = np.expand_dims(axial_labels, axis=1) # (30, 1, 256, 256)

        d['image'] = axial_images
        d['la_label'] = axial_labels

        return d

class NormalizeAxialImages(MapTransform):
        def __init__(
            self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None, nonzero=False,
        ) -> None:
            super().__init__(keys, allow_missing_keys)
            self.normalize_intensity = NormalizeIntensity(nonzero=nonzero)
    
        def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
    
            d = dict(data)

            axial_images = d['image']

            d['original_axial_images'] = copy.deepcopy(axial_images)
    
            d['image'] = torch.stack([self.normalize_intensity(image) for image in axial_images])

            del d['original_axial_images']
            
            return d

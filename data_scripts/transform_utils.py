from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    Resized,
    Spacingd,
    ToTensord,
    EnsureTyped,
    RandAffined,
)
import torch

#from config import gpu_id
from data_scripts.data_utils import LoadAxialViewLAMedianSlices, LoadNrrd, NormalizeAxialImages, ResizeAxialViewImages


def get_transforms(model_name, CONFIG):
    if model_name == "AFibQCNet":
        train_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                Spacingd(
                    keys=["image", "la_label"],
                    pixdim=(
                        CONFIG["spacing"][0],
                        CONFIG["spacing"][1],
                        CONFIG["spacing"][2],
                    ),
                    # Why not nearest for first index?
                    mode=("trilinear", "nearest"),
                ),
                Resized(
                    keys=["image", "la_label"],
                    spatial_size=(
                        CONFIG["input_shape"][0],
                        CONFIG["input_shape"][1],
                        CONFIG["input_shape"][2],
                    ),
                    mode=("trilinear", "nearest"),
                ),  # Resize the image and label to the given spatial size
                RandRotate90d(
                    keys=["image", "la_label"], prob=0.5
                ),  # Randomly flip the image and label
                RandFlipd(
                    keys=["image", "la_label"], prob=0.5, spatial_axis=0
                ),  # Randomly flip the image and label
                RandFlipd(
                    keys=["image", "la_label"], prob=0.5, spatial_axis=1
                ),  # Randomly flip the image and label
                RandFlipd(
                    keys=["image", "la_label"], prob=0.5, spatial_axis=2
                ),  # Randomly flip the image and label
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeIntensityd(keys=["image"], nonzero=True),
                # Convert the input data to PyTorch Tensors
                ToTensord(keys=["image", "la_label"]),
            ]
        )

        val_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                Spacingd(
                    keys=["image", "la_label"],
                    pixdim=(
                        CONFIG["spacing"][0],
                        CONFIG["spacing"][1],
                        CONFIG["spacing"][2],
                    ),
                    # Why not nearest for first index?
                    mode=("trilinear", "nearest"),
                ),
                Resized(
                    keys=["image", "la_label"],
                    spatial_size=(
                        CONFIG["input_shape"][0],
                        CONFIG["input_shape"][1],
                        CONFIG["input_shape"][2],
                    ),
                    mode=("trilinear", "nearest"),
                ),  # Resize the image and label to the given spatial size
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeIntensityd(keys=["image"], nonzero=True),
                # Convert the input data to PyTorch Tensors
                ToTensord(keys=["image", "la_label"]),
            ]
        )
    elif model_name == 'AFibQCNet2D':
        train_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                LoadAxialViewLAMedianSlices(keys=["image", "la_label"]),
                ResizeAxialViewImages(keys=["image", "la_label"], spatial_size=(
                    CONFIG['input_shape'][0], CONFIG['input_shape'][1])),  # Resize the image and label to the given spatial size
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeAxialImages(keys=["image"], nonzero=False),
                # RandRotated(keys=["image"], prob=0.8, range_x=np.pi / 4, keep_size=True), # Randomly rotate the image and label
                # RandFlipd(keys=["image"], prob=0.8), # Randomly flip the image and label
                RandAffined(keys=["image"], prob=0.8, scale_range=(0.2, 0.2), mode='bilinear', padding_mode='border', translate_range=(0.1, 0.1)), # Randomly flip the image and label
                ToTensord(keys=["image", "labels"], dtype=torch.float32),
                EnsureTyped(keys=["image","labels"],
                            device=f'cuda',
                            dtype=[torch.float32, torch.float32],
                            track_meta=False), # Ensure the input data to be a PyTorch Tensor or numpy array
            ]
        )
        val_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                LoadAxialViewLAMedianSlices(keys=["image", "la_label"]),
                ResizeAxialViewImages(keys=["image", "la_label"], spatial_size=(
                    CONFIG['input_shape'][0], CONFIG['input_shape'][1])),  # Resize the image and label to the given spatial size
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeAxialImages(keys=["image"], nonzero=False),
                ToTensord(keys=["image", "labels"], dtype=torch.float32),
                EnsureTyped(keys=["image","labels"],
                            device=f'cuda',
                            dtype=[torch.float32, torch.float32],
                            track_meta=False), # Ensure the input data to be a PyTorch Tensor or numpy array
            ]
        )
    return train_transform, val_transform

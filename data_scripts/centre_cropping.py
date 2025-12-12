import os
import json
import numpy as np
import nrrd

def crop_with_dynamic_margin(image, mask, desired_size=(283,219,45)):
    assert image.shape == mask.shape, "Image and mask shapes must match"

    # The format is assumed to be (X, Y, Z)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Segmentation mask is empty.")

    # Determine bounding box without margin
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0) + 1

    # Current cropped size
    current_x, current_y, current_z = xmax - xmin, ymax - ymin, zmax - zmin
    desired_x, desired_y, desired_z = desired_size

    # Adjust margins if the crop is smaller than desired size
    if current_y < desired_y:
        y_diff = desired_y - current_y
        ymin = max(0, ymin - y_diff // 2)
        ymax = min(image.shape[1], ymax + (y_diff - y_diff // 2))

    if current_x < desired_x:
        x_diff = desired_x - current_x
        xmin = max(0, xmin - x_diff // 2)
        xmax = min(image.shape[0], xmax + (x_diff - x_diff // 2))

    # Crop the image with adjusted margins (X and Y only)
    cropped = image[xmin:xmax, ymin:ymax, zmin:zmax]

    # Handle Z-dimension padding if needed
    cropped_z_size = cropped.shape[2]
    print(f"Cropped Z size: {cropped_z_size}")
    print(f"Desired Z size: {desired_z}")
    
    if cropped_z_size < desired_z:
        # Calculate padding needed
        z_diff = desired_z - cropped_z_size
        pad_before = z_diff // 2
        pad_after = z_diff - pad_before

        # Apply zero-padding along Z axis only (axis=2)
        cropped = np.pad(cropped, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)

    # Return the cropped and padded image
    return cropped, (cropped.shape[0], cropped.shape[1], cropped.shape[2])

def process_all_folders(root_dir, label_file, margin=30):
    max_dims_cropped = (0, 0, 0)
    max_dims_original = (0, 0, 0)
    
    # Load the JSON file with labels
    with open(label_file, 'r') as f:
        labels = json.load(f)

    for subfolder in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue

        # Check if the folder is in the label file
        if subfolder not in labels:
            print(f"[!] {subfolder} not found in label file, skipping.")
            continue
        
        label_info = labels[subfolder]

        # Skip folders with label = 0
        if label_info["label"] == 0:
            print(f"[!] {subfolder} has label 0, skipping.")
            continue

        image_path = os.path.join(sub_path, "data.nrrd")
        mask_path = os.path.join(sub_path, "shrinkwrap.nrrd")

        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"[!] Missing files in {subfolder}, skipping.")
            continue

        image, image_header = nrrd.read(image_path)
        mask, _ = nrrd.read(mask_path)

        # Apply margin to the mask
        try:
            cropped, dims = crop_with_dynamic_margin(image, mask, desired_size=(283,219,45))
        except Exception as e:
            print(f"[!] Error in {subfolder}: {e}")
            continue

        # Track the largest original and cropped dimensions
        max_dims_original = tuple(max(m, d) for m, d in zip(max_dims_original, image.shape))
        max_dims_cropped = tuple(max(m, d) for m, d in zip(max_dims_cropped, dims))

        # Save the cropped image in the same folder with a new name
        cropped_image_path = os.path.join(sub_path, "cropped_data.nrrd")
        nrrd.write(cropped_image_path, cropped, header=image_header)

        print(f"[✓] {subfolder} | Original: {image.shape} → Mask: {mask.shape} → Cropped: {cropped.shape} | Saved to {cropped_image_path}")

    print("\n=== Max Dimensions Across All Folders ===")
    print(f"Original Image (X, Y, Z): {max_dims_original}")
    print(f"Cropped Image (X, Y, Z): {max_dims_cropped}")

if __name__ == "__main__":
    root_dir = "/usr/sci/scratch/arefeen_sultan/dataset/afib_db"
    label_file = "/usr/sci/scratch/arefeen_sultan/dataset/labels.json"
    process_all_folders(root_dir, label_file, margin=30)

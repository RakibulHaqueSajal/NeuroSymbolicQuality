import os
import json
import numpy as np
import nrrd

def crop_with_margin(image, mask, margin=30):
    assert image.shape == mask.shape, "Image and mask shapes must match"

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Segmentation mask is empty.")

    # Determine bounding box without margin (assuming X, Y, Z format)
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0) + 1

    # Only apply margin to X and Y axes, not Z (Depth)
    ymin = max(ymin - margin, 0)
    xmin = max(xmin - margin, 0)
    ymax = min(ymax + margin, image.shape[1])
    xmax = min(xmax + margin, image.shape[0])

    # Crop the image with margin applied to X and Y only
    cropped = image[xmin:xmax, ymin:ymax, zmin:zmax]
    return cropped, (xmax - xmin, ymax - ymin, zmax - zmin)

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

        try:
            cropped, dims = crop_with_margin(image, mask, margin=margin)
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
    print("Done")
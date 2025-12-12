import json
import os
import re
import copy
from tqdm import tqdm


def get_patient_records_monai(filenames, data_category='train', qc_dict_json=None, classification='binary', require_concept_labels=False):
    patients = []

    for subject_id in tqdm(range(len(filenames)), desc=f'Loading {data_category} data', unit='subject'):
        subject_name = filenames[subject_id].split('/')[-1]
        # image_name = filenames[subject_id] + '/data.nii.gz'
        image_name = filenames[subject_id] + '/data.nrrd'
        # la_label_name = filenames[subject_id] + '/shrinkwrap.npy'
        la_label_name = filenames[subject_id] + '/shrinkwrap.nrrd'
        qc_labels = copy.deepcopy(qc_dict_json[subject_name]['label'])
        
        if qc_labels == 0:
            raise ValueError(f"Quality control label is 0 for {subject_name}")

        if require_concept_labels:
            # Check if individual concept labels exist
            has_sharpness = 'sharpness' in qc_labels
            has_myocardium_nulling = 'myocardium_nulling' in qc_labels
            has_enhancement = 'enhancement_of_aorta_and_valves' in qc_labels
            
            # Only include if all three concept labels are present
            if not (has_sharpness and has_myocardium_nulling and has_enhancement):
                continue  # Skip this patient if concepts are missing

        if classification == 'binary':
            for key in qc_labels.keys():
                qc_labels[key] = 0.0 if qc_labels[key] <= 2.0 else 1.0
        elif classification == 'multiclass':
            for key in qc_labels.keys():
                qc_labels[key] -= 1  # Assuming labels are 1-indexed


        data = {'image': image_name, 
                'la_label': la_label_name,
                'labels': qc_labels,
                'p_id': subject_name,
                }

        patients.append(data)

    return patients



def get_50_scans_labeled(file_path, qc_dict_json_path=None, labelers_list=None):
    """
    Reads .nrrd files, matches image and mask files, and processes labelers to create patient records.

    Args:
        file_path (str): Directory containing the .nrrd files.
        qc_dict_json_path (str): Path to the JSON file containing quality control data.
        labelers_list (list): List of labelers in a specific order for one-hot encoding.

    Returns:
        list: List of patient records with metadata.
    """
    patients = []

    # List all files in the directory
    files = os.listdir(file_path)

    with open(qc_dict_json_path, 'r') as f:
        qc_dict_json = json.load(f)

    # Separate image files
    image_files = [f for f in files if re.match(r"IQ\d+\.nrrd", f)]

    # Process each image file
    for image_file in image_files:
        # Process labelers
        image_file_key = image_file.split('.nrrd')[0]  # Extract key without .nrrd
        if image_file_key not in qc_dict_json:
            continue

        image_id = re.search(r"IQ(\d+)\.nrrd", image_file).group(1)  # Extract numeric ID
        image_path = os.path.join(file_path, image_file)

        # Construct the corresponding mask file path
        mask_file = f"Segmentation_{image_id}B_LA.nrrd"
        mask_path = os.path.join(file_path, mask_file)

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            continue

        labelers_from_json = qc_dict_json[image_file_key]['labeler']

        for labeler_name in labelers_list:
            if labeler_name not in labelers_from_json:
                continue

            # Assign qc_labels
            qc_labels = labelers_from_json[labeler_name]

            # Create one-hot vector for the labeler
            labeler_onehot = [1 if labeler == labeler_name else 0 for labeler in labelers_list]

            # Create a unique patient ID
            unique_id = f"{image_id}_{labeler_name}"

            # Add matched files and metadata to the patients list
            patients.append({
                'image': image_path,
                'la_label': mask_path,
                'labels': qc_labels,
                'p_id': unique_id,
                'labeler_onehot': labeler_onehot,
                'labeler_name': labeler_name
            })

    return patients
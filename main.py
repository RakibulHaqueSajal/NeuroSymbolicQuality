import glob
import json
import os
import random

import numpy as np
import torch
from comet_ml import Experiment
#from dotenv import load_dotenv
from monai.data import CacheDataset, Dataset
from sklearn.model_selection import train_test_split

from config import CONFIG, MODEL_NAME, disable_comet
from data_scripts.qc_dataset import get_patient_records_monai
from data_scripts.transform_utils import get_transforms
from early_stopping import EarlyStopping
from model_scripts.qc_models import QCModel2D, QCModel3D ,QCModel2DWithConcept
from trainer_scripts.qc_trainers import AFibQCMonaiTrainer, AFibQCSliceMonaiTrainer

# Set the seed for reproducibility
torch.cuda.manual_seed_all(CONFIG["seed"])  # type: ignore
random.seed(CONFIG["seed"])  # type: ignore
np.random.seed(CONFIG["seed"])  # type: ignore
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed(CONFIG["seed"])  # type: ignore

#load_dotenv()

def get_qc_scores(data_files, qc_dict_json_path, require_concept_labels=False):
    """
    Get the quality scores for the data files.

    Args:
        data_files (list): List of data file paths.
        qc_dict_json_path (str): Path to the JSON file containing quality scores.
        require_concept_labels (bool): If True, only include files with sharpness, myocardium_nulling, 
                                       and enhancement_of_aorta_and_valves labels. Default is False.
        radiomics_csv_path (str): Path to the radiomics features CSV file. If provided, only include
                                  patients with radiomics features.

    Returns:
        tuple: (qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files)

    """
    with open(qc_dict_json_path, 'r') as f:
        qc_dict = json.load(f)

    qc_scores = []
    qc_dict_scores = {}
    labeled_data_files = []
    unlabeled_data_files = []
    data_files_with_labels = []

    for data_file in data_files:
        data_file_name = data_file.split("/")[-1]

        # if qc_dict[data_file_name]["label"] != 0 and qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] not in [2,3]:
        if qc_dict[data_file_name]["segmented_region_indices"] == "wrong":
            continue

        if qc_dict[data_file_name]["label"] != 0:

            # Check for concept labels only if required
            if require_concept_labels:
                has_sharpness = "sharpness" in qc_dict[data_file_name]["label"].keys()
                has_myocardium_nulling = "myocardium_nulling" in qc_dict[data_file_name]["label"].keys()
                has_enhancement = "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"].keys()

                if not (has_sharpness and has_myocardium_nulling and has_enhancement):
                    continue

                # Only process concept labels if they exist
                if "sharpness" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["sharpness"] == 5:
                        qc_dict[data_file_name]["label"]["sharpness"] = 4
                if "myocardium_nulling" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["myocardium_nulling"] == 5:
                        qc_dict[data_file_name]["label"]["myocardium_nulling"] = 4
                if "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"]:
                    if qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"] == 5:
                        qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"] = 4

            labeled_data_files.append(data_file)
            data_files_with_labels.append(data_file)

            if qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] == 5:
                qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] = 4

            qc_scores.append(qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"])
            
            if data_file_name not in qc_dict_scores:
                qc_dict_scores[data_file_name] = {"label": {}}

            if CONFIG['classification'] == 'binary':
                qc_dict_scores[data_file_name]["label"]["quality_for_fibrosis_assessment"] = 0.0 if qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"] <= 2.0 else 1.0
                
                if require_concept_labels:
                    # Only process concept labels if they exist
                    if "sharpness" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["sharpness"] = 0.0 if qc_dict[data_file_name]["label"]["sharpness"] <= 2.0 else 1.0
                    if "myocardium_nulling" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["myocardium_nulling"] = 0.0 if qc_dict[data_file_name]["label"]["myocardium_nulling"] <= 2.0 else 1.0
                    if "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["enhancement_of_aorta_and_valves"] = 0.0 if qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"] <= 2.0 else 1.0                    
            elif CONFIG['classification'] == 'multiclass':
                qc_dict_scores[data_file_name]["label"]["quality_for_fibrosis_assessment"] = qc_dict[data_file_name]["label"]["quality_for_fibrosis_assessment"]
                
                # Only process concept labels if they exist
                if require_concept_labels:
                    if "sharpness" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["sharpness"] = qc_dict[data_file_name]["label"]["sharpness"]
                    if "myocardium_nulling" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["myocardium_nulling"] = qc_dict[data_file_name]["label"]["myocardium_nulling"]
                    if "enhancement_of_aorta_and_valves" in qc_dict[data_file_name]["label"]:
                        qc_dict_scores[data_file_name]["label"]["enhancement_of_aorta_and_valves"] = qc_dict[data_file_name]["label"]["enhancement_of_aorta_and_valves"]
        else:
            unlabeled_data_files.append(data_file)

    return qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files

def get_train_test_split(data_files, test_size, seed, qc_dict_json_path, require_concept_labels=False):
    """
    Split the data files into training, validation, and testing sets based on the quality scores.

    Args:
        data_files (list): List of data file paths.
        test_size (float): The proportion of the data to include in the test split.
        seed (int): Seed for random number generation.
        qc_dict_json_path (str): Path to the JSON file containing quality scores.

    Returns:
        tuple: A tuple containing the training files, validation files, and testing files.

    """
    qc_scores, labeled_data_files, qc_dict_scores, unlabeled_data_files = get_qc_scores(data_files, qc_dict_json_path, require_concept_labels=require_concept_labels)

    train_files, test_files = train_test_split(
        labeled_data_files, test_size=test_size, random_state=seed, stratify=qc_scores
    )

    train_scores_list = get_qc_scores(train_files, qc_dict_json_path, require_concept_labels=require_concept_labels)[0]

    train_files, val_files = train_test_split(
        train_files, test_size=test_size, random_state=seed, stratify=train_scores_list
    )

    train_qc_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    val_qc_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    test_qc_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    train_file_labels = []

    with open(qc_dict_json_path, "r") as f:
        qc_dict = json.load(f)

    for train_file in train_files:
        train_file_name = train_file.split("/")[-1]
        train_qc_scores[
            qc_dict_scores[train_file_name]["label"]["quality_for_fibrosis_assessment"]
        ] += 1

        if CONFIG["classification"] == "binary":
            train_file_labels.append(
                0.0
                if qc_dict_scores[train_file_name]["label"]["quality_for_fibrosis_assessment"]
                <= 2.0
                else 1.0
            )
        elif CONFIG["classification"] == "multiclass":
            train_file_labels.append(
                qc_dict_scores[train_file_name]["label"]["quality_for_fibrosis_assessment"] - 1
            )

    for val_file in val_files:
        val_file_name = val_file.split("/")[-1]
        val_qc_scores[
            qc_dict_scores[val_file_name]["label"]["quality_for_fibrosis_assessment"]
        ] += 1

    for test_file in test_files:
        test_file_name = test_file.split("/")[-1]
        test_qc_scores[
            qc_dict_scores[test_file_name]["label"]["quality_for_fibrosis_assessment"]
        ] += 1

    class_sample_counts = np.bincount(train_file_labels)
    train_class_weight = 1.0 / class_sample_counts
    samples_train_weights = np.array(
        [train_class_weight[int(label)] for label in train_file_labels]
    )

    # Create sample weights

    print("Class sample counts:", class_sample_counts)
    print("No of Train files:", len(train_files))
    print("No of Val files:", len(val_files))
    print("No of Test files:", len(test_files))
    print("Train Class Weight:", train_class_weight)

    print("Train QC Scores:", train_qc_scores)
    print("Val QC Scores:", val_qc_scores)
    print("Test QC Scores:", test_qc_scores)

    # experiment.log_text({'No of Train files': str(len(train_files)), 'No of Val files': str(len(val_files)), 'No of Test files': str(len(test_files))})
    # experiment.log_text({"Train QC Scores": str(train_qc_scores), "Val QC Scores": str(val_qc_scores), "Test QC Scores": str(test_qc_scores)})

    return train_files, val_files, test_files, samples_train_weights, qc_dict_scores, class_sample_counts, unlabeled_data_files

def main(train_transform, val_transform):
    saved_model_name = CONFIG["saved_model_name"]

    data_files = glob.glob(CONFIG["data_path"] + "/*")
    data_files.sort()

    train_files, val_files, test_files, train_class_weight, qc_dict_scores, class_sample_counts, unlabeled_data_files = get_train_test_split(
        data_files,
        test_size=0.2,
        seed=CONFIG["seed"],
        qc_dict_json_path=CONFIG["qc_label_dict"],
        require_concept_labels=CONFIG["require_concept_labels"],
    )

    train_patient_records = get_patient_records_monai(
        train_files,
        data_category="train",
        qc_dict_json=qc_dict_scores,
        classification=CONFIG["classification"],
        require_concept_labels=CONFIG["require_concept_labels"],
    )
    val_patient_records = get_patient_records_monai(
        val_files,
        data_category="val",
        qc_dict_json=qc_dict_scores,
        classification=CONFIG["classification"],
        require_concept_labels=CONFIG["require_concept_labels"],
    )
    test_patient_records = get_patient_records_monai(
        test_files,
        data_category="test",
        qc_dict_json=qc_dict_scores,
        classification=CONFIG["classification"],
        require_concept_labels=CONFIG["require_concept_labels"],
    )

    if MODEL_NAME in ["AFibQCNet", "AFibQCNet2D"]:
        AFibQCDataset_train = CacheDataset(
            data=train_patient_records,
            transform=train_transform,
            cache_rate=1.0,
            num_workers=18,copy_cache=False
        )
        AFibQCDataset_val = CacheDataset(
            data=val_patient_records,
            transform=val_transform,cache_rate=1.0,
            num_workers=10,copy_cache=False
        )
        AFibQCDataset_test = CacheDataset(
            data=test_patient_records,
            transform=val_transform,cache_rate=1.0,
            num_workers=10,copy_cache=False
        )
        
        # AFibQCDataset_train = Dataset(
        #     data=train_patient_records, transform=train_transform
        # )
        # AFibQCDataset_val = Dataset(
        #     data=val_patient_records, transform=val_transform
        # )
        # AFibQCDataset_test = Dataset(
        #     data=test_patient_records, transform=val_transform
        # )

    print(f"AFib dataset created for {MODEL_NAME}")

    print(f"Model saved at: {CONFIG['model_path'] + f'/{saved_model_name}'}")
    early_stopping = EarlyStopping(
        patience=CONFIG["training_patience"],
        verbose=False,
        delta=1e-6,
        path=[CONFIG["model_path"] + f"/{saved_model_name}"],
        score_name="auroc",
        start_epoch=0,
    )

    experiment = Experiment(
        api_key="Vr3vky03wyHWTXJTZ6phb4zEF",
        project_name="quality",
        workspace="rakibulhaq56",
        log_code=True,
        disabled=disable_comet,
    )
    experiment.set_name(f"{MODEL_NAME}_Pretrain_Concept_MLP_Weighted_Final_{CONFIG['concept_weight']}_Semantic_Loss_Weight_{CONFIG['semantic_weight']}_ImplicationLoss_product_norm_learnable_weight_epoch increased")

    experiment.log_code(file_name="trainer_scripts/qc_trainers.py")
    experiment.log_code(file_name="config.py")

    hyper_params = {
        "seed": CONFIG["seed"],
        "stride": CONFIG["stride"] if MODEL_NAME not in ["AFibQCNet", "AFibQCNet2D"] else None,
        "enlarge_xy": CONFIG["enlarge_xy"] if MODEL_NAME not in ["AFibQCNet", "AFibQCNet2D"] else None,
        "log_gradinfo": CONFIG["log_gradinfo"]
        if MODEL_NAME not in ["AFibQCNet", "AFibQCNet2D"]
        else None,
        "batch_size": CONFIG["batch_size"],
        "num_epochs": CONFIG["epochs"],
        "spacing": CONFIG["spacing"],
        "learning_rate": CONFIG["learning_rate"],
    }

    experiment.log_parameters(hyper_params)

    afib_qc_model = None

    if MODEL_NAME == "AFibQCNet":
        if CONFIG["classification"] == "binary":
            afib_qc_model = QCModel3D(
                encoder_name="resnet", n_input_channels=1, num_classes=1, spatial_dims=3
            )
        elif CONFIG["classification"] == "multiclass":
            afib_qc_model = QCModel3D(
                encoder_name="resnet", n_input_channels=1, num_classes=5, spatial_dims=3
            )

        print("AFibQCNet model created")
    elif MODEL_NAME == "AFibQCNet2D":
        if CONFIG["require_concept_labels"]:
           afib_qc_model = QCModel2DWithConcept(
                  encoder_name="resnet", n_input_channels=1, num_classes=4, spatial_dims=2
        )
        else:
            afib_qc_model = QCModel2D(
                encoder_name="resnet", n_input_channels=1, num_classes=4, spatial_dims=2
            )

        print("AFibQCNet2D model created")

    afib_trainer = None

    if MODEL_NAME == "AFibQCNet":
        afib_trainer = AFibQCMonaiTrainer(
            model=afib_qc_model,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["learning_rate"],
            train_dataset=AFibQCDataset_train,
            val_dataset=AFibQCDataset_val,
            test_dataset=AFibQCDataset_test,
            experiment=experiment,
        )
    elif MODEL_NAME == "AFibQCNet2D":
        afib_trainer = AFibQCSliceMonaiTrainer(
            model=afib_qc_model,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            lr=CONFIG["learning_rate"],
            train_dataset=AFibQCDataset_train,
            val_dataset=AFibQCDataset_val,
            test_dataset=AFibQCDataset_test,
            experiment=experiment,
            require_concept_labels=CONFIG["require_concept_labels"],
            just_concept=CONFIG["just_concept"],
            concept_weight=CONFIG["concept_weight"],
            semantic_weight=CONFIG["semantic_weight"],
            require_semantic_loss=CONFIG["require_semantic_loss"]
        )

    print(f"AFib trainer created for {MODEL_NAME}")
    print("Training started")

    afib_trainer.train(early_stopping=early_stopping, train_weights=train_class_weight)

    print(f"Loading the saved model: {CONFIG['model_path'] + f'/{saved_model_name}'}")
    afib_trainer.test(model_save_path=CONFIG["model_path"] + f"/{saved_model_name}")
    print("Testing finished")

    experiment.end()


if __name__ == "__main__":
    train_transform, val_transform = get_transforms(MODEL_NAME, CONFIG)

    main(train_transform=train_transform, val_transform=val_transform)

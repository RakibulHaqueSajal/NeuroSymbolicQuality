import torch

gpu_no = 0
machine_id = "sci"
seed = 0

load_self_sup_model = False
enlarge_xy = 2
#n_epochs=800
n_epochs = 2000
disable_comet = False

patches_dir = "patches_axial_60_32"
patches_dir_3d = "patches_axial_60_32_7"
saved_model_name_self_supervised = (
    "self_supervised_pretrained_baseline_512_1_patches_axial_60_32.pth"
)

# Model configuration for a convolutional neural network
config = {
    "AFibQCNet": {
        # This is the minimum shape among all the images
        "input_shape": (256, 256, 40),
        "crop_input_shape": (256, 256, 40),
        "batch_size": 4,
        "epochs": n_epochs,
        "training_patience": 100,
        "classification": "multiclass",  # 'binary', 'multiclass'
        "seed": seed,
        "learning_rate": 1e-6,
        "weight_decay": 1e-8,
        "spacing": [0.625, 0.625, 2.5],
        "data_path": "dataset/afib_db",
        "qc_label_dict": "dataset/labels.json",
        "model_path": "saved_models",
        "saved_model_name": f"afib_qc_baseline_{machine_id}_{gpu_no}_prtrain_concept.pth",
        "test_size": 0.2,
    },
    "AFibQCNet2D": {
        'input_shape': (256, 256),
        'batch_size': 8,
        'spatial_dims': 2,
        'epochs': n_epochs,
        'fine_tune_epochs': 10,
        'training_patience': 500,
        'load_self_sup_model': load_self_sup_model,
        'saved_model_name_self_supervised': saved_model_name_self_supervised,
        'classification': 'multiclass',  # 'binary', 'multiclass'
        'seed': seed,
        'learning_rate': 1e-4,
        'weight_decay': 3e-2,
        'fine_tune_weight_decay': 1e-1,
        'initial_loss': 'corn_loss',
        'fine_tune_loss': 'cb_corn_loss',
        'require_concept_labels': True,
        'require_semantic_loss':True,
        'log_gradinfo': False,
        'gradient_clipping': False,
        'max_clip_grad': 5,
        'concept_weight':0.8,
        'semantic_weight':0.5,
        'just_concept': False,
        'spacing': [0.625, 0.625, 2.5],
        'data_path': '/uufs/sci.utah.edu/projects/medvic-lab/Arefeen/left_atrium_qc_dataset/afib_db',
        'data_path_50':'/uufs/sci.utah.edu/projects/medvic-lab/Arefeen/left_atrium_qc_dataset/afib_db',
        'qc_label_dict': '/uufs/sci.utah.edu/projects/medvic-lab/Arefeen/left_atrium_qc_dataset/new_surface_area.json',
        'qc_label_dict_50': '/uufs/sci.utah.edu/projects/medvic-lab/Arefeen/left_atrium_qc_dataset/new_surface_area.json',
        'model_path': 'saved_models/pretrain_weighted_concept_mlp_1.0',
        'saved_model_name': f'best.pth',
        'test_size': 0.2,
    },
}

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "AFibQCNet2D"
CONFIG = config[MODEL_NAME]

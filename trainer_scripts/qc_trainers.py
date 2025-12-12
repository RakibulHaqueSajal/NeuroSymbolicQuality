import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import torch
from matplotlib import pyplot as plt
from monai.data import ThreadDataLoader, decollate_batch
from monai.transforms import Activations, Compose
from numpy import copy
from torch import nn, optim
from torch.utils.data import WeightedRandomSampler
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
    MulticlassSpecificity,
)
from tqdm import tqdm

from config import CONFIG, device
from trainer_scripts import base_dir
from trainer_scripts.corn_utils import corn_label_from_logits
from trainer_scripts.losses import cb_corn_loss, corn_loss
from trainer_scripts.regularizers import MaxNorm_via_PGD, Normalizer
import seaborn as sns

from scipy.stats import skew, kurtosis, kendalltau

from util.metrics import ScottsPiQuadratic, accuracy_off1_macro, amae, gmsec, mmae
from util.semantic_loss import *

mpl.use("Agg")  # Use the Agg backend for non-interactive plotting


class AFibQCMonaiTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        epochs,
        lr,
        experiment,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.experiment = experiment

        self.device = device
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=CONFIG["weight_decay"]
        )
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.post_process = Compose([Activations(softmax=True)])

        self.ac, self.f1, self.spec, self.cm, self.auroc, self.roc, self.prcurve = (
            self.get_classification_metrics(classification=CONFIG["classification"])
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def get_classification_metrics(self, classification="binary"):
        if classification == "binary":
            return (
                BinaryAccuracy(threshold=0.5).to(self.device),
                BinaryF1Score(threshold=0.5).to(self.device),
                BinarySpecificity(threshold=0.5).to(self.device),
                BinaryConfusionMatrix(threshold=0.5).to(self.device),
                BinaryAUROC(thresholds=5).to(self.device),
                BinaryROC(thresholds=5).to(self.device),
                BinaryPrecisionRecallCurve(thresholds=5).to(self.device),
            )

        if classification == "multiclass":
            return (
                MulticlassAccuracy(num_classes=5).to(self.device),
                MulticlassF1Score(num_classes=5).to(self.device),
                MulticlassSpecificity(num_classes=5).to(self.device),
                MulticlassConfusionMatrix(num_classes=5).to(self.device),
                MulticlassAUROC(num_classes=5, thresholds=5).to(self.device),
                MulticlassROC(num_classes=5, thresholds=5).to(self.device),
                MulticlassPrecisionRecallCurve(num_classes=5, thresholds=5).to(
                    self.device
                ),
            )
        return None

    def train_one_epoch(self, dataloader, epoch_number):
        self.model.train()

        self.ac.reset()
        self.f1.reset()
        self.auroc.reset()
        self.roc.reset()

        running_loss = 0.0

        predicted_logits_of_overall_train = []
        true_labels_of_overall_train = []

        epoch_iterator = tqdm(
            dataloader,
            desc="Phase: Train",
            total=len(dataloader),
            unit="batch",
            dynamic_ncols=True,
        )

        for batch in epoch_iterator:
            inputs, _masks, labels, _p_id, labeler = (
                batch["image"].to(self.device),
                batch["la_label"].to(self.device),
                batch["labels"],
                batch["p_id"],
                batch["labeler_onehot"].to(self.device),
            )
            labels = {
                key: value.to(self.device).unsqueeze(1) for key, value in labels.items()
            }

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)

                loss = self.ce_loss(
                    input=outputs,
                    target=labels["quality_for_fibrosis_assessment"].long().squeeze(),
                )

                loss.backward()
                self.optimizer.step()

            predicted_overall = [self.post_process(i) for i in decollate_batch(outputs)]

            predicted_logits_of_overall_train.extend(predicted_overall)
            true_labels_of_overall_train.extend(
                decollate_batch(labels["quality_for_fibrosis_assessment"])
            )

            running_loss += loss.item()

        self.scheduler.step()

        running_loss /= len(dataloader)

        predicted_logits_of_overall_train = torch.stack(
            predicted_logits_of_overall_train
        )
        true_labels_of_overall_train = (
            torch.stack(true_labels_of_overall_train).squeeze().long()
        )

        accuracy = self.ac(
            predicted_logits_of_overall_train, true_labels_of_overall_train
        )
        f1_score = self.f1(
            predicted_logits_of_overall_train, true_labels_of_overall_train
        )
        auroc = self.auroc(
            predicted_logits_of_overall_train, true_labels_of_overall_train
        )

        print(
            f"Train Loss: {running_loss:.4f}, Train Accuracy: {accuracy:.4f}, "
            f"Train F1 Score: {f1_score:.4f}, Train AUROC: {auroc:.4f}"
        )

        self.experiment.log_metrics(
            {
                "Train Accuracy": accuracy,
                "Train Loss": running_loss,
                "Train F1 Score": f1_score,
                "Train AUROC": auroc,
            },
            step=epoch_number,
        )

        return running_loss

    def test_one_epoch(self, dataloader, epoch_number, phase="val"):
        self.model.eval()

        self.ac.reset()
        self.f1.reset()
        self.auroc.reset()

        running_loss = 0.0

        predicted_logits_of_overall_test = []
        true_labels_of_overall_test = []

        for batch in tqdm(
            dataloader,
            desc=f"Phase: {phase}",
            total=len(dataloader),
            unit="batch",
            dynamic_ncols=True,
        ):
            inputs, _masks, labels, _p_id = (
                batch["image"].to(self.device),
                batch["la_label"].to(self.device),
                batch["labels"],
                batch["p_id"],
            )

            labels = {
                key: value.to(self.device).unsqueeze(1) for key, value in labels.items()
            }

            with torch.no_grad():
                outputs = self.model(inputs)

                loss = self.ce_loss(
                    input=outputs,
                    target=labels["quality_for_fibrosis_assessment"].long().squeeze(1),
                )

                predicted_overall = [
                    self.post_process(i) for i in decollate_batch(outputs)
                ]

                predicted_logits_of_overall_test.extend(predicted_overall)
                true_labels_of_overall_test.extend(
                    decollate_batch(labels["quality_for_fibrosis_assessment"])
                )

                running_loss += loss.item()

        predicted_logits_of_overall_test = torch.stack(predicted_logits_of_overall_test)
        true_labels_of_overall_test = (
            torch.stack(true_labels_of_overall_test).squeeze().long()
        )

        running_loss /= len(dataloader)

        accuracy = self.ac(
            predicted_logits_of_overall_test, true_labels_of_overall_test
        )
        f1_score = self.f1(
            predicted_logits_of_overall_test, true_labels_of_overall_test
        )
        auroc = self.auroc(
            predicted_logits_of_overall_test, true_labels_of_overall_test
        )

        print(
            f"{phase} Loss: {running_loss:.4f}, {phase} Accuracy: {accuracy:.4f}, "
            f"{phase} F1 Score: {f1_score:.4f}, {phase} AUROC: {auroc:.4f}"
        )

        self.experiment.log_metrics(
            {
                f"{phase} Accuracy": accuracy,
                f"{phase} Loss": running_loss,
                f"{phase} F1 Score": f1_score,
                f"{phase} AUROC": auroc,
            },
            step=epoch_number,
        )

        return auroc

    def train(self, early_stopping, train_weights=None):
        samples_train_weights = torch.from_numpy(train_weights).float()

        sampler = WeightedRandomSampler(
            weights=samples_train_weights,
            num_samples=len(samples_train_weights),
            replacement=True,
        )

        # For multi-class use the sampler in train loader
        # Disable multi-workers because ThreadDataLoader works with multi-threads
        train_loader = ThreadDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=16,
            use_thread_workers=True,
            sampler=sampler,
        )
        val_loader = ThreadDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            use_thread_workers=True,
        )

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}/{self.epochs}")

            self.train_one_epoch(dataloader=train_loader, epoch_number=epoch)
            val_auroc = self.test_one_epoch(
                dataloader=val_loader, epoch_number=epoch, phase="val"
            )

            if epoch >= early_stopping.start_epoch:
                early_stopping(val_score=val_auroc, model=[self.model])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    def test(self, model_save_path):
        test_loader = ThreadDataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            use_thread_workers=True,
        )

        print(f"Loading the model from {base_dir + '/' + model_save_path}")
        self.model.load_state_dict(
            torch.load(base_dir + "/" + model_save_path, weights_only=True)
        )

        self.test_one_epoch(dataloader=test_loader, epoch_number=0, phase="test")


class AFibQCSliceMonaiTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size, epochs, lr, load_self_sup_model=False,experiment=None,require_concept_labels=False,just_concept=False,concept_weight=0.5,semantic_weight=0.5,require_semantic_loss=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.experiment = experiment
        self.concept_labels = require_concept_labels
        self.just_concept = just_concept
        self.concept_weight = concept_weight
        self.semantic_weight=semantic_weight
        self.require_semantic_loss=require_semantic_loss
        self.device = device
        self.rule_semantic=RuleSemanticLoss(normalize="softmax").to(self.device)
        # self.data_generator = torch.Generator().manual_seed(CONFIG['seed'])

        

        if load_self_sup_model:
            print("Loading the weights of the self-supervised model to the ABMIL model")
            self.load_self_sup_model_weights_to_model()

        self.model.to(self.device)

        # self.initial_weights = {}
        # for name, param in self.model.named_parameters():
        #     self.initial_weights[name] = param.data.clone()

        # self.optimizer = create_optimizer(model=self.model, optimizer_name='radam', lr=self.lr, weight_decay=CONFIG['weight_decay'], use_lookahead=True)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=CONFIG['weight_decay'], betas = (0.9, 0.98), eps = 1e-6)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=CONFIG['weight_decay'], momentum=0.9)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.post_process = self.get_post_process()

        self.ac, self.f1, self.spec, self.cm, self.auroc, self.avgprec= (
            self.get_classification_metrics(classification=CONFIG["classification"])
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)

    def load_self_sup_model_weights_to_model(self):
        encoder_state_dict = copy.deepcopy(self.model.encoder_3d.state_dict()) # Create a deep copy of the state_dict of the encoder_3d of the supervised model

        self.model.encoder_3d.load_state_dict(torch.load(base_dir + f'/{CONFIG["model_path"]}/{CONFIG["saved_model_name_self_supervised"]}')) 

        # The values of encoder_3d_state_dict should be different after loading the weights
        for key in encoder_state_dict.keys():
            if key not in ['fc.weight', 'fc.bias']:
                assert not torch.equal(encoder_state_dict[key], self.model.encoder_3d.state_dict()[key]), f"The weights of the self-supervised model and the supervised model are the same for the key: {key}"

        del encoder_state_dict

        # freeze the weights of the encoder
        for param in self.model.encoder_3d.parameters():
            param.requires_grad = False
            
    def check_gradients(self):
        # print(self.model)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError(f"Gradient of {name} contains NaN values")
                if param.grad is None:
                    raise ValueError(f"Gradient of {name} is None")

    def get_classification_metrics(self, classification="binary"):
        if classification == "binary":
            return (
                BinaryAccuracy(threshold=0.5).to(self.device),
                BinaryF1Score(threshold=0.5).to(self.device),
                BinarySpecificity(threshold=0.5).to(self.device),
                BinaryConfusionMatrix(threshold=0.5).to(self.device),
                BinaryAUROC(thresholds=5).to(self.device),
                BinaryAveragePrecision(thresholds=5).to(self.device),
            )

        elif classification == "multiclass":
            return (
                MulticlassAccuracy(num_classes=4).to(self.device),
                MulticlassF1Score(num_classes=4).to(self.device),
                MulticlassSpecificity(num_classes=4).to(self.device),
                MulticlassConfusionMatrix(num_classes=4).to(self.device),
                MulticlassAUROC(num_classes=4, thresholds=5).to(self.device),
                MulticlassAveragePrecision(num_classes=4, thresholds=5).to(self.device),
             )
        
    def get_post_process(self):
        if CONFIG['classification'] == 'binary':
            return Compose([Activations(sigmoid=True)])
        elif CONFIG['classification'] == 'multiclass':
            return Compose([Activations(softmax=True)])

    def get_loss(self, outputs, labels, loss_fn_to_use=None):
        if CONFIG['classification'] == 'binary':
            loss = self.bce_logits_loss(outputs.squeeze(1), labels)
        elif CONFIG['classification'] == 'multiclass':
            labels_reshaped = labels.clone()
            outputs_reshaped = outputs.clone()

            labels_reshaped = labels_reshaped.unsqueeze(1).repeat(1, outputs.size(1), 1)
            labels_reshaped = labels_reshaped.reshape(-1)
            outputs_reshaped = outputs_reshaped.reshape(-1, outputs.size(2))

            # loss = self.ce_loss(outputs_reshaped, labels_reshaped.long())
            # loss = CB_loss(labels=labels_reshaped.long(), logits=outputs_reshaped, samples_per_cls=self.class_sample_counts, no_of_classes=4, loss_type='focal', beta=0.9999, gamma=2.0, device=self.device)

            if loss_fn_to_use == 'corn_loss':
                loss = corn_loss(logits=outputs_reshaped, y_train=labels_reshaped.long(), num_classes=4)
            elif loss_fn_to_use == 'cb_corn_loss':
                loss = cb_corn_loss(logits=outputs_reshaped, y_train=labels_reshaped.long(), num_classes=4, samples_per_cls=self.class_sample_counts, beta=0.9)

        return loss

    def calculate_ordinal_metrics(self, predicted_labels, true_labels):
        """
        Calculate ordinal-aware metrics using both custom implementations and external metrics.
        
        Args:
            predicted_labels: Predicted class labels (0-3 representing classes 1-4)
            true_labels: True class labels (0-3 representing classes 1-4)
        
        Returns:
            dict: Dictionary containing ordinal metrics
        """
        predicted_labels_np = predicted_labels.cpu().numpy()
        true_labels_np = true_labels.cpu().numpy()
        
        scotts_pi = ScottsPiQuadratic()
        # Metrics from util.metrics module
        try:
            # 1-off accuracy from metrics module
            acc_off1 = accuracy_off1_macro(true_labels_np, predicted_labels_np)
            
            # AMAE (Average Mean Absolute Error)
            amae_score = amae(true_labels_np, predicted_labels_np)
            
            # MMAE (Maximum Mean Absolute Error)  
            mmae_score = mmae(true_labels_np, predicted_labels_np)
            
            # GMSEC (Geometric Mean of Sensitivity and Specificity for Each Class)
            gmsec_score = gmsec(true_labels_np, predicted_labels_np)
            
            scotts_pi_score = scotts_pi.fit_score(true_labels_np, predicted_labels_np)
        except Exception as e:
            print(f"Warning: Error calculating metrics from util.metrics: {e}")
            raise ValueError(f"Error calculating metrics from util.metrics: {e}")
            
        # Agreement metrics
        num_classes = len(np.unique(true_labels_np))
        
        # Quadratic Weighted Kappa (QWK)
        qwk = cohen_kappa_score(true_labels_np, predicted_labels_np, weights='quadratic')
        
        # Linear Weighted Kappa  
        linear_kappa = cohen_kappa_score(true_labels_np, predicted_labels_np, weights='linear')

        # Kendall's Tau
        kendall_tau, _ = kendalltau(true_labels_np, predicted_labels_np)

        # Class-wise metrics
        class_wise_sensitivity = {}
        class_wise_specificity = {}
        
        for class_idx in range(num_classes):
            class_mask = true_labels_np == class_idx
            if np.sum(class_mask) > 0:
                # Sensitivity (Recall) for this class
                tp = np.sum((predicted_labels_np == class_idx) & (true_labels_np == class_idx))
                fn = np.sum((predicted_labels_np != class_idx) & (true_labels_np == class_idx))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                class_wise_sensitivity[f'sensitivity_class_{class_idx+1}'] = sensitivity
                
                # Specificity for this class
                tn = np.sum((predicted_labels_np != class_idx) & (true_labels_np != class_idx))
                fp = np.sum((predicted_labels_np == class_idx) & (true_labels_np != class_idx))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                class_wise_specificity[f'specificity_class_{class_idx+1}'] = specificity
        
        
        # Macro-averaged metrics
        macro_sensitivity = np.mean(list(class_wise_sensitivity.values()))
        macro_specificity = np.mean(list(class_wise_specificity.values()))
        
        return {
            # Metrics from util.metrics module
            'one_off_accuracy': acc_off1,
            'amae': amae_score,
            'mmae': mmae_score,
            'gmsec': gmsec_score,
            'scotts_pi': scotts_pi_score,
            
            # Agreement metrics
            'qwk': qwk,
            'linear_weighted_kappa': linear_kappa,
            'kendall_tau': kendall_tau,
            
            # Macro metrics
            'macro_sensitivity': macro_sensitivity,
            'macro_specificity': macro_specificity,
            
            # Class-wise metrics
            **class_wise_sensitivity,
            **class_wise_specificity,
        }

    def train_one_epoch(self, epoch_number, loss_fn_to_use=None, pgdFunc=None):
        self.model.train()

        self.ac.reset(), self.f1.reset(), self.spec.reset(), self.cm.reset(), self.auroc.reset(), self.avgprec.reset()

        running_total_loss = 0.0
        running_loss_original = 0.0
        running_loss_concpet = 0.0
        running_loss_semantic = 0.0

        
        predicted_logits_of_overall_train_and_val = []
        true_labels_of_overall_train_and_val = []

        epoch_iterator = tqdm(self.train_loader, desc=f'Phase: train:', total=len(self.train_loader), unit='batch', dynamic_ncols=True)

        iterat = 0 
        for batch in epoch_iterator:
            inputs, labels, p_id = batch['image'].to(self.device), batch['labels'], batch['p_id'] # inputs shape: (batch_size=1, no_of_patches, 1, 64, 64, 7)
            
            labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
               
                if self.concept_labels:
                    outputs,c1,c2,c3 = self.model(inputs)
                else:
                     outputs = self.model(inputs)
                

                # loss = self.get_loss(outputs, labels['quality_for_fibrosis_assessment'], loss_fn_to_use=loss_fn_to_use)
                if self.concept_labels:
                    loss_original  = corn_loss(logits=outputs, y_train=labels['quality_for_fibrosis_assessment'].view(-1).long(), num_classes=4)
                    c1_loss_sharpness = corn_loss(logits=c1, y_train=labels['sharpness'].view(-1).long(), num_classes=4)
                    c2_loss_myocardium = corn_loss(logits=c2, y_train=labels['myocardium_nulling'].view(-1).long(), num_classes=4)
                    c3_loss_enhancement = corn_loss(logits=c3, y_train=labels['enhancement_of_aorta_and_valves'].view(-1).long(), num_classes=4)
                    concepts_loss= c1_loss_sharpness + c2_loss_myocardium + c3_loss_enhancement
                    if self.just_concept:
                        loss = concepts_loss
                    else:
                        loss = loss_original + self.concept_weight*concepts_loss
                    
                    if self.require_semantic_loss:
                        semantic_loss ,rules_loss_vec, rules_loss_weight= self.rule_semantic(main_logits=outputs, c1_logits=c1, c2_logits=c2, c3_logits=c3)
                      #  semantic_loss=rule_based_semantic_loss_product(main_logits=outputs, c1_logits=c1, c2_logits=c2, c3_logits=c3)
                        loss = loss+self.concept_weight*concepts_loss+self.semantic_weight*semantic_loss

                else:
                    loss = corn_loss(logits=outputs, y_train=labels['quality_for_fibrosis_assessment'].view(-1).long(), num_classes=4)

                loss.backward()

                if CONFIG['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['max_clip_grad'])

                self.optimizer.step()

            running_total_loss += loss.item()
            if self.concept_labels:
                running_loss_original += loss_original.item()
                running_loss_concpet += concepts_loss.item()
                running_loss_semantic += semantic_loss.item()
                
            # predicted_overall = [self.post_process(i) for i in decollate_batch(outputs.squeeze(1))]
            # predicted_logits_of_overall_test.extend(predicted_overall)
            predicted_logits_of_overall_train_and_val.append(outputs)
            true_labels_of_overall_train_and_val.extend(decollate_batch(labels['quality_for_fibrosis_assessment']))

            # iterat += 1

            # if iterat > 2:
            #     break

        if pgdFunc is not None:
            pgdFunc.PGD(self.model)

        self.scheduler.step()
    
        epoch_total_loss = running_total_loss / len(self.train_loader)
        if self.concept_labels:
            epoch_loss_original = running_loss_original / len(self.train_loader)
            epoch_concepts_loss = running_loss_concpet / len(self.train_loader)
            epoch_semantic_loss = running_loss_semantic / len(self.train_loader)

        predicted_logits_of_overall_train_and_val = torch.cat(predicted_logits_of_overall_train_and_val)
        # predicted_logits_of_overall_train_and_val = self.post_process(predicted_logits_of_overall_train_and_val)
        predicted_logits_of_overall_train_and_val = corn_label_from_logits(predicted_logits_of_overall_train_and_val)
        # predicted_logits_of_overall_train_and_val = torch.mode(predicted_logits_of_overall_train_and_val, dim=1)[0]
        true_labels_of_overall_train_and_val = torch.stack(true_labels_of_overall_train_and_val).squeeze().long() if CONFIG['classification'] == 'multiclass' else torch.stack(true_labels_of_overall_train_and_val).long()

        accuracy = self.ac(predicted_logits_of_overall_train_and_val, true_labels_of_overall_train_and_val)
        f1_score = self.f1(predicted_logits_of_overall_train_and_val, true_labels_of_overall_train_and_val)
        ordinal_metrics = self.calculate_ordinal_metrics(predicted_logits_of_overall_train_and_val, true_labels_of_overall_train_and_val)
        # auroc = self.auroc(predicted_logits_of_overall_train_and_val, true_labels_of_overall_train_and_val)
        # avg_precision = self.avgprec(predicted_logits_of_overall_train_and_val, true_labels_of_overall_train_and_val)

        # Log all metrics\
        if self.concept_labels:
            metrics_to_log = {
                f"Train Loss": epoch_total_loss,
                f"Train concepts Loss": epoch_concepts_loss,
                f"Train semantic Loss": epoch_semantic_loss,
                f"Train Original Loss": epoch_loss_original,
                f"Train Accuracy (Exact)": accuracy,
                f"Train F1 Score": f1_score,
                f"Train 1-Off Accuracy": ordinal_metrics['one_off_accuracy'],
                f"Train QWK": ordinal_metrics['qwk'],
                f"Train Kendall Tau": ordinal_metrics['kendall_tau'],
                f"Train Scott's Pi": ordinal_metrics['scotts_pi']
                
            }
        else: 
            metrics_to_log = {
                f"Train Loss": epoch_total_loss,
                f"Train Accuracy (Exact)": accuracy,
                f"Train F1 Score": f1_score,
                f"Train 1-Off Accuracy": ordinal_metrics['one_off_accuracy'],
                f"Train QWK": ordinal_metrics['qwk'],
                f"Train Kendall Tau": ordinal_metrics['kendall_tau'],
                f"Train Scott's Pi": ordinal_metrics['scotts_pi']

            }
        # Add class-wise metrics
        # for key, value in ordinal_metrics.items():
        #     if key.startswith(('sensitivity_class_', 'specificity_class_')):
        #         metrics_to_log[f"Train {key}"] = value

        self.experiment.log_metrics(metrics_to_log, step=epoch_number)
        if self.concept_labels:
            print(f'Concept Loss: {epoch_concepts_loss:.4f}, Original Loss: {epoch_loss_original:.4f}, Train Loss: {epoch_total_loss:.4f}, Exact Accuracy: {accuracy:.4f}, 1-Off Accuracy: {ordinal_metrics["one_off_accuracy"]:.4f}, '
              f'F1 Score: {f1_score:.4f}, AMAE: {ordinal_metrics["amae"]:.4f}, Semantic Loss: {epoch_semantic_loss:.4f}',
              f'MMAE: {ordinal_metrics["mmae"]:.4f}, QWK: {ordinal_metrics["qwk"]:.4f}, Kendall Tau: {ordinal_metrics["kendall_tau"]:.4f}, '
              f'Scotts Pi: {ordinal_metrics["scotts_pi"]:.4f}')
        else:
            # print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1 Score: {f1_score:.4f}, Avg Precision: {avg_precision:.4f}')
            print(f'Train Loss: {epoch_total_loss:.4f}, Exact Accuracy: {accuracy:.4f}, 1-Off Accuracy: {ordinal_metrics["one_off_accuracy"]:.4f}, '
            f'F1 Score: {f1_score:.4f}, AMAE: {ordinal_metrics["amae"]:.4f}, '
            f'MMAE: {ordinal_metrics["mmae"]:.4f}, QWK: {ordinal_metrics["qwk"]:.4f}, ')

    def test_one_epoch(self, epoch_number, phase, loss_fn_to_use=None):
        self.model.eval()

        self.ac.reset(), self.f1.reset(), self.spec.reset(), self.cm.reset(), self.auroc.reset(), self.avgprec.reset()

        if phase == 'val':
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader

        running_loss_original = 0.0
        running_loss_total= 0.0
        running_loss_concpet=0.0

        predicted_logits_of_overall_test = []
        weightnorm_classifier = []
        true_labels_of_overall_test = []
        p_ids = []

        epoch_iterator = tqdm(dataloader, desc=f'Phase: {phase}:', total=len(dataloader), unit='batch', dynamic_ncols=True)

        iterat = 0
        for batch in epoch_iterator:
            inputs, labels, p_id = batch['image'].to(self.device), batch['labels'], batch['p_id']
            
            labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}
            p_ids.extend(p_id)

            with torch.no_grad():
                if self.concept_labels:
                    outputs,c1,c2,c3 = self.model(inputs)
                    
                else:
                     outputs = self.model(inputs)
             
               
                if self.concept_labels:
                    loss_original=corn_loss(logits=outputs, y_train=labels['quality_for_fibrosis_assessment'].view(-1).long(), num_classes=4)
                    concept1_loss = corn_loss(logits=c1, y_train=labels['sharpness'].view(-1).long(), num_classes=4)
                    concept2_loss = corn_loss(logits=c2, y_train=labels['myocardium_nulling'].view(-1).long(), num_classes=4)
                    concept3_loss = corn_loss(logits=c3, y_train=labels['enhancement_of_aorta_and_valves'].view(-1).long(), num_classes=4)
                    concepts_loss= concept1_loss + concept2_loss + concept3_loss
                    total_loss = loss_original + concepts_loss
                    running_loss_total += total_loss.item()
                    running_loss_concpet += concepts_loss.item()
                    running_loss_original += loss_original.item()
                else:

                     # loss = self.get_loss(outputs, labels['quality_for_fibrosis_assessment'], loss_fn_to_use=loss_fn_to_use)
                    loss = corn_loss(logits=outputs, y_train=labels['quality_for_fibrosis_assessment'].view(-1).long(), num_classes=4)
                    running_loss_total += loss.item()

                # predicted_overall = [self.post_process(i) for i in decollate_batch(outputs.squeeze(1))]
                # predicted_logits_of_overall_test.extend(predicted_overall)
                predicted_logits_of_overall_test.append(outputs)
                true_labels_of_overall_test.extend(decollate_batch(labels['quality_for_fibrosis_assessment']))

            # iterat += 1

            # if iterat > 2:
            #     break

        epoch_loss = running_loss_total / len(dataloader)
        if self.concept_labels:
            epoch_loss_original = running_loss_original / len(dataloader)
            epoch_concepts_loss = running_loss_concpet / len(dataloader)
            print(f'{phase} Loss: {epoch_loss:.4f}, Loss Original: {epoch_loss_original:.4f}, Loss Concepts: {epoch_concepts_loss:.4f}')
        predicted_logits_of_overall_test = torch.cat(predicted_logits_of_overall_test)
        # predicted_logits_of_overall_test = self.post_process(predicted_logits_of_overall_test)
        predicted_logits_of_overall_test = corn_label_from_logits(predicted_logits_of_overall_test)
        # predicted_logits_of_overall_test = torch.mode(predicted_logits_of_overall_test, dim=1)[0]
        true_labels_of_overall_test = torch.stack(true_labels_of_overall_test).squeeze().long() if CONFIG['classification'] == 'multiclass' else torch.stack(true_labels_of_overall_test).long()

        accuracy = self.ac(predicted_logits_of_overall_test, true_labels_of_overall_test)
        f1_score = self.f1(predicted_logits_of_overall_test, true_labels_of_overall_test)

        # Calculate ordinal-aware metrics
        ordinal_metrics = self.calculate_ordinal_metrics(predicted_logits_of_overall_test, true_labels_of_overall_test)

        # Log all metrics
        if self.concept_labels:
            metrics_to_log = {
                f"{phase} Loss": epoch_loss,
                f"{phase} concepts Loss": epoch_concepts_loss,
                f"{phase} Original Loss": epoch_loss_original,
                f"{phase} Accuracy (Exact)": accuracy,
                f"{phase} F1 Score": f1_score,
                f"{phase} 1-Off Accuracy": ordinal_metrics['one_off_accuracy'],
                f"{phase} QWK": ordinal_metrics['qwk'],
                f"{phase} Kendall Tau": ordinal_metrics['kendall_tau'],
                f"{phase} Scott's Pi": ordinal_metrics['scotts_pi']
                
            }
        else:
            metrics_to_log = {
                f"{phase} Loss": epoch_loss,
                f"{phase} Accuracy (Exact)": accuracy,
                f"{phase} F1 Score": f1_score,
                f"{phase} 1-Off Accuracy": ordinal_metrics['one_off_accuracy'],
                f"{phase} QWK": ordinal_metrics['qwk'],
                f"{phase} Kendall Tau": ordinal_metrics['kendall_tau'],
                f"{phase} Scott's Pi": ordinal_metrics['scotts_pi']
            }
        
        # Add class-wise metrics
        # for key, value in ordinal_metrics.items():
        #     if key.startswith(('sensitivity_class_', 'specificity_class_')):
        #         metrics_to_log[f"{phase} {key}"] = value

        self.experiment.log_metrics(metrics_to_log, step=epoch_number)
        
        print(f'{phase} Loss: {epoch_loss:.4f}, Exact Accuracy: {accuracy:.4f}, 1-Off Accuracy: {ordinal_metrics["one_off_accuracy"]:.4f}, '
            f'F1 Score: {f1_score:.4f}, AMAE: {ordinal_metrics["amae"]:.4f}, '
            f'MMAE: {ordinal_metrics["mmae"]:.4f}, QWK: {ordinal_metrics["qwk"]:.4f}')
        
        if phase == 'test':
            weights_classifier_norm = torch.linalg.norm(self.model.classifier.weight.data, ord=2, dim=1).cpu().detach().numpy() # shape: (num_classes,)
            weights_classifier_norm = np.where(np.isclose(weights_classifier_norm, 1.0, atol=1e-2), 1.0, weights_classifier_norm) # Adjust values close to 1.0
            
            #  plot line plot for classifier weights norm for each class
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(weights_classifier_norm)), weights_classifier_norm, marker='o', linestyle='-', color='b')
            plt.title('Classifier Weights Norm')
            plt.xlabel('Class Index')
            plt.ylabel('L2 Norm of Classifier Weights')
            plt.xticks(range(len(weights_classifier_norm)))

            # Save the figure to the experiment
            self.experiment.log_figure(
                    figure_name="Classifier Weights Norm",
                    figure=plt.gcf(),
                    overwrite=True
                )

            plt.close() # Close the figure to free memory

            confusion_matrix = self.cm(predicted_logits_of_overall_test, true_labels_of_overall_test)
            confusion_matrix = confusion_matrix.cpu().numpy()
            # log confusion matrix
            self.experiment.log_confusion_matrix(
                matrix=confusion_matrix,
                title=f"{phase} Confusion Matrix",
                file_name=f"{phase}_confusion_matrix.png"
            )

        return ordinal_metrics["qwk"]

    def train(self, early_stopping, train_weights=None, class_sample_counts=None):
        # Disable multi-workers because ThreadDataLoader works with multi-threads
        # train_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        # val_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.class_sample_counts = torch.tensor(class_sample_counts).float().to(self.device) if class_sample_counts is not None else None
        
        samples_train_weights = torch.from_numpy(train_weights).float()

        sampler = WeightedRandomSampler(weights=samples_train_weights, num_samples=len(samples_train_weights), replacement=True)
        
        self.train_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0, sampler=sampler)
        self.val_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)
        
        # self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available(), pin_memory_device=f"cuda:{gpu_id}", sampler=sampler)
        # self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=torch.cuda.is_available(), pin_memory_device=f"cuda:{gpu_id}")

        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            self.train_one_epoch(epoch_number=epoch, loss_fn_to_use=CONFIG['initial_loss'])

            accuracy = self.test_one_epoch(epoch_number=epoch, phase='val', loss_fn_to_use=CONFIG['initial_loss'])

            if epoch >= early_stopping.start_epoch:
                early_stopping(accuracy, [self.model])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
        # model_save_path = f"{CONFIG['model_path']}/{CONFIG['saved_model_name']}"
        # state_dict = torch.load(base_dir + "/" + model_save_path, weights_only=True)

        # self.model.load_state_dict(state_dict)

        # self.fine_tune_w_maxnorm_wd(train_weights=train_weights, class_sample_counts=class_sample_counts)
        
    def fine_tune_w_maxnorm_wd(self, train_weights=None, class_sample_counts=None):
        print("-" * 50)
        print("Fine tuning the model with max norm and weight decay")
        self.class_sample_counts = torch.tensor(class_sample_counts).float().to(self.device) if class_sample_counts is not None else None
        
        active_layers = [self.model.classifier.weight, self.model.classifier.bias]
        
        for layer in active_layers:
            if layer is not None:
                if layer.dim() > 1:  # For weights
                    nn.init.kaiming_normal_(layer, mode='fan_out', nonlinearity='relu')
                else:  # For biases
                    nn.init.zeros_(layer)

        self.optimizer = optim.SGD(params=active_layers,  # Only update the classifier layer 
                                   lr=self.lr, 
                                   weight_decay=CONFIG['fine_tune_weight_decay'])
        
        self.epochs = CONFIG['fine_tune_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0)

        thresh = 1e-1 #threshold value
        pgdFunc = MaxNorm_via_PGD(thresh=thresh)
        pgdFunc.setPerLayerThresh(self.model)

        # fine tune the last layer only
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            self.train_one_epoch(epoch_number=epoch, loss_fn_to_use=CONFIG['fine_tune_loss'], pgdFunc=pgdFunc)

            accuracy = self.test_one_epoch(epoch_number=epoch, phase='val', loss_fn_to_use=CONFIG['fine_tune_loss'])

        torch.save(self.model.state_dict(), base_dir + f"/model/saved_models/{CONFIG['fine_tune_model_name']}")

    def test(self, model_save_path):
        # test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        # self.train_loader = DataLoader(self.train_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available(), pin_memory_device=f"cuda:{gpu_id}")
        # self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=torch.cuda.is_available(), pin_memory_device=f"cuda:{gpu_id}") if self.val_loader is None else self.val_loader
        # self.test_loader = DataLoader(self.test_dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=torch.cuda.is_available(), pin_memory_device=f"cuda:{gpu_id}")

        self.test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

        print(f"Loading the model from {base_dir + '/' + model_save_path}")

        state_dict = torch.load(base_dir + "/" + model_save_path, weights_only=True)
        self.model.load_state_dict(state_dict)

        # tau_norm = Normalizer(tau=1.0)
        # tau_norm.apply_on(self.model)

        # norm = torch.linalg.norm(self.model.classifier.weight.data, ord=2, dim=1) # shape: (num_classes,)
        
        # for (k1, v1), (k2, v2) in zip(self.model.state_dict().items(), self.best_model_state_dict.items()):
        #     if not torch.equal(v1, v2):
        #         print(f"Mismatch in layer {k1}")

        f1_score = self.test_one_epoch(epoch_number=self.epochs, phase='test', loss_fn_to_use=CONFIG['initial_loss'])

        self.log_test_images_to_comet(model_save_path=model_save_path, phase='val')
        self.log_test_images_to_comet(model_save_path=model_save_path, phase='test')

    def predict_unlabeled(self, model_save_path, unlabeled_dataset):
        """
        Predict on unlabeled dataset and return predictions with patient IDs
        """
        self.unlabeled_loader = ThreadDataLoader(dataset=unlabeled_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

        print(f"Loading the model from {base_dir + '/' + model_save_path}")

        state_dict = torch.load(base_dir + "/" + model_save_path, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()

        # Reset metrics
        self.ac.reset()
        self.f1.reset()
        self.spec.reset()
        self.cm.reset()
        self.auroc.reset()
        self.avgprec.reset()

        predicted_logits_list = []
        predicted_probabilities_list = []
        predicted_labels_list = []
        p_ids_list = []

        epoch_iterator = tqdm(self.unlabeled_loader, desc='Phase: prediction:', total=len(self.unlabeled_loader), unit='batch', dynamic_ncols=True)

        for batch in epoch_iterator:
            inputs, p_id = batch['image'].to(self.device), batch['p_id']
            p_ids_list.extend(p_id)

            with torch.no_grad():
                outputs = self.model(inputs)  # outputs shape: (1, 1) or (1, n_slices, n_classes)

                predicted_logits_list.append(outputs)

        # Process predictions
        predicted_logits = torch.cat(predicted_logits_list)
        
        if CONFIG['classification'] == 'multiclass':
            predicted_logits = corn_label_from_logits(predicted_logits)
            predicted_labels = torch.mode(predicted_logits, dim=1)[0] if len(predicted_logits.shape) > 1 else predicted_logits
            # For multiclass, also get probabilities using softmax
            predicted_probabilities = self.post_process(torch.cat(predicted_logits_list))
            predicted_probabilities = torch.mode(predicted_probabilities, dim=1)[0] if len(predicted_probabilities.shape) > 1 else predicted_probabilities
        else:
            predicted_probabilities = self.post_process(predicted_logits)
            predicted_labels = (predicted_probabilities > 0.5).float()

        # Create results dictionary
        predictions = {
            'patient_ids': p_ids_list,
            'predicted_logits': predicted_logits.cpu().numpy(),
            'predicted_probabilities': predicted_probabilities.cpu().numpy(),
            'predicted_labels': predicted_labels.cpu().numpy()
        }

        # Save predictions to CSV
        import pandas as pd
        df_predictions = pd.DataFrame({
            'patient_id': predictions['patient_ids'],
            'predicted_probability': predictions['predicted_probabilities'].flatten() if CONFIG['classification'] == 'binary' else predictions['predicted_probabilities'].tolist(),
            'predicted_label': predictions['predicted_labels'].flatten(),
            'predicted_logits': predictions['predicted_logits'].flatten() if CONFIG['classification'] == 'binary' else predictions['predicted_logits'].tolist()
        })

        # Add quality score mapping for interpretability
        if CONFIG['classification'] == 'binary':
            df_predictions['predicted_quality_score'] = df_predictions['predicted_label'].apply(
                lambda x: 'Low Quality (1-2)' if x == 0 else 'High Quality (3-4)'
            )
        else:
            df_predictions['predicted_quality_score'] = df_predictions['predicted_label'] + 1  # Convert back to 1-4 scale

        prediction_save_path = f"{base_dir}/predictions/{MODEL_NAME}_unlabeled_predictions.csv"
        os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
        df_predictions.to_csv(prediction_save_path, index=False)

        print(f"Predictions saved to: {prediction_save_path}")
        print(f"Total unlabeled cases predicted: {len(predictions['patient_ids'])}")

        if CONFIG['classification'] == 'binary':
            low_quality_count = (predictions['predicted_labels'] == 0).sum()
            high_quality_count = (predictions['predicted_labels'] == 1).sum()
            print(f"Predicted Low Quality (1-2): {low_quality_count}")
            print(f"Predicted High Quality (3-4): {high_quality_count}")
        else:
            for i in range(4):
                count = (predictions['predicted_labels'] == i).sum()
                print(f"Predicted Quality Score {i+1}: {count}")

        # Log to experiment if available
        if self.experiment:
            self.experiment.log_table(filename="unlabeled_predictions.csv", tabular_data=df_predictions)
            self.experiment.log_metrics({
                "total_unlabeled_predictions": len(predictions['patient_ids']),
                "low_quality_predictions": low_quality_count if CONFIG['classification'] == 'binary' else (predictions['predicted_labels'] <= 1).sum(),
                "high_quality_predictions": high_quality_count if CONFIG['classification'] == 'binary' else (predictions['predicted_labels'] >= 2).sum(),
            })

        return predictions

    def log_test_images_to_comet(self, model_save_path, num_images_per_patient=5, phase='test'):
        """
        Log axial images from test set to Comet with ground truth and prediction labels
        Create one figure per patient showing multiple axial slices
        
        Args:
            model_save_path: Path to the saved model
            num_images_per_patient: Number of axial slices to log per patient
        """
        
        # Load the trained model
        print(f"Loading the model from {base_dir + '/' + model_save_path}")
        state_dict = torch.load(base_dir + "/" + model_save_path, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        if phase == 'test':
            data_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        elif phase == 'val':
            data_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        elif phase == 'train':
            data_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Reset metrics
        self.ac.reset()
        self.f1.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Logging {phase} images', total=len(data_loader))):
                # Get batch data
                inputs = batch['image'].to(self.device)  # Shape: (1, no_of_slices, 2, 224, 224)
                labels = batch['labels']
                patient_id = batch['p_id'][0] if isinstance(batch['p_id'], list) else batch['p_id']
                
                # Prepare labels
                labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}
                
                # Get model prediction
                if self.concept_labels:
                    outputs,c1,c2,c3 = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                if CONFIG['classification'] == 'multiclass':
                    # For multiclass, use CORN decoding
                    predicted_label = corn_label_from_logits(outputs)
                    # predicted_label = torch.mode(predicted_label, dim=1)[0].cpu().item()
                    true_label = labels['quality_for_fibrosis_assessment'].squeeze().cpu().item()
                    
                    # Convert to quality scores (assuming 0-3 maps to 1-4)
                    predicted_quality = predicted_label + 1
                    true_quality = true_label + 1
                    
                else:
                    # For binary classification
                    predicted_probs = torch.sigmoid(outputs)
                    predicted_prob = torch.mode(predicted_probs, dim=1)[0].cpu().item()
                    predicted_label = (predicted_prob > 0.5).float().cpu().item()
                    true_label = labels['quality_for_fibrosis_assessment'].squeeze().cpu().item()
                    
                    # Map binary to quality interpretation
                    predicted_quality = "Good" if predicted_label == 1 else "Poor"
                    true_quality = "Good" if true_label == 1 else "Poor"
                
                # Extract axial slices from input
                # inputs shape: (1, no_of_slices, 2, 224, 224)
                axial_slices = inputs.squeeze(0).cpu().numpy()  # Shape: (no_of_slices, 2, 224, 224)
                
                # Extract only the first channel
                axial_slices = axial_slices[:, 0, :, :]  # Shape: (no_of_slices, 224, 224)
                
                num_slices = axial_slices.shape[0]
                
                # Select evenly spaced slices to display
                num_slices_to_show = min(num_images_per_patient, num_slices)
                if num_slices > num_slices_to_show:
                    slice_indices = np.linspace(0, num_slices-1, num_slices_to_show, dtype=int)
                else:
                    slice_indices = np.arange(num_slices)
                
                selected_slices = axial_slices[slice_indices]
                
                # Create figure for this patient
                fig, axes = plt.subplots(1, len(selected_slices), figsize=(4*len(selected_slices), 4))
                if len(selected_slices) == 1:
                    axes = [axes]  # Make it iterable
                
                fig.suptitle(f'Patient {patient_id}\nGT: {true_quality} | Pred: {predicted_quality}', 
                        fontsize=14, fontweight='bold')
                
                for idx, (ax, slice_data) in enumerate(zip(axes, selected_slices)):
                    # Denormalize slice back to 0-255 range
                    # Assuming the images were normalized (common normalization is (img - mean) / std or img / 255)
                    # For safety, we'll first check the current range and then denormalize appropriately
                    slice_min, slice_max = slice_data.min(), slice_data.max()
                    
                    if slice_min >= 0 and slice_max <= 1:
                        # Likely normalized to [0, 1], scale to [0, 255]
                        slice_denormalized = (slice_data * 255).astype(np.uint8)
                    elif slice_min >= -1 and slice_max <= 1:
                        # Likely normalized to [-1, 1], scale to [0, 255]
                        slice_denormalized = ((slice_data + 1) * 127.5).astype(np.uint8)
                    else:
                        # Assume it's already in a reasonable range, just convert to uint8
                        slice_normalized = ((slice_data - slice_min) / (slice_max - slice_min + 1e-8) * 255).astype(np.uint8)
                        slice_denormalized = slice_normalized
                    
                    # Display the slice
                    ax.imshow(slice_denormalized, cmap='gray', aspect='equal')
                    ax.set_title(f'Slice {slice_indices[idx]}')
                    ax.axis('off')
                
                # Adjust layout
                plt.tight_layout()
                
                # Log figure to Comet
                figure_name = f"Patient_{patient_id}_axial_slices_phase_{phase}"
                self.experiment.log_figure(
                    figure_name=figure_name,
                    figure=fig,
                    overwrite=True
                )
                
                # Close the figure to free memory
                plt.close(fig)
                
                # Limit number of patients to avoid too many images (optional)
                # if batch_idx >= 50:  # Log first 50 patients only
                #     break
        
        print(f"Successfully logged axial images for test set patients to Comet")


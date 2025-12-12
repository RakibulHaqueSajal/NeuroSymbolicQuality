import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score


def ranked_probability_score(y_true, y_proba):
    """Computes the ranked probability score as presented in :footcite:t:`janitza2016random`.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_proba : array-like
            Predicted probabilities.

    Returns
    -------
    rps : float
            The ranked probability score.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import ranked_probability_score
    >>> y_true = np.array([0, 0, 3, 2])
    >>> y_pred = np.array([[0.2, 0.4, 0.2, 0.2], [0.7, 0.1, 0.1, 0.1], [0.5, 0.05, 0.1, 0.35], [0.1, 0.05, 0.65, 0.2]])
    >>> ranked_probability_score(y_true, y_pred)
    0.5068750000000001
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    y_oh = np.zeros(y_proba.shape)
    y_oh[np.arange(len(y_true)), y_true] = 1

    y_oh = y_oh.cumsum(axis=1)
    y_proba = y_proba.cumsum(axis=1)

    rps = 0
    for i in range(len(y_true)):
        if y_true[i] in np.arange(y_proba.shape[1]):
            rps += np.power(y_proba[i] - y_oh[i], 2).sum()
        else:
            rps += 1
    return rps / len(y_true)


def minimum_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the sensitivity by class and returns the lowest value.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    ms : float
            Minimum sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import minimum_sensitivity
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> minimum_sensitivity(y_true, y_pred)
    0.5
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.min(sensitivities)


def accuracy_off1(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Computes the accuracy of the predictions, allowing errors if they occur in an
    adjacent class.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.
    labels : array-like or None
            Labels of the classes. If None, the labels are inferred from the data.

    Returns
    -------
    acc : float
            1-off accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import accuracy_off1
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> accuracy_off1(y_true, y_pred)
    0.8571428571428571
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1), +np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return 1.0 * np.sum(correct) / np.sum(conf_mat)

def accuracy_off1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Macro-averaged 1-off accuracy"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(y_true)
    
    class_accuracies = []
    for label in labels:
        class_mask = y_true == label
        if np.sum(class_mask) > 0:  # Avoid empty classes
            class_pred = y_pred[class_mask]
            class_true = y_true[class_mask]
            # Calculate 1-off accuracy for this class
            abs_diff = np.abs(class_pred - class_true)
            class_acc = np.mean(abs_diff <= 1)
            class_accuracies.append(class_acc)
    
    return np.mean(class_accuracies)  # Macro average

def gmsec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric Mean of the Sensitivity of the Extreme Classes (GMSEC). It was proposed
    in (:footcite:t:`vargas2024improving`) with the aim of assessing the performance of
    the classification performance for the first and the last classes.

    Parameters
    ----------
    y_true : array-like
            Target labels.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    gmec : float
            Geometric mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import gmsec
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gmsec(y_true, y_pred)
    0.7071067811865476
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.sqrt(sensitivities[0] * sensitivities[-1])


def amae(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes the average mean absolute error computed independently for each class
    as presented in :footcite:t:`baccianella2009evaluation`.

    Parameters
    ----------
    y_true : array-like
            Targets labels with one-hot or integer encoding.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    amae : float
            Average mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import amae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> amae(y_true, y_pred)
    0.125
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return np.mean(per_class_maes)


def mmae(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes the maximum mean absolute error computed independently for each class
    as presented in :footcite:t:`cruz2014metrics`.

    Parameters
    ----------
    y_true : array-like
            Target labels with one-hot or integer encoding.
    y_pred : array-like
            Predicted probabilities or labels.

    Returns
    -------
    mmae : float
            Maximum mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from dlordinal.metrics import mmae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mmae(y_true, y_pred)
    0.5
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return per_class_maes.max()

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Union, Tuple
import warnings

class ScottsPiQuadratic:
    """
    Implementation of Scott's Pi with Quadratic Weights for ordinal classification.
    
    Scott's Pi is a chance-corrected agreement measure that's particularly suitable
    for ordinal data with imbalanced class distributions. The quadratic weighting
    scheme penalizes distant misclassifications more severely than close ones.
    
    Mathematical Formula:
    π = (Po - Pe) / (1 - Pe)
    
    Where:
    - Po = Observed weighted agreement = Σᵢⱼ wᵢⱼpᵢⱼ
    - Pe = Expected weighted agreement = Σᵢⱼ wᵢⱼpᵢpⱼ, pᵢ = (pᵢ. + p.ᵢ)/2
    - wᵢⱼ = Quadratic weights = 1 - (i-j)²/(R-1)²
    """
    
    def __init__(self):
        self.weights_ = None
        self.confusion_matrix_ = None
        self.categories_ = None
    
    def _create_quadratic_weights(self, n_categories: int) -> np.ndarray:
        """
        Create quadratic weight matrix for ordinal categories.
        
        Formula: wᵢⱼ = 1 - (i-j)²/(R-1)²
        
        Args:
            n_categories: Number of ordinal categories
            
        Returns:
            Weight matrix where diagonal = 1, off-diagonal decreases quadratically
        """
        weights = np.zeros((n_categories, n_categories))
        
        for i in range(n_categories):
            for j in range(n_categories):
                if n_categories == 1:
                    weights[i, j] = 1.0
                else:
                    distance_squared = (i - j) ** 2
                    max_distance_squared = (n_categories - 1) ** 2
                    weights[i, j] = 1.0 - (distance_squared / max_distance_squared)
        
        return weights
    
    def _calculate_cell_probabilities(self, cm: np.ndarray) -> np.ndarray:
        """Calculate cell probabilities from confusion matrix."""
        total = np.sum(cm)
        if total == 0:
            raise ValueError("Confusion matrix is empty")
        return cm / total
    
    def _calculate_marginal_probabilities(self, p_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate joint marginal probabilities for Scott's Pi.
        
        Scott's Pi uses averaged marginals: pᵢ = (pᵢ. + p.ᵢ)/2
        This differs from Cohen's Kappa which uses separate marginals.
        """
        row_marginals = np.sum(p_matrix, axis=1)  # pᵢ.
        col_marginals = np.sum(p_matrix, axis=0)  # p.ᵢ
        
        # Scott's Pi uses joint proportions (averaged marginals)
        joint_marginals = (row_marginals + col_marginals) / 2
        
        return joint_marginals
    
    def _calculate_observed_agreement(self, p_matrix: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate observed weighted agreement: Po = Σᵢⱼ wᵢⱼpᵢⱼ
        """
        return np.sum(weights * p_matrix)
    
    def _calculate_expected_agreement(self, joint_marginals: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate expected weighted agreement: Pe = Σᵢⱼ wᵢⱼpᵢpⱼ
        
        For Scott's Pi, we use joint marginals: pᵢ = (pᵢ. + p.ᵢ)/2
        """
        expected_matrix = np.outer(joint_marginals, joint_marginals)
        return np.sum(weights * expected_matrix)
    
    def fit_score(self, y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]) -> float:
        """
        Calculate Scott's Pi with Quadratic Weights.
        
        Args:
            y_true: True ordinal labels
            y_pred: Predicted ordinal labels
            
        Returns:
            Scott's Pi coefficient (-1 to 1, where 1 = perfect agreement)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Get unique categories and create mapping
        all_categories = np.unique(np.concatenate([y_true, y_pred]))
        self.categories_ = all_categories
        n_categories = len(all_categories)
        
        # Create category mapping for 0-indexed confusion matrix
        category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        y_true_idx = np.array([category_to_idx[cat] for cat in y_true])
        y_pred_idx = np.array([category_to_idx[cat] for cat in y_pred])
        
        # Create confusion matrix
        self.confusion_matrix_ = confusion_matrix(y_true_idx, y_pred_idx, 
                                                labels=range(n_categories))
        
        # Create quadratic weights
        self.weights_ = self._create_quadratic_weights(n_categories)
        
        # Calculate cell probabilities
        p_matrix = self._calculate_cell_probabilities(self.confusion_matrix_)
        
        # Calculate joint marginal probabilities (Scott's Pi approach)
        joint_marginals = self._calculate_marginal_probabilities(p_matrix)
        
        # Calculate observed and expected agreements
        po = self._calculate_observed_agreement(p_matrix, self.weights_)
        pe = self._calculate_expected_agreement(joint_marginals, self.weights_)
        
        # Calculate Scott's Pi
        if pe == 1.0:
            if po == 1.0:
                return 1.0
            else:
                warnings.warn("Expected agreement is 1.0 but observed is not. "
                            "This may indicate perfect chance agreement.")
                return 0.0
        
        scotts_pi = (po - pe) / (1.0 - pe)
        
        return scotts_pi
    
    def get_weight_matrix(self) -> np.ndarray:
        """Return the quadratic weight matrix used in the last calculation."""
        if self.weights_ is None:
            raise ValueError("Must call fit_score first")
        return self.weights_.copy()
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return the confusion matrix from the last calculation."""
        if self.confusion_matrix_ is None:
            raise ValueError("Must call fit_score first")
        return self.confusion_matrix_.copy()
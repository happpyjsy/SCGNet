import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, f1_score

class AgeGenderMetrics:
    """
    Metrics for age and gender prediction
    
    Metrics for gender classification:
    - Accuracy
    - F1 Score
    
    Metrics for age estimation:
    - Mean Absolute Error (MAE)
    - Accuracy (if age is treated as classification)
    """
    
    def __init__(self, age_bins=None):
        """
        Initialize metrics
        
        Args:
            age_bins (list, optional): Age bins for classification
        """
        self.age_bins = age_bins
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        # Gender metrics
        self.gender_true = []
        self.gender_pred = []
        
        # Age metrics
        self.age_true = []
        self.age_pred = []
        self.raw_age_true = []  # Original age values for regression
        
    def update(self, outputs, targets):
        """
        Update metrics with batch results
        
        Args:
            outputs (dict): Model outputs with 'gender' and 'age' keys
            targets (dict): Ground truth with 'gender' and 'age' keys
        """
        # Gender metrics
        gender_pred = outputs['gender'].argmax(dim=1).cpu().numpy()
        gender_true = targets['gender'].cpu().numpy()
        
        self.gender_pred.extend(gender_pred)
        self.gender_true.extend(gender_true)
        
        # Age metrics
        if self.age_bins is not None:
            # Classification
            age_pred = outputs['age'].argmax(dim=1).cpu().numpy()
            age_true = targets['age'].cpu().numpy()
            
            self.age_pred.extend(age_pred)
            self.age_true.extend(age_true)
        else:
            # Regression
            age_pred = outputs['age'].cpu().numpy()
            age_true = targets['raw_age'].cpu().numpy()  # Original age values
            
            self.age_pred.extend(age_pred)
            self.age_true.extend(age_true)
            
        # Store raw age values for additional metrics
        if 'raw_age' in targets:
            self.raw_age_true.extend(targets['raw_age'].cpu().numpy())
    
    def compute(self):
        """
        Compute all metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # Gender metrics
        metrics['gender_accuracy'] = accuracy_score(self.gender_true, self.gender_pred)
        metrics['gender_f1'] = f1_score(self.gender_true, self.gender_pred, average='weighted')
        
        # Confusion matrix for gender
        gender_cm = confusion_matrix(self.gender_true, self.gender_pred, labels=[0, 1])
        metrics['gender_cm'] = gender_cm
        
        # Age metrics
        if self.age_bins is not None:
            # Classification
            metrics['age_accuracy'] = accuracy_score(self.age_true, self.age_pred)
            metrics['age_f1'] = f1_score(self.age_true, self.age_pred, average='weighted')
            
            # Convert predicted class to age range for MAE calculation
            if self.raw_age_true:
                pred_ages = np.zeros_like(self.age_pred, dtype=float)
                for i, pred_class in enumerate(self.age_pred):
                    if pred_class == 0:
                        pred_ages[i] = 1.5  # Midpoint of 0-2
                    elif pred_class == len(self.age_bins):
                        pred_ages[i] = 80  # Midpoint of 70+
                    else:
                        lower = self.age_bins[pred_class - 1]
                        upper = self.age_bins[pred_class]
                        pred_ages[i] = (lower + upper) / 2
                
                metrics['age_mae'] = mean_absolute_error(self.raw_age_true, pred_ages)
        else:
            # Regression
            metrics['age_mae'] = mean_absolute_error(self.age_true, self.age_pred)
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Get confusion matrices for gender and age
        
        Returns:
            tuple: (gender_cm, age_cm)
        """
        gender_cm = confusion_matrix(self.gender_true, self.gender_pred, labels=[0, 1])
        
        if self.age_bins is not None:
            age_cm = confusion_matrix(self.age_true, self.age_pred, 
                                      labels=list(range(len(self.age_bins) + 1)))
        else:
            age_cm = None
            
        return gender_cm, age_cm


def compute_accuracy(output, target):
    """
    Compute accuracy for classification
    
    Args:
        output (torch.Tensor): Model output
        target (torch.Tensor): Ground truth
        
    Returns:
        float: Accuracy
    """
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)

def compute_mae(output, target):
    """
    Compute Mean Absolute Error for regression
    
    Args:
        output (torch.Tensor): Model output
        target (torch.Tensor): Ground truth
        
    Returns:
        float: MAE
    """
    return torch.abs(output - target).mean().item()

def compute_metrics_batch(outputs, targets, age_bins=None):
    """
    Compute metrics for a single batch
    
    Args:
        outputs (dict): Model outputs with 'gender' and 'age' keys
        targets (dict): Ground truth with 'gender' and 'age' keys
        age_bins (list, optional): Age bins for classification
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Gender metrics
    gender_pred = outputs['gender'].argmax(dim=1)
    gender_true = targets['gender']
    metrics['gender_accuracy'] = (gender_pred == gender_true).float().mean().item()
    
    # Age metrics
    if age_bins is not None:
        # Classification
        age_pred = outputs['age'].argmax(dim=1)
        age_true = targets['age']
        metrics['age_accuracy'] = (age_pred == age_true).float().mean().item()
    else:
        # Regression
        age_pred = outputs['age']
        age_true = targets['age']
        metrics['age_mae'] = torch.abs(age_pred - age_true).mean().item()
    
    return metrics

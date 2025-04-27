import os
import torch
from datetime import datetime

class Config:
    """
    Configuration for SCGNet training and evaluation
    """
    
    def __init__(self):
        # Basic settings
        self.project_name = "SCGNet"
        self.description = "Gender and Age Detection using SCGNet"
        
        # Paths
        self.data_path = "./data/UTKFace"  # Path to UTKFace dataset
        self.output_path = "./outputs"
        self.log_dir = os.path.join(self.output_path, "logs")
        self.checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        self.results_dir = os.path.join(self.output_path, "results")
        
        # Create directories if they don't exist
        for dir_path in [self.output_path, self.log_dir, self.checkpoint_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model settings
        self.model_name = "scgnet"
        self.img_size = 224
        self.patch_size = 16
        self.in_chans = 3
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.mlp_ratio = 4.0
        self.qkv_bias = True
        self.gamma = 0.7  # Threshold for applying SATA
        self.alpha = 1.0  # Controlling factor for bounds
        
        # Task settings
        self.gender_classes = 2  # Male, Female
        self.age_as_classification = True  # True: classification, False: regression
        self.age_bins = [3, 10, 20, 30, 40, 50, 60, 70]  # Age bins for classification
        self.num_age_classes = len(self.age_bins) + 1 if self.age_as_classification else 1
        
        # Training settings
        self.batch_size = 32
        self.num_workers = 4
        self.epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.lr_scheduler = "cosine"  # "cosine", "step", "multistep", "plateau"
        self.lr_warmup_epochs = 5
        self.lr_decay_rate = 0.1
        self.lr_decay_epochs = [30, 60, 90]  # For multistep scheduler
        self.early_stopping_patience = 10
        
        # Loss weights
        self.gender_loss_weight = 1.0
        self.age_loss_weight = 1.0
        
        # Augmentation settings
        self.use_cutmix = True
        self.cutmix_prob = 0.5
        self.cutmix_beta = 1.0
        
        # Regularization
        self.dropout = 0.1
        self.drop_path_rate = 0.1
        
        # Optimizer settings
        self.optimizer = "adamw"  # "adam", "adamw", "sgd"
        self.momentum = 0.9  # For SGD
        self.beta1 = 0.9  # For Adam/AdamW
        self.beta2 = 0.999  # For Adam/AdamW
        
        # Checkpoint settings
        self.save_freq = 5  # Save checkpoint every N epochs
        self.resume = None  # Path to checkpoint to resume from
        
        # Evaluation settings
        self.eval_freq = 1  # Evaluate every N epochs
        
        # Inference settings
        self.threshold = 0.5  # Threshold for gender classification
        
        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = [0]  # List of GPU IDs to use
        
        # Experiment tracking
        self.experiment_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Visualization settings
        self.num_vis_samples = 8  # Number of samples to visualize
        
    def update(self, args):
        """
        Update config with command line arguments
        
        Args:
            args: Command line arguments
        """
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        # Update dependent settings
        if self.age_as_classification:
            self.num_age_classes = len(self.age_bins) + 1
        else:
            self.num_age_classes = 1
            
        # Create directories if they don't exist
        for dir_path in [self.output_path, self.log_dir, self.checkpoint_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
    def __str__(self):
        """String representation of config"""
        config_str = "Configuration:\n"
        for key, value in vars(self).items():
            config_str += f"  {key}: {value}\n"
        return config_str

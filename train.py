import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import time
import json

# Import project modules
from models.net import SCGNET
from data.dataset import get_dataloaders
from data.transforms import CutMix
from utils.logger import Logger, AverageMeter, Timer
from utils.metrics import AgeGenderMetrics, compute_metrics_batch
from utils.visualization import visualize_batch, plot_confusion_matrix, plot_training_history
from config import Config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, config):
    """Get optimizer based on config"""
    if config.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported")

def get_scheduler(optimizer, config, num_epochs=None):
    """Get learning rate scheduler based on config"""
    if num_epochs is None:
        num_epochs = config.epochs
        
    if config.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs - config.lr_warmup_epochs
        )
    elif config.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_decay_epochs[0],
            gamma=config.lr_decay_rate
        )
    elif config.lr_scheduler == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_decay_epochs,
            gamma=config.lr_decay_rate
        )
    elif config.lr_scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_decay_rate,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Scheduler {config.lr_scheduler} not supported")

def get_loss_fn(config):
    """Get loss functions based on config"""
    if config.age_as_classification:
        age_loss_fn = nn.CrossEntropyLoss()
    else:
        age_loss_fn = nn.L1Loss()  # MAE loss for regression
        
    gender_loss_fn = nn.CrossEntropyLoss()
    
    return {
        "gender": gender_loss_fn,
        "age": age_loss_fn
    }

class SCGNetForAgeGender(nn.Module):
    """SCGNet model for age and gender detection"""
    def __init__(self, config):
        super(SCGNetForAgeGender, self).__init__()
        self.config = config
        
        # Base SCGNet model
        self.base_model = SCGNET(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            num_classes=1000,  # Temporary value, we'll replace the head
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            gamma=config.gamma,
            alpha=config.alpha
        )
        
        # Remove the original classification head
        embed_dim = config.embed_dim * 4  # Final embedding dimension after SPM modules
        
        # Add task-specific heads
        self.gender_head = nn.Linear(embed_dim, config.gender_classes)
        
        if config.age_as_classification:
            self.age_head = nn.Linear(embed_dim, config.num_age_classes)
        else:
            self.age_head = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()  # Normalize to [0, 1]
            )
    
    def forward(self, x):
        # Get features from base model
        features = self.base_model.forward_features(x)
        
        # Task-specific predictions
        gender_logits = self.gender_head(features)
        age_output = self.age_head(features)
        
        if not self.config.age_as_classification:
            age_output = age_output.squeeze(-1)  # Remove last dimension for regression
        
        return {
            "gender": gender_logits,
            "age": age_output
        }

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config, logger):
    """Train model for one epoch"""
    model.train()
    
    # Metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    gender_losses = AverageMeter()
    age_losses = AverageMeter()
    metrics = AgeGenderMetrics(config.age_bins if config.age_as_classification else None)
    
    # CutMix augmentation
    cutmix = CutMix(beta=config.cutmix_beta, cutmix_prob=config.cutmix_prob) if config.use_cutmix else None
    
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    
    for i, batch in pbar:
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Get data
        images = batch['image'].to(device)
        gender_labels = batch['gender'].to(device)
        age_labels = batch['age'].to(device)
        
        # Apply CutMix if enabled
        if cutmix is not None:
            batch_dict = {
                'image': images,
                'gender': gender_labels,
                'age': age_labels
            }
            batch_dict = cutmix(batch_dict)
            images = batch_dict['image']
            gender_labels = batch_dict['gender']
            age_labels = batch_dict['age']
        
        # Forward pass
        outputs = model(images)
        
        # Calculate losses
        gender_loss = loss_fn['gender'](outputs['gender'], gender_labels)
        age_loss = loss_fn['age'](outputs['age'], age_labels)
        
        # Combined loss
        loss = config.gender_loss_weight * gender_loss + config.age_loss_weight * age_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        gender_losses.update(gender_loss.item(), batch_size)
        age_losses.update(age_loss.item(), batch_size)
        
        # Update evaluation metrics
        metrics.update(outputs, {
            'gender': gender_labels,
            'age': age_labels,
            'raw_age': batch['raw_age'].to(device) if 'raw_age' in batch else None
        })
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'gender_loss': f"{gender_losses.avg:.4f}",
            'age_loss': f"{age_losses.avg:.4f}",
            'batch_time': f"{batch_time.avg:.4f}s"
        })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Log metrics
    log_metrics = {
        'loss': losses.avg,
        'gender_loss': gender_losses.avg,
        'age_loss': age_losses.avg,
        **final_metrics
    }
    logger.log_metrics(log_metrics, epoch, prefix='train')
    
    return log_metrics

def validate(model, val_loader, loss_fn, device, epoch, config, logger):
    """Validate model"""
    model.eval()
    
    # Metrics
    losses = AverageMeter()
    gender_losses = AverageMeter()
    age_losses = AverageMeter()
    metrics = AgeGenderMetrics(config.age_bins if config.age_as_classification else None)
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, batch in pbar:
            # Get data
            images = batch['image'].to(device)
            gender_labels = batch['gender'].to(device)
            age_labels = batch['age'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate losses
            gender_loss = loss_fn['gender'](outputs['gender'], gender_labels)
            age_loss = loss_fn['age'](outputs['age'], age_labels)
            
            # Combined loss
            loss = config.gender_loss_weight * gender_loss + config.age_loss_weight * age_loss
            
            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            gender_losses.update(gender_loss.item(), batch_size)
            age_losses.update(age_loss.item(), batch_size)
            
            # Update evaluation metrics
            metrics.update(outputs, {
                'gender': gender_labels,
                'age': age_labels,
                'raw_age': batch['raw_age'].to(device) if 'raw_age' in batch else None
            })
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'gender_loss': f"{gender_losses.avg:.4f}",
                'age_loss': f"{age_losses.avg:.4f}"
            })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Visualize a batch of predictions
    if epoch % 5 == 0 or epoch == config.epochs - 1:
        # Get a batch for visualization
        for batch in val_loader:
            images = batch['image'].to(device)
            gender_labels = batch['gender']
            age_labels = batch['age']
            
            with torch.no_grad():
                outputs = model(images)
            
            # Visualize
            fig = visualize_batch(
                images.cpu(),
                outputs['gender'].cpu(),
                outputs['age'].cpu(),
                gender_labels,
                age_labels,
                config.age_bins if config.age_as_classification else None,
                num_images=min(config.num_vis_samples, images.size(0)),
                save_path=os.path.join(config.results_dir, f"val_predictions_epoch_{epoch}.png")
            )
            
            # Log to TensorBoard
            logger.log_image(f"val_predictions", fig, epoch)
            break
    
    # Log metrics
    log_metrics = {
        'loss': losses.avg,
        'gender_loss': gender_losses.avg,
        'age_loss': age_losses.avg,
        **final_metrics
    }
    logger.log_metrics(log_metrics, epoch, prefix='val')
    
    return log_metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint
    if epoch % config.save_freq == 0 or epoch == config.epochs - 1:
        checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")

def train(config):
    """Main training function"""
    # Set random seed
    set_seed(42)
    
    # Create logger
    logger = Logger(config.log_dir, name=config.experiment_name)
    logger.log_hyperparams(vars(config))
    
    # Get dataloaders
    dataloaders = get_dataloaders(
        config.data_path,
        batch_size=config.batch_size,
        age_bins=config.age_bins if config.age_as_classification else None,
        num_workers=config.num_workers
    )
    
    # Create model
    model = SCGNetForAgeGender(config)
    model = model.to(config.device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    
    # Get optimizer, scheduler, and loss function
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    loss_fn = get_loss_fn(config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float('inf') if config.age_as_classification else float('inf')  # Lower is better for MAE
    history = {
        'train_loss': [],
        'train_gender_accuracy': [],
        'val_loss': [],
        'val_gender_accuracy': [],
    }
    
    if config.age_as_classification:
        history['train_age_accuracy'] = []
        history['val_age_accuracy'] = []
    else:
        history['train_age_mae'] = []
        history['val_age_mae'] = []
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(f"Loading checkpoint from {config.resume}")
            checkpoint = torch.load(config.resume, map_location=config.device)
            
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint['scheduler_state_dict'] is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'metrics' in checkpoint and 'val_age_mae' in checkpoint['metrics']:
                best_metric = checkpoint['metrics']['val_age_mae']
            elif 'metrics' in checkpoint and 'val_age_accuracy' in checkpoint['metrics']:
                best_metric = -checkpoint['metrics']['val_age_accuracy']  # Higher is better for accuracy
            
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {config.resume}")
    
    # Training loop
    timer = Timer()
    timer.start()
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, dataloaders['train'], optimizer, loss_fn, 
            config.device, epoch, config, logger
        )
        
        # Validate
        if epoch % config.eval_freq == 0 or epoch == config.epochs - 1:
            val_metrics = validate(
                model, dataloaders['val'], loss_fn, 
                config.device, epoch, config, logger
            )
            
            # Update learning rate
            if config.lr_scheduler == "plateau":
                if config.age_as_classification:
                    scheduler.step(val_metrics['age_accuracy'])
                else:
                    scheduler.step(val_metrics['age_mae'])
            else:
                scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_gender_accuracy'].append(train_metrics['gender_accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_gender_accuracy'].append(val_metrics['gender_accuracy'])
            
            if config.age_as_classification:
                history['train_age_accuracy'].append(train_metrics['age_accuracy'])
                history['val_age_accuracy'].append(val_metrics['age_accuracy'])
                current_metric = -val_metrics['age_accuracy']  # Higher is better
            else:
                history['train_age_mae'].append(train_metrics['age_mae'])
                history['val_age_mae'].append(val_metrics['age_mae'])
                current_metric = val_metrics['age_mae']  # Lower is better
            
            # Check if current model is the best
            is_best = current_metric < best_metric
            if is_best:
                best_metric = current_metric
                print(f"New best model with metric: {-best_metric if config.age_as_classification else best_metric:.4f}")
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                val_metrics, config, is_best
            )
            
            # Plot and save training history
            if epoch > 0:
                plot_training_history(
                    history,
                    save_path=os.path.join(config.results_dir, "training_history.png")
                )
        else:
            # Update learning rate
            if config.lr_scheduler != "plateau":
                scheduler.step()
        
        # Early stopping
        if config.early_stopping_patience > 0:
            if len(history['val_loss']) > config.early_stopping_patience:
                recent_losses = history['val_loss'][-config.early_stopping_patience:]
                if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Training complete
    timer.stop()
    print(f"Training completed in {timer.elapsed_time_str()}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    torch.save({
        'epoch': config.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'metrics': val_metrics if 'val_metrics' in locals() else None,
        'config': config
    }, final_path)
    print(f"Final model saved to {final_path}")
    
    # Save training history
    with open(os.path.join(config.results_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
    
    # Close logger
    logger.close()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(
        model, dataloaders['test'], loss_fn, 
        config.device, config.epochs, config, logger
    )
    
    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save test metrics
    with open(os.path.join(config.results_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)
    
    return model, test_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train SCGNet for gender and age detection")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to UTKFace dataset")
    parser.add_argument("--output_path", type=str, help="Path to save outputs")
    
    # Model arguments
    parser.add_argument("--img_size", type=int, help="Input image size")
    parser.add_argument("--embed_dim", type=int, help="Embedding dimension")
    parser.add_argument("--depth", type=int, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    
    # Task arguments
    parser.add_argument("--age_as_classification", action="store_true", help="Treat age as classification")
    parser.add_argument("--age_bins", type=int, nargs="+", help="Age bins for classification")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer", type=str, help="Optimizer")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, help="Learning rate scheduler")
    
    # Regularization arguments
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--use_cutmix", action="store_true", help="Use CutMix augmentation")
    
    # Checkpoint arguments
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, help="Number of workers")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = Config()
    
    # Update config with command line arguments
    config.update(args)
    
    # Print config
    print(config)
    
    # Train model
    model, test_metrics = train(config)

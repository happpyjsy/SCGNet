import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from models.net import SCGNET
from data.dataset import get_dataloaders
from utils.metrics import AgeGenderMetrics
from utils.visualization import visualize_batch, plot_confusion_matrix
from config import Config
from train import SCGNetForAgeGender

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate SCGNet for gender and age detection")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to UTKFace dataset")
    parser.add_argument("--output_path", type=str, help="Path to save outputs")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Evaluation arguments
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to visualize")
    
    return parser.parse_args()

def evaluate(model, dataloader, device, config, output_dir):
    """Evaluate model on dataset"""
    model.eval()
    
    # Metrics
    metrics = AgeGenderMetrics(config.age_bins if config.age_as_classification else None)
    
    # Lists to store predictions and ground truth
    all_gender_preds = []
    all_gender_true = []
    all_age_preds = []
    all_age_true = []
    all_images = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    
    with torch.no_grad():
        for i, batch in pbar:
            # Get data
            images = batch['image'].to(device)
            gender_labels = batch['gender'].to(device)
            age_labels = batch['age'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics.update(outputs, {
                'gender': gender_labels,
                'age': age_labels,
                'raw_age': batch['raw_age'].to(device) if 'raw_age' in batch else None
            })
            
            # Store predictions and ground truth
            gender_preds = outputs['gender'].argmax(dim=1).cpu()
            all_gender_preds.extend(gender_preds.tolist())
            all_gender_true.extend(gender_labels.cpu().tolist())
            
            if config.age_as_classification:
                age_preds = outputs['age'].argmax(dim=1).cpu()
                all_age_preds.extend(age_preds.tolist())
            else:
                age_preds = outputs['age'].cpu()
                all_age_preds.extend(age_preds.tolist())
            
            all_age_true.extend(batch['raw_age'].tolist() if 'raw_age' in batch else age_labels.cpu().tolist())
            
            # Store images for visualization
            if len(all_images) < config.num_vis_samples:
                num_to_add = min(config.num_vis_samples - len(all_images), images.size(0))
                all_images.extend([images[j].cpu() for j in range(num_to_add)])
    
    # Compute final metrics
    final_metrics = metrics.compute()
    
    # Print metrics
    print("\nEvaluation Results:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in final_metrics.items()}, f, indent=4)
    
    # Gender confusion matrix
    gender_cm = confusion_matrix(all_gender_true, all_gender_preds)
    gender_cm_fig = plot_confusion_matrix(
        gender_cm, 
        classes=["Male", "Female"],
        title="Gender Confusion Matrix",
        save_path=os.path.join(output_dir, "gender_confusion_matrix.png")
    )
    
    # Gender classification report
    gender_report = classification_report(
        all_gender_true, 
        all_gender_preds,
        target_names=["Male", "Female"],
        output_dict=True
    )
    
    with open(os.path.join(output_dir, "gender_classification_report.json"), "w") as f:
        json.dump(gender_report, f, indent=4)
    
    # Age confusion matrix if classification
    if config.age_as_classification:
        age_class_names = []
        for i in range(len(config.age_bins) + 1):
            if i == 0:
                age_class_names.append("0-2")
            elif i == len(config.age_bins):
                age_class_names.append(f"{config.age_bins[-1]}+")
            else:
                age_class_names.append(f"{config.age_bins[i-1]}-{config.age_bins[i]}")
        
        age_cm = confusion_matrix(
            all_age_true, 
            all_age_preds,
            labels=list(range(len(config.age_bins) + 1))
        )
        
        age_cm_fig = plot_confusion_matrix(
            age_cm, 
            classes=age_class_names,
            title="Age Confusion Matrix",
            save_path=os.path.join(output_dir, "age_confusion_matrix.png")
        )
        
        # Age classification report
        age_report = classification_report(
            all_age_true, 
            all_age_preds,
            target_names=age_class_names,
            output_dict=True
        )
        
        with open(os.path.join(output_dir, "age_classification_report.json"), "w") as f:
            json.dump(age_report, f, indent=4)
    else:
        # Age regression scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(all_age_true, all_age_preds, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel("True Age")
        plt.ylabel("Predicted Age")
        plt.title("Age Prediction")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "age_scatter_plot.png"))
        plt.close()
    
    # Visualize a batch of predictions
    if all_images:
        # Convert to batch
        vis_images = torch.stack(all_images[:config.num_vis_samples])
        
        # Get predictions for visualization
        model.eval()
        with torch.no_grad():
            vis_outputs = model(vis_images.to(device))
        
        # Visualize
        fig = visualize_batch(
            vis_images,
            vis_outputs['gender'].cpu(),
            vis_outputs['age'].cpu(),
            None,  # No ground truth for visualization
            None,
            config.age_bins if config.age_as_classification else None,
            num_images=min(config.num_vis_samples, vis_images.size(0)),
            save_path=os.path.join(output_dir, "sample_predictions.png")
        )
    
    return final_metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Using config from checkpoint")
    else:
        config = Config()
        print("Using default config")
    
    # Update config with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_path:
        config.output_path = args.output_path
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Set up output directory
    output_dir = os.path.join(config.results_dir, f"evaluation_{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SCGNetForAgeGender(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get dataloader
    dataloaders = get_dataloaders(
        config.data_path,
        batch_size=config.batch_size,
        age_bins=config.age_bins if config.age_as_classification else None,
        num_workers=4
    )
    
    dataloader = dataloaders[args.split]
    
    # Evaluate model
    print(f"Evaluating model on {args.split} set...")
    metrics = evaluate(model, dataloader, device, config, output_dir)
    
    print("\nEvaluation complete.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

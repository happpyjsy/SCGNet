import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import seaborn as sns
from PIL import Image
import cv2
import os

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image with mean and standard deviation
    
    Args:
        tensor (torch.Tensor): Tensor image of size (C, H, W)
        mean (list): Mean for each channel
        std (list): Standard deviation for each channel
        
    Returns:
        torch.Tensor: Denormalized image
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    img = tensor.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    
    return img

def visualize_batch(images, gender_preds, age_preds, gender_true=None, age_true=None, 
                   age_bins=None, num_images=8, save_path=None):
    """
    Visualize a batch of images with predictions
    
    Args:
        images (torch.Tensor): Batch of images
        gender_preds (torch.Tensor): Gender predictions
        age_preds (torch.Tensor): Age predictions
        gender_true (torch.Tensor, optional): True gender labels
        age_true (torch.Tensor, optional): True age labels
        age_bins (list, optional): Age bins for classification
        num_images (int): Number of images to visualize
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Limit the number of images
    num_images = min(num_images, images.size(0))
    
    # Create figure
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Process each image
    for i in range(num_images):
        # Denormalize image
        img = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
        
        # Get gender prediction
        gender_pred = "Male" if gender_preds[i].argmax().item() == 0 else "Female"
        gender_conf = torch.softmax(gender_preds[i], dim=0).max().item() * 100
        
        # Get age prediction
        if age_bins is not None:
            # Classification
            age_class = age_preds[i].argmax().item()
            if age_class == 0:
                age_pred = "0-2"
            elif age_class == len(age_bins):
                age_pred = f"{age_bins[-1]}+"
            else:
                age_pred = f"{age_bins[age_class-1]}-{age_bins[age_class]}"
        else:
            # Regression
            age_pred = f"{age_preds[i].item() * 116:.1f}"
        
        # Display image
        axes[i].imshow(img)
        
        # Display predictions
        title = f"Pred: {gender_pred} ({gender_conf:.1f}%), Age: {age_pred}"
        
        # Add ground truth if available
        if gender_true is not None and age_true is not None:
            gender_true_label = "Male" if gender_true[i].item() == 0 else "Female"
            
            if age_bins is not None:
                age_class_true = age_true[i].item()
                if age_class_true == 0:
                    age_true_label = "0-2"
                elif age_class_true == len(age_bins):
                    age_true_label = f"{age_bins[-1]}+"
                else:
                    age_true_label = f"{age_bins[age_class_true-1]}-{age_bins[age_class_true]}"
            else:
                age_true_label = f"{age_true[i].item() * 116:.1f}"
                
            title += f"\nTrue: {gender_true_label}, Age: {age_true_label}"
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): List of class names
        title (str): Title of the plot
        cmap (matplotlib.colors.Colormap): Colormap
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Gender accuracy plot
    axes[0, 1].plot(history['train_gender_accuracy'], label='Train')
    axes[0, 1].plot(history['val_gender_accuracy'], label='Validation')
    axes[0, 1].set_title('Gender Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Age metric plot (MAE or accuracy)
    if 'train_age_mae' in history:
        axes[1, 0].plot(history['train_age_mae'], label='Train')
        axes[1, 0].plot(history['val_age_mae'], label='Validation')
        axes[1, 0].set_title('Age MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
    else:
        axes[1, 0].plot(history['train_age_accuracy'], label='Train')
        axes[1, 0].plot(history['val_age_accuracy'], label='Validation')
        axes[1, 0].set_title('Age Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Learning rate plot
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_attention(model, image, layer_idx=11, head_idx=0, save_path=None):
    """
    Visualize attention maps from a Vision Transformer model
    
    Args:
        model (torch.nn.Module): Vision Transformer model
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        layer_idx (int): Index of the transformer layer to visualize
        head_idx (int): Index of the attention head to visualize
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Register hook to get attention maps
    attention_maps = []
    
    def get_attention(module, input, output):
        attention_maps.append(output)
    
    # Register forward hook on the specified layer's attention module
    hook = model.blocks[layer_idx].attn.register_forward_hook(get_attention)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hook
    hook.remove()
    
    # Get attention map for the specified head
    attention = attention_maps[0]  # (B, num_heads, seq_len, seq_len)
    attention = attention[0, head_idx].cpu().numpy()  # (seq_len, seq_len)
    
    # Remove CLS token attention if present
    if attention.shape[0] > 1:
        attention = attention[1:, 1:]  # Remove CLS token
    
    # Reshape attention map to image dimensions
    h = w = int(np.sqrt(attention.shape[0]))
    attention_map = attention.reshape(h, w, h, w)
    
    # Average attention across all query positions
    attention_map = attention_map.mean(axis=(0, 1))
    
    # Resize attention map to match input image size
    attention_map = cv2.resize(attention_map, (image.shape[2], image.shape[3]))
    
    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = denormalize(image[0]).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention_map, cmap='viridis')
    axes[1].set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
    axes[1].axis('off')
    
    # Overlay attention on image
    axes[2].imshow(img)
    axes[2].imshow(attention_map, alpha=0.5, cmap='viridis')
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_gradcam(model, image, target_layer, target_category=None, save_path=None):
    """
    Visualize Grad-CAM for a Vision Transformer model
    
    Args:
        model (torch.nn.Module): Vision Transformer model
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        target_layer (torch.nn.Module): Target layer for Grad-CAM
        target_category (int, optional): Target category for Grad-CAM
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Forward pass
    image.requires_grad_()
    outputs = model(image)
    
    if target_category is None:
        if isinstance(outputs, dict):
            # Multi-task model
            target_category = outputs['gender'].argmax(dim=1)
        else:
            # Single-task model
            target_category = outputs.argmax(dim=1)
    
    # Get gradients
    model.zero_grad()
    
    if isinstance(outputs, dict):
        # Multi-task model (use gender output)
        score = outputs['gender'][:, target_category]
    else:
        # Single-task model
        score = outputs[:, target_category]
    
    score.backward()
    
    # Get feature maps
    feature_maps = target_layer.output
    gradients = target_layer.gradients
    
    # Calculate weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    
    # Calculate CAM
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    
    # Normalize CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Resize CAM to match input image size
    cam = torch.nn.functional.interpolate(
        cam, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = denormalize(image[0]).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM
    cam = cam[0, 0].cpu().numpy()
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    
    # Overlay Grad-CAM on image
    axes[2].imshow(img)
    axes[2].imshow(cam, alpha=0.5, cmap='jet')
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

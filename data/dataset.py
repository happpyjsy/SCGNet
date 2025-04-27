import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import pandas as pd
import random

class UTKFaceDataset(Dataset):
    """
    UTKFace dataset for gender and age detection
    
    The dataset contains over 20,000 face images with annotations of age, gender, and ethnicity.
    Each image is annotated with:
    [age]_[gender]_[race]_[date&time].jpg
    where:
    - age is an integer from 0 to 116
    - gender is 0 (male) or 1 (female)
    - race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others
    """
    
    def __init__(self, root_dir, split='train', transform=None, age_bins=None):
        """
        Args:
            root_dir (string): Directory with the UTKFace dataset
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            age_bins (list, optional): Age bins for classification. If None, age is treated as regression
        """
        self.root_dir = root_dir
        self.transform = transform
        self.age_bins = age_bins
        
        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Filter out files without proper annotations
        self.image_files = [f for f in self.image_files if len(f.split('_')) >= 3]
        
        # Create train/val/test split (80/10/10)
        random.seed(42)  # For reproducibility
        random.shuffle(self.image_files)
        
        n_samples = len(self.image_files)
        if split == 'train':
            self.image_files = self.image_files[:int(0.8 * n_samples)]
        elif split == 'val':
            self.image_files = self.image_files[int(0.8 * n_samples):int(0.9 * n_samples)]
        elif split == 'test':
            self.image_files = self.image_files[int(0.9 * n_samples):]
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train', 'val', or 'test'")
        
        print(f"Loaded {len(self.image_files)} images for {split}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        # Parse filename to get labels
        try:
            age, gender, *_ = self.image_files[idx].split('_')
            age = int(age)
            gender = int(gender)
        except (ValueError, IndexError):
            # If parsing fails, use default values
            age, gender = 0, 0
            
        # Handle age bins if provided
        if self.age_bins is not None:
            age_class = np.digitize(age, self.age_bins) - 1
            age_label = torch.tensor(age_class, dtype=torch.long)
        else:
            # Normalize age to [0, 1] for regression
            age_label = torch.tensor(age / 116.0, dtype=torch.float)
        
        gender_label = torch.tensor(gender, dtype=torch.long)
        
        # Load and transform image
        try:
            image = Image.open(img_name).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                image = transforms.ToTensor()(image)
            
            return {
                'image': image,
                'age': age_label,
                'gender': gender_label,
                'raw_age': age  # Store original age for evaluation
            }
            
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder in case of error
            placeholder = torch.zeros((3, 224, 224))
            return {
                'image': placeholder,
                'age': age_label,
                'gender': gender_label,
                'raw_age': age
            }

def get_dataloaders(data_path, batch_size=32, age_bins=None, num_workers=4):
    """
    Create dataloaders for training, validation, and testing
    
    Args:
        data_path (string): Path to UTKFace dataset
        batch_size (int): Batch size
        age_bins (list): Age bins for classification
        num_workers (int): Number of workers for dataloader
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = UTKFaceDataset(data_path, split='train', transform=train_transform, age_bins=age_bins)
    val_dataset = UTKFaceDataset(data_path, split='val', transform=val_transform, age_bins=age_bins)
    test_dataset = UTKFaceDataset(data_path, split='test', transform=val_transform, age_bins=age_bins)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Define age bins (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
    age_bins = [3, 10, 20, 30, 40, 50, 60, 70]
    
    # Create a test dataset
    dataset = UTKFaceDataset("path/to/UTKFace", split='train', age_bins=age_bins)
    
    # Display a sample
    sample = dataset[0]
    img = sample['image']
    
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
        
    plt.imshow(img)
    plt.title(f"Gender: {'Female' if sample['gender'] == 1 else 'Male'}, Age: {sample['raw_age']}")
    plt.show()

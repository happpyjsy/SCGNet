import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image

class RandomErasing(object):
    """
    Randomly erase a rectangular region from the image.
    This helps the model be more robust to occlusions.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, value=0):
        self.probability = probability
        self.sl = sl  # min erasing area
        self.sh = sh  # max erasing area
        self.r1 = r1  # min aspect ratio
        self.r2 = r2  # max aspect ratio
        self.value = value  # erasing value
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        if isinstance(img, torch.Tensor):
            img_c, img_h, img_w = img.shape
            img_area = img_h * img_w
            
            for _ in range(100):
                target_area = random.uniform(self.sl, self.sh) * img_area
                aspect_ratio = random.uniform(self.r1, self.r2)
                
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                
                if w < img_w and h < img_h:
                    x1 = random.randint(0, img_h - h)
                    y1 = random.randint(0, img_w - w)
                    
                    if img_c == 3:
                        img[0, x1:x1+h, y1:y1+w] = self.value
                        img[1, x1:x1+h, y1:y1+w] = self.value
                        img[2, x1:x1+h, y1:y1+w] = self.value
                    else:
                        img[0, x1:x1+h, y1:y1+w] = self.value
                    return img
            
            return img
        else:
            # Handle PIL Image
            img_c, img_h, img_w = img.split(), img.height, img.width
            img_area = img_h * img_w
            
            for _ in range(100):
                target_area = random.uniform(self.sl, self.sh) * img_area
                aspect_ratio = random.uniform(self.r1, self.r2)
                
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                
                if w < img_w and h < img_h:
                    x1 = random.randint(0, img_h - h)
                    y1 = random.randint(0, img_w - w)
                    
                    if len(img_c) == 3:
                        pixel = Image.new('RGB', (w, h), (self.value, self.value, self.value))
                        img.paste(pixel, (y1, x1))
                    else:
                        pixel = Image.new('L', (w, h), self.value)
                        img.paste(pixel, (y1, x1))
                    return img
            
            return img

class CutMix(object):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    https://arxiv.org/abs/1905.04899
    """
    def __init__(self, beta=1.0, cutmix_prob=0.5):
        self.beta = beta
        self.cutmix_prob = cutmix_prob
        
    def __call__(self, batch):
        """
        Apply CutMix to a batch of images
        
        Args:
            batch (dict): Batch containing 'image', 'age', 'gender'
            
        Returns:
            dict: Modified batch with CutMix applied
        """
        if random.random() > self.cutmix_prob:
            return batch
            
        images = batch['image']
        ages = batch['age']
        genders = batch['gender']
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.beta, self.beta)
        
        # Get random box coordinates
        h, w = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust labels
        mixed_ages = lam * ages + (1 - lam) * ages[indices]
        mixed_genders = lam * genders.float() + (1 - lam) * genders[indices].float()
        
        batch['image'] = images
        batch['age'] = mixed_ages
        batch['gender'] = mixed_genders
        
        return batch

def get_train_transforms(img_size=224):
    """
    Get transforms for training
    
    Args:
        img_size (int): Image size
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing()
    ])

def get_val_transforms(img_size=224):
    """
    Get transforms for validation and testing
    
    Args:
        img_size (int): Image size
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

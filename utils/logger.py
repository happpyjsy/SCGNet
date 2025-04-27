import os
import logging
import time
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Logger class for training and evaluation
    
    Features:
    - Console logging
    - File logging
    - TensorBoard logging
    """
    
    def __init__(self, log_dir, name=None):
        """
        Initialize logger
        
        Args:
            log_dir (str): Directory to save logs
            name (str, optional): Logger name
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler
        self.logger = logging.getLogger(name or "scgnet")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        
        self.step = 0
        self.epoch = 0
        
    def log_metrics(self, metrics, step=None, prefix='train'):
        """
        Log metrics to console, file, and TensorBoard
        
        Args:
            metrics (dict): Dictionary of metrics
            step (int, optional): Step number
            prefix (str): Prefix for metrics (train/val/test)
        """
        if step is not None:
            self.step = step
        
        # Log to console and file
        message = f"{prefix.capitalize()} - Step: {self.step}"
        for key, value in metrics.items():
            message += f", {key}: {value:.4f}"
        self.logger.info(message)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.step)
    
    def log_hyperparams(self, hparams):
        """
        Log hyperparameters
        
        Args:
            hparams (dict): Dictionary of hyperparameters
        """
        # Log to console and file
        message = "Hyperparameters:"
        for key, value in hparams.items():
            message += f" {key}: {value},"
        self.logger.info(message)
        
        # Log to TensorBoard
        self.writer.add_text("hyperparameters", str(hparams), 0)
    
    def log_image(self, tag, img_tensor, step=None):
        """
        Log image to TensorBoard
        
        Args:
            tag (str): Image tag
            img_tensor (torch.Tensor): Image tensor
            step (int, optional): Step number
        """
        if step is not None:
            self.step = step
            
        self.writer.add_image(tag, img_tensor, self.step)
    
    def log_model_graph(self, model, input_tensor):
        """
        Log model graph to TensorBoard
        
        Args:
            model (torch.nn.Module): Model
            input_tensor (torch.Tensor): Input tensor
        """
        self.writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    """Timer for measuring execution time"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start timer"""
        self.start_time = time.time()
        
    def stop(self):
        """Stop timer"""
        self.end_time = time.time()
        
    def elapsed_time(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def elapsed_time_str(self):
        """Get elapsed time as string"""
        elapsed = self.elapsed_time()
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class SID_Logger:
    """
    A comprehensive logging class for CNN PyTorch training programs.
    Supports both file and console logging with configurable levels and formats.
    """
    
    def __init__(self, 
                 name: str = 'training',
                 log_dir: str = 'logs',
                 log_level: int = logging.INFO,
                 console_output: bool = True,
                 file_output: bool = True,
                 timestamp_format: str = '%Y%m%d_%H%M%S'):
        """
        Initialize the CNN Logger.
        
        Args:
            name: Logger name (default: 'training')
            log_dir: Directory to store log files (default: 'logs')
            log_level: Logging level (default: logging.INFO)
            console_output: Enable console output (default: True)
            file_output: Enable file output (default: True)
            timestamp_format: Format for timestamp in log filenames
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_output = console_output
        self.file_output = file_output
        self.timestamp_format = timestamp_format
        self._logger: Optional[logging.Logger] = None
        self._setup()
    
    def _setup(self):
        """
        Set up and configure the logger.
        
        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if self._logger and self._logger.handlers:
            return self._logger
            
        # Create logs directory
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or create logger
        self._logger = logging.getLogger(self.name)
        
        # Clear existing handlers to avoid duplicates
        self._logger.handlers.clear()
        
        # Set logger level
        self._logger.setLevel(self.log_level)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if enabled
        if self.file_output:
            timestamp = datetime.now().strftime(self.timestamp_format)
            log_file = self.log_dir / f'{self.name}_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self._logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            self._logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self._logger.propagate = False
        
        return
    
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance, setting it up if necessary."""
        if self._logger is None:
            return self._setup()
        return self._logger
    
    def log_model_info(self, model, total_params: int = None, trainable_params: int = None):
        """
        Log CNN model information.
        
        Args:
            model: PyTorch model
            total_params: Total number of parameters
            trainable_params: Number of trainable parameters
        """
        self.logger.info("=" * 50)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 50)
        self.logger.info(f"Model Architecture: {model.__class__.__name__}")
        
        if total_params:
            self.logger.info(f"Total Parameters: {total_params:,}")
        if trainable_params:
            self.logger.info(f"Trainable Parameters: {trainable_params:,}")
            
        self.logger.info("=" * 50)
    
    def log_training_config(self, config: dict):
        """
        Log training configuration parameters.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.logger.info("=" * 50)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 50)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 50)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of a training epoch."""
        self.logger.info(f"Starting Epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_results(self, epoch: int, train_loss: float, train_acc: float = None, 
                         val_loss: float = None, val_acc: float = None, 
                         learning_rate: float = None, epoch_time: float = None):
        """
        Log epoch training results.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy (optional)
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
            learning_rate: Current learning rate (optional)
            epoch_time: Time taken for epoch in seconds (optional)
        """
        msg = f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}"
        
        if train_acc is not None:
            msg += f", Train Acc: {train_acc:.4f}"
        if val_loss is not None:
            msg += f", Val Loss: {val_loss:.4f}"
        if val_acc is not None:
            msg += f", Val Acc: {val_acc:.4f}"
        if learning_rate is not None:
            msg += f", LR: {learning_rate:.6f}"
        if epoch_time is not None:
            msg += f", Time: {epoch_time:.2f}s"
            
        self.logger.info(msg)
    
    def log_batch_progress(self, batch_idx: int, total_batches: int, loss: float, 
                          batch_size: int, log_interval: int = 100):
        """
        Log batch-level progress during training.
        
        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches
            loss: Current batch loss
            batch_size: Size of current batch
            log_interval: Log every N batches
        """
        if batch_idx % log_interval == 0:
            progress = 100.0 * batch_idx / total_batches
            self.logger.info(f'Train Batch: {batch_idx}/{total_batches} '
                           f'({progress:.1f}%) Loss: {loss:.6f}')
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log when a new best model is saved."""
        self.logger.info(f"New best model saved at epoch {epoch + 1}! "
                        f"{metric_name}: {metric_value:.4f}")
    
    def log_training_complete(self, total_time: float, best_metric: float = None, 
                            best_epoch: int = None):
        """
        Log training completion summary.
        
        Args:
            total_time: Total training time in seconds
            best_metric: Best achieved metric value
            best_epoch: Epoch where best metric was achieved
        """
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        if best_metric is not None and best_epoch is not None:
            self.logger.info(f"Best Performance: {best_metric:.4f} at epoch {best_epoch + 1}")
        
        self.logger.info("=" * 50)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


# Example usage in your CNN training program
# if __name__ == "__main__":
#     import torch
#     import torch.nn as nn
    
#     # Initialize the logger
#     cnn_logger = Logger(
#         name='deepfake_cnn',
#         log_dir='training_logs',
#         log_level=logging.INFO
#     )
    
#     # Set up logger
#     logger = cnn_logger.setup()
    
#     # Example model parameter calculation
#     class DeepfakeCNN(nn.Module):
#         def __init__(self, num_classes=2):
#             super().__init__()
#             self.features = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#             )
#             self.classifier = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.Linear(128 * 56 * 56, 512),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(512, num_classes)
#             )
        
#         def forward(self, x):
#             x = self.features(x)
#             x = torch.flatten(x, 1)
#             x = self.classifier(x)
#             return x
    
#     # Create model and log info
#     model = DeepfakeCNN()
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     # Log training configuration
#     config = {
#         'model': 'DeepfakeCNN',
#         'dataset': 'Custom Deepfake Dataset', 
#         'batch_size': 32,
#         'learning_rate': 0.001,
#         'optimizer': 'Adam',
#         'loss_function': 'CrossEntropyLoss',
#         'epochs': 50,
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu'
#     }
    
#     cnn_logger.log_training_config(config)
#     cnn_logger.log_model_info(model, total_params, trainable_params)
    
#     # Example training loop usage
#     epochs = 5
#     for epoch in range(epochs):
#         cnn_logger.log_epoch_start(epoch, epochs)
        
#         # Simulate training metrics
#         train_loss = 0.8 - (epoch * 0.1)
#         train_acc = 0.6 + (epoch * 0.08)
#         val_loss = 0.9 - (epoch * 0.08)  
#         val_acc = 0.55 + (epoch * 0.07)
        
#         # Log epoch results
#         cnn_logger.log_epoch_results(
#             epoch=epoch,
#             train_loss=train_loss,
#             train_acc=train_acc,
#             val_loss=val_loss,
#             val_acc=val_acc,
#             learning_rate=0.001,
#             epoch_time=180.5
#         )
    
#     cnn_logger.log_training_complete(900.0, 0.87, 4)
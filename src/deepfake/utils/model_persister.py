import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class IModelPersister(ABC):
    """Interface for model persistence operations"""
    
    @abstractmethod
    def save_model(self, model: torch.nn.Module, path: str, **kwargs) -> str:
        """Save model to specified path"""
        pass
    
    @abstractmethod
    def load_model(self, model: torch.nn.Module, path: str, **kwargs) -> Dict[str, Any]:
        """Load model from specified path"""
        pass

class TorchModelPersister(IModelPersister):
    """PyTorch implementation of model persistence"""
    
    def save_model(self, model: torch.nn.Module, path: str, 
                  optimizer: Optional[torch.optim.Optimizer] = None, **kwargs) -> str:
        """Save PyTorch model with optional optimizer state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {"model_state_dict": model.state_dict()}
        
        if optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add any additional metadata
        if kwargs:
            save_dict.update(kwargs)
        
        torch.save(save_dict, path)
        return path
    
    def load_model(self, model: torch.nn.Module, path: str, 
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  device: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Load PyTorch model with optional optimizer state"""
        checkpoint = torch.load(path, map_location=torch.device(device) if device else None)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Load optimizer state if available
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('additional_info', {})
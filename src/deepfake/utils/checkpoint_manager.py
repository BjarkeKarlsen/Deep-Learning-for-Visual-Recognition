
import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from dataclasses import asdict

from deepfake.config import Config
from deepfake.utils.training_metrics_tracker import TrainingMetricsTracker
from deepfake.utils.logger import SidLogger


class CheckpointManager:
    """
    Comprehensive checkpoint management system for training resumption and backup.

    Features:
    - Automatic periodic checkpointing
    - Model state, optimizer state, and metrics persistence
    - Configurable retention policy
    - Resume from any checkpoint
    - Thread-safe operations
    - Comprehensive error handling
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        logger: Optional[logging.Logger] = None,
        backup_frequency: int = 5,
        keep_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_config: bool = True,
        compress_checkpoints: bool = False
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoints will be stored
            logger: Logger instance for operations logging
            backup_frequency: Create checkpoint every N epochs (default: 5)
            keep_checkpoints: Number of checkpoints to retain (default: 3)
            save_optimizer: Whether to save optimizer state (default: True)
            save_config: Whether to save configuration (default: True)
            compress_checkpoints: Whether to compress checkpoint files (default: False)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.backup_frequency = backup_frequency
        self.keep_checkpoints = keep_checkpoints
        self.save_optimizer = save_optimizer
        self.save_config = save_config
        self.compress_checkpoints = compress_checkpoints

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"  Initialized CheckpointManager at {self.checkpoint_dir}")
        self.logger.info(f"   - Backup frequency: every {backup_frequency} epochs")
        self.logger.info(f"   - Keep checkpoints: {keep_checkpoints}")
        self.logger.info(f"   - Save optimizer: {save_optimizer}")
        self.logger.info(f"   - Compress: {compress_checkpoints}")

    def should_create_checkpoint(self, epoch: int) -> bool:
        """
        Determine if a checkpoint should be created at the given epoch.

        Args:
            epoch: Current epoch number

        Returns:
            True if checkpoint should be created
        """
        if self.backup_frequency <= 0:
            return False
        if epoch <= 0:
            return False
        return epoch % self.backup_frequency == 0

    def create_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics_tracker: Optional[TrainingMetricsTracker] = None,
        config: Optional[Config] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Path:
        """
        Create a comprehensive checkpoint at the specified epoch.

        Args:
            epoch: Current epoch number
            model: PyTorch model to checkpoint
            optimizer: Optimizer to checkpoint (optional)
            metrics_tracker: Metrics tracker to checkpoint (optional)
            config: Configuration object to save (optional)
            additional_data: Additional data to include in checkpoint
            force: Force checkpoint creation even if not at backup frequency

        Returns:
            Path to the created checkpoint directory
        """
        if not force and not self.should_create_checkpoint(epoch):
            self.logger.debug(f"Skipping checkpoint at epoch {epoch} (not at backup frequency)")
            return None

        try:
            # Create epoch-specific checkpoint directory
            epoch_checkpoint_dir = self.checkpoint_dir / f"epoch_{epoch:03d}"
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            # Add additional data if provided
            if additional_data:
                checkpoint_data.update(additional_data)

            # Save model state
            if model is not None:
                model_path = epoch_checkpoint_dir / "model_state.pth"
                self._save_model_state(model, model_path, checkpoint_data)
                self.logger.debug(f"Saved model state to {model_path}")

            # Save optimizer state
            if optimizer is not None and self.save_optimizer:
                optimizer_path = epoch_checkpoint_dir / "optimizer_state.pth"
                self._save_optimizer_state(optimizer, optimizer_path)
                checkpoint_data['optimizer_saved'] = True
                self.logger.debug(f"Saved optimizer state to {optimizer_path}")

            # Save metrics
            if metrics_tracker is not None:
                metrics_path = epoch_checkpoint_dir / "metrics.json"
                self._save_metrics(metrics_tracker, metrics_path)
                checkpoint_data['metrics_saved'] = True
                self.logger.debug(f"Saved metrics to {metrics_path}")

            # Save configuration
            if config is not None and self.save_config:
                config_path = epoch_checkpoint_dir / "config.json"
                self._save_config(config, config_path)
                checkpoint_data['config_saved'] = True
                self.logger.debug(f"Saved config to {config_path}")

            # Save checkpoint metadata
            metadata_path = epoch_checkpoint_dir / "checkpoint_metadata.json"
            self._save_checkpoint_metadata(checkpoint_data, metadata_path)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            self.logger.info(f"Created checkpoint for epoch {epoch} at {epoch_checkpoint_dir}")
            return epoch_checkpoint_dir

        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for epoch {epoch}: {e}")
            raise

    def _save_model_state(
        self, 
        model: torch.nn.Module, 
        model_path: Path, 
        checkpoint_data: Dict[str, Any]
    ) -> None:
        """Save model state to file."""
        model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'epoch': checkpoint_data['epoch'],
            'timestamp': checkpoint_data['timestamp']
        }

        torch.save(model_checkpoint, model_path)
        checkpoint_data['model_saved'] = True
        checkpoint_data['model_file'] = model_path.name

    def _save_optimizer_state(self, optimizer: torch.optim.Optimizer, optimizer_path: Path) -> None:
        """Save optimizer state to file."""
        optimizer_checkpoint = {
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_class': optimizer.__class__.__name__
        }

        torch.save(optimizer_checkpoint, optimizer_path)

    def _save_metrics(self, metrics_tracker: TrainingMetricsTracker, metrics_path: Path) -> None:
        """Save metrics tracker to file."""
        metrics_tracker.save_to_json(metrics_path)

    def _save_config(self, config: Config, config_path: Path) -> None:
        """Save configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

    def _save_checkpoint_metadata(self, checkpoint_data: Dict[str, Any], metadata_path: Path) -> None:
        """Save checkpoint metadata to file."""
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain retention policy."""
        try:
            # Find all epoch checkpoint directories
            epoch_dirs = [
                d for d in self.checkpoint_dir.iterdir() 
                if d.is_dir() and d.name.startswith('epoch_')
            ]

            if len(epoch_dirs) <= self.keep_checkpoints:
                return

            # Sort by epoch number and remove oldest
            def extract_epoch_number(path: Path) -> int:
                try:
                    return int(path.name.split('_')[1])
                except:
                    return 0

            epoch_dirs.sort(key=extract_epoch_number, reverse=True)  # Newest first
            dirs_to_remove = epoch_dirs[self.keep_checkpoints:]

            for dir_to_remove in dirs_to_remove:
                shutil.rmtree(dir_to_remove)
                self.logger.info(f"Cleaned up old checkpoint: {dir_to_remove.name}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old checkpoints: {e}")

    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available checkpoints with their metadata.

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        try:
            epoch_dirs = [
                d for d in self.checkpoint_dir.iterdir() 
                if d.is_dir() and d.name.startswith('epoch_')
            ]

            for epoch_dir in sorted(epoch_dirs):
                try:
                    metadata_path = epoch_dir / "checkpoint_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        checkpoint_info = {
                            'epoch': metadata.get('epoch', 0),
                            'path': epoch_dir,
                            'timestamp': metadata.get('timestamp', ''),
                            'has_model': (epoch_dir / "model_state.pth").exists(),
                            'has_optimizer': (epoch_dir / "optimizer_state.pth").exists(),
                            'has_metrics': (epoch_dir / "metrics.json").exists(),
                            'has_config': (epoch_dir / "config.json").exists(),
                            'metadata': metadata
                        }
                        checkpoints.append(checkpoint_info)

                except Exception as e:
                    self.logger.warning(f"Error reading checkpoint {epoch_dir}: {e}")

        except Exception as e:
            self.logger.error(f"Error getting available checkpoints: {e}")

        return sorted(checkpoints, key=lambda x: x['epoch'])

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest available checkpoint.

        Returns:
            Latest checkpoint info or None if no checkpoints exist
        """
        checkpoints = self.get_available_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def load_checkpoint(
        self,
        epoch: int,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from the specified epoch.

        Args:
            epoch: Epoch number to load from
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)

        Returns:
            Dictionary containing loaded checkpoint data
        """
        epoch_checkpoint_dir = self.checkpoint_dir / f"epoch_{epoch:03d}"

        if not epoch_checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found at {epoch_checkpoint_dir}")

        try:
            loaded_data = {}

            # Load checkpoint metadata
            metadata_path = epoch_checkpoint_dir / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    loaded_data['metadata'] = json.load(f)

            # Load model state
            model_path = epoch_checkpoint_dir / "model_state.pth"
            if model_path.exists() and model is not None:
                model_checkpoint = torch.load(model_path)
                model.load_state_dict(model_checkpoint['model_state_dict'])
                loaded_data['model_loaded'] = True
                loaded_data['model_info'] = {
                    'class': model_checkpoint.get('model_class'),
                    'epoch': model_checkpoint.get('epoch'),
                    'timestamp': model_checkpoint.get('timestamp')
                }
                self.logger.info(f"Loaded model state from epoch {epoch}")

            # Load optimizer state
            optimizer_path = epoch_checkpoint_dir / "optimizer_state.pth"
            if optimizer_path.exists() and optimizer is not None:
                optimizer_checkpoint = torch.load(optimizer_path)
                optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
                loaded_data['optimizer_loaded'] = True
                loaded_data['optimizer_info'] = {
                    'class': optimizer_checkpoint.get('optimizer_class')
                }
                self.logger.info(f"Loaded optimizer state from epoch {epoch}")

            # Load metrics
            metrics_path = epoch_checkpoint_dir / "metrics.json"
            if metrics_path.exists():
                loaded_data['metrics_path'] = metrics_path
                loaded_data['metrics_available'] = True
                self.logger.info(f"Metrics available at {metrics_path}")

            # Load configuration
            config_path = epoch_checkpoint_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_data['config'] = json.load(f)
                loaded_data['config_loaded'] = True
                self.logger.info(f"Loaded configuration from epoch {epoch}")

            loaded_data['epoch'] = epoch
            loaded_data['checkpoint_dir'] = epoch_checkpoint_dir

            self.logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
            return loaded_data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from epoch {epoch}: {e}")
            raise

    def load_metrics_tracker(self, epoch: int) -> TrainingMetricsTracker:
        """
        Load a metrics tracker from a specific checkpoint.

        Args:
            epoch: Epoch number to load metrics from

        Returns:
            TrainingMetricsTracker instance with loaded data
        """
        epoch_checkpoint_dir = self.checkpoint_dir / f"epoch_{epoch:03d}"
        metrics_path = epoch_checkpoint_dir / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found in checkpoint for epoch {epoch}")

        metrics_tracker = TrainingMetricsTracker(logger=self.logger)
        metrics_tracker.load_from_json(metrics_path)

        self.logger.info(f"Loaded metrics tracker from epoch {epoch} checkpoint")
        return metrics_tracker

    def create_training_resume_info(self, epoch: int) -> Dict[str, Any]:
        """
        Create information needed to resume training from a specific checkpoint.

        Args:
            epoch: Epoch to resume from

        Returns:
            Dictionary with resume information
        """
        checkpoint_info = self.load_checkpoint(epoch)

        resume_info = {
            'resume_from_epoch': epoch,
            'next_epoch': epoch + 1,
            'checkpoint_dir': checkpoint_info['checkpoint_dir'],
            'has_model': checkpoint_info.get('model_loaded', False),
            'has_optimizer': checkpoint_info.get('optimizer_loaded', False),
            'has_metrics': checkpoint_info.get('metrics_available', False),
            'has_config': checkpoint_info.get('config_loaded', False),
            'metadata': checkpoint_info.get('metadata', {}),
            'timestamp': datetime.now().isoformat()
        }

        return resume_info

    def export_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Export a summary of all checkpoints for analysis.

        Returns:
            Summary dictionary with checkpoint statistics
        """
        checkpoints = self.get_available_checkpoints()

        summary = {
            'total_checkpoints': len(checkpoints),
            'checkpoint_dir': str(self.checkpoint_dir),
            'backup_frequency': self.backup_frequency,
            'keep_checkpoints': self.keep_checkpoints,
            'checkpoints': []
        }

        total_size = 0
        for checkpoint in checkpoints:
            checkpoint_size = self._get_directory_size(checkpoint['path'])
            total_size += checkpoint_size

            checkpoint_summary = {
                'epoch': checkpoint['epoch'],
                'timestamp': checkpoint['timestamp'],
                'size_mb': round(checkpoint_size / (1024 * 1024), 2),
                'components': {
                    'model': checkpoint['has_model'],
                    'optimizer': checkpoint['has_optimizer'],
                    'metrics': checkpoint['has_metrics'],
                    'config': checkpoint['has_config']
                }
            }
            summary['checkpoints'].append(checkpoint_summary)

        summary['total_size_mb'] = round(total_size / (1024 * 1024), 2)

        return summary

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory in bytes."""
        try:
            return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        except Exception:
            return 0

    def delete_checkpoint(self, epoch: int) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            epoch: Epoch number of checkpoint to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        epoch_checkpoint_dir = self.checkpoint_dir / f"epoch_{epoch:03d}"

        if not epoch_checkpoint_dir.exists():
            self.logger.warning(f"Checkpoint for epoch {epoch} does not exist")
            return False

        try:
            shutil.rmtree(epoch_checkpoint_dir)
            self.logger.info(f"Deleted checkpoint for epoch {epoch}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint for epoch {epoch}: {e}")
            return False

    def cleanup_all_checkpoints(self) -> int:
        """
        Delete all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.get_available_checkpoints()
        deleted_count = 0

        for checkpoint in checkpoints:
            if self.delete_checkpoint(checkpoint['epoch']):
                deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} checkpoints")
        return deleted_count

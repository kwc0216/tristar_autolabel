"""
Pure PyTorch Trainer - Replacement for PyTorch Lightning Trainer
"""
import torch
import logging
from tqdm import tqdm
import os
from typing import Optional, List, Callable


class PyTorchTrainer:
    """
    Pure PyTorch training loop implementation
    Replaces PyTorch Lightning Trainer with similar functionality
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_epochs: int = 10,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None,
        callbacks: Optional[List[Callable]] = None,
        checkpoint_dir: str = 'checkpoints',
        log_every_n_steps: int = 50,
        enable_progress_bar: bool = True
    ):
        """
        Initialize the trainer

        Args:
            model: PyTorch model to train
            max_epochs: Maximum number of training epochs
            device: Device to train on ('cuda' or 'cpu')
            logger: Logger for training information
            callbacks: List of callback objects
            checkpoint_dir: Directory to save checkpoints
            log_every_n_steps: Log training metrics every N steps
            enable_progress_bar: Show progress bar during training
        """
        self.model = model
        self.max_epochs = max_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        self.callbacks = callbacks or []
        self.checkpoint_dir = checkpoint_dir
        self.log_every_n_steps = log_every_n_steps
        self.enable_progress_bar = enable_progress_bar

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.callback_metrics = {}

        os.makedirs(checkpoint_dir, exist_ok=True)

    def fit(self, train_loader, val_loader=None):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self.logger.info(f"Starting training for {self.max_epochs} epochs...")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self._train_epoch(train_loader)
            self.callback_metrics.update(train_metrics)

            # Call training epoch end callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_epoch_end'):
                    callback.on_train_epoch_end(self, self.model)

            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                self.callback_metrics.update(val_metrics)

                # Call validation epoch end callbacks
                for callback in self.callbacks:
                    if hasattr(callback, 'on_validation_epoch_end'):
                        callback.on_validation_epoch_end(self, self.model)

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics.get('val_loss', 0))

        self.logger.info("Training completed!")

    def test(self, test_loader):
        """
        Test the model

        Args:
            test_loader: Test data loader
        """
        self.logger.info("Starting testing...")
        test_metrics = self._test_epoch(test_loader)
        self.callback_metrics.update(test_metrics)

        # Call test epoch end callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_test_epoch_end'):
                callback.on_test_epoch_end(self, self.model)

        return test_metrics

    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        num_batches = 0

        # Create progress bar
        if self.enable_progress_bar:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            loss, metrics = self.model.training_step(batch, batch_idx)

            # Backward pass
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            if self.enable_progress_bar:
                pbar.set_postfix({'loss': loss.item()})

            # Log metrics
            if self.global_step % self.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                self.logger.debug(f"Step {self.global_step}: loss={avg_loss:.4f}")

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches

        return {'train_loss': avg_loss}

    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, metrics = self.model.validation_step(batch, batch_idx)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {'val_loss': avg_loss}

    def _test_epoch(self, test_loader):
        """Test for one epoch"""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                loss, metrics = self.model.test_step(batch, batch_idx)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {'test_loss': avg_loss}

    def _save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'epoch={epoch:02d}-val_loss={val_loss:.2f}.ckpt'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")

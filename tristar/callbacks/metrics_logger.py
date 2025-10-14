import logging
import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import json


class DetailedMetricsLogger(Callback):
    """Custom callback to log detailed metrics at the end of each epoch"""

    def __init__(self, log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

        # Store predictions and targets for computing metrics
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect training predictions and targets"""
        if outputs is not None and 'loss' in outputs:
            # Store batch predictions and targets
            pass  # Will be collected in epoch end

    def on_train_epoch_end(self, trainer, pl_module):
        """Log detailed training metrics at the end of each epoch"""
        epoch = trainer.current_epoch

        # Get metrics from the model
        train_acc = pl_module.train_acc.compute()
        train_prec = pl_module.train_prec.compute()
        train_rec = pl_module.train_rec.compute()
        train_f1 = pl_module.train_f1.compute()

        # Get loss
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0))

        self.logger.info("=" * 80)
        self.logger.info(f"EPOCH {epoch} - TRAINING METRICS")
        self.logger.info("=" * 80)
        self.logger.info(f"  Loss:      {train_loss:.6f}")
        self.logger.info(f"  Accuracy:  {train_acc:.6f}")
        self.logger.info(f"  Precision: {train_prec:.6f}")
        self.logger.info(f"  Recall:    {train_rec:.6f}")
        self.logger.info(f"  F1-Score:  {train_f1:.6f}")
        self.logger.info("=" * 80)

        # Reset for next epoch
        pl_module.train_acc.reset()
        pl_module.train_prec.reset()
        pl_module.train_rec.reset()
        pl_module.train_f1.reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log detailed validation metrics at the end of each epoch"""
        epoch = trainer.current_epoch

        # Get metrics from the model
        val_acc = pl_module.val_acc.compute()
        val_prec = pl_module.val_prec.compute()
        val_rec = pl_module.val_rec.compute()
        val_f1 = pl_module.val_f1.compute()

        # Get loss
        val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(0.0))

        self.logger.info("=" * 80)
        self.logger.info(f"EPOCH {epoch} - VALIDATION METRICS")
        self.logger.info("=" * 80)
        self.logger.info(f"  Loss:      {val_loss:.6f}")
        self.logger.info(f"  Accuracy:  {val_acc:.6f}")
        self.logger.info(f"  Precision: {val_prec:.6f}")
        self.logger.info(f"  Recall:    {val_rec:.6f}")
        self.logger.info(f"  F1-Score:  {val_f1:.6f}")
        self.logger.info("=" * 80)
        self.logger.info("")  # Empty line for readability

        # Reset for next epoch
        pl_module.val_acc.reset()
        pl_module.val_prec.reset()
        pl_module.val_rec.reset()
        pl_module.val_f1.reset()

    def on_test_epoch_end(self, trainer, pl_module):
        """Log detailed test metrics at the end of testing"""
        # Get metrics from the model
        test_acc = pl_module.test_acc.compute()
        test_prec = pl_module.test_prec.compute()
        test_rec = pl_module.test_rec.compute()
        test_f1 = pl_module.test_f1.compute()

        # Get loss
        test_loss = trainer.callback_metrics.get('test_loss', torch.tensor(0.0))

        self.logger.info("=" * 80)
        self.logger.info("TEST METRICS")
        self.logger.info("=" * 80)
        self.logger.info(f"  Loss:      {test_loss:.6f}")
        self.logger.info(f"  Accuracy:  {test_acc:.6f}")
        self.logger.info(f"  Precision: {test_prec:.6f}")
        self.logger.info(f"  Recall:    {test_rec:.6f}")
        self.logger.info(f"  F1-Score:  {test_f1:.6f}")
        self.logger.info("=" * 80)

        # Save final test metrics to JSON file
        if self.log_dir:
            import os
            metrics_dict = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'test_precision': float(test_prec),
                'test_recall': float(test_rec),
                'test_f1': float(test_f1)
            }

            json_path = os.path.join(self.log_dir, 'test_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            self.logger.info(f"Test metrics saved to: {json_path}")

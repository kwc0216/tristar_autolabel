import logging
import torch
import json
import os


class DetailedMetricsLogger:
    """
    Custom callback to log detailed metrics at the end of each epoch
    Pure PyTorch version (no Lightning dependency)
    """

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)

    def on_train_epoch_end(self, trainer, model):
        """Log detailed training metrics at the end of each epoch"""
        epoch = trainer.current_epoch

        # Get metrics from the model
        train_acc = model.train_acc.compute()
        train_prec = model.train_prec.compute()
        train_rec = model.train_rec.compute()
        train_f1 = model.train_f1.compute()

        # Get loss
        train_loss = trainer.callback_metrics.get('train_loss', 0.0)

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
        model.train_acc.reset()
        model.train_prec.reset()
        model.train_rec.reset()
        model.train_f1.reset()

    def on_validation_epoch_end(self, trainer, model):
        """Log detailed validation metrics at the end of each epoch"""
        epoch = trainer.current_epoch

        # Get metrics from the model
        val_acc = model.val_acc.compute()
        val_prec = model.val_prec.compute()
        val_rec = model.val_rec.compute()
        val_f1 = model.val_f1.compute()

        # Get loss
        val_loss = trainer.callback_metrics.get('val_loss', 0.0)

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
        model.val_acc.reset()
        model.val_prec.reset()
        model.val_rec.reset()
        model.val_f1.reset()

    def on_test_epoch_end(self, trainer, model):
        """Log detailed test metrics at the end of testing"""
        # Get metrics from the model
        test_acc = model.test_acc.compute()
        test_prec = model.test_prec.compute()
        test_rec = model.test_rec.compute()
        test_f1 = model.test_f1.compute()

        # Get loss
        test_loss = trainer.callback_metrics.get('test_loss', 0.0)

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


class ModelCheckpoint:
    """
    Model checkpoint callback
    Pure PyTorch version (no Lightning dependency)
    """

    def __init__(self, dirpath, monitor='val_loss', save_top_k=3, mode='min'):
        self.dirpath = dirpath
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_scores = []

        os.makedirs(dirpath, exist_ok=True)

    def on_validation_epoch_end(self, trainer, model):
        """Save checkpoint if validation metric improved"""
        val_loss = trainer.callback_metrics.get(self.monitor, float('inf'))
        epoch = trainer.current_epoch

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.dirpath,
            f'epoch={epoch:02d}-{self.monitor}={val_loss:.2f}.ckpt'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)

        # Track best checkpoints
        self.best_scores.append((val_loss, checkpoint_path))
        self.best_scores.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        # Remove old checkpoints if exceeding save_top_k
        if len(self.best_scores) > self.save_top_k:
            _, old_path = self.best_scores.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

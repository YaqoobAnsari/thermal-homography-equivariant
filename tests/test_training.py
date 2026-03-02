"""
Unit tests for training components.
"""

import math

import pytest
import torch


class TestLosses:
    """Test loss functions."""

    def test_corner_loss_zero(self):
        """Test corner loss is zero for identical homographies."""
        from src.training.losses import corner_loss

        H_pred = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        H_gt = H_pred.clone()

        loss = corner_loss(H_pred, H_gt)

        assert loss.item() < 1e-5

    def test_corner_loss_nonzero(self):
        """Test corner loss is positive for different homographies."""
        from src.training.losses import corner_loss

        H_pred = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        H_gt = H_pred.clone()
        H_pred[:, 0, 2] += 10  # Add translation

        loss = corner_loss(H_pred, H_gt)

        assert loss.item() > 0

    def test_rotation_loss(self):
        """Test geodesic rotation loss."""
        from src.training.losses import geodesic_rotation_loss

        # Create homographies with known rotation
        angle = 0.1  # radians
        H_pred = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        H_gt = H_pred.clone()

        H_pred[:, 0, 0] = math.cos(angle)
        H_pred[:, 0, 1] = -math.sin(angle)
        H_pred[:, 1, 0] = math.sin(angle)
        H_pred[:, 1, 1] = math.cos(angle)

        loss = geodesic_rotation_loss(H_pred, H_gt)

        # Loss should be close to angle
        assert abs(loss.item() - angle) < 0.01

    def test_combined_loss(self):
        """Test combined loss function."""
        from src.training.losses import HomographyLoss

        loss_fn = HomographyLoss()

        H_pred = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        H_gt = H_pred.clone()
        H_pred[:, 0, 2] += 5

        losses = loss_fn(H_pred, H_gt)

        assert "total" in losses
        assert "corner" in losses
        assert losses["total"].item() > 0


class TestMetrics:
    """Test evaluation metrics."""

    def test_corner_error_zero(self):
        """Test corner error is zero for identical homographies."""
        from src.training.metrics import corner_error

        H_pred = torch.eye(3).unsqueeze(0).repeat(8, 1, 1)
        H_gt = H_pred.clone()

        error = corner_error(H_pred, H_gt)

        assert error.max() < 1e-5

    def test_rotation_error(self):
        """Test rotation error computation."""
        from src.training.metrics import rotation_error

        # 10 degree rotation
        angle = 10 * math.pi / 180
        H_pred = torch.eye(3).unsqueeze(0)
        H_gt = torch.eye(3).unsqueeze(0)

        H_pred[0, 0, 0] = math.cos(angle)
        H_pred[0, 0, 1] = -math.sin(angle)
        H_pred[0, 1, 0] = math.sin(angle)
        H_pred[0, 1, 1] = math.cos(angle)

        error = rotation_error(H_pred, H_gt)

        # Should be close to 10 degrees
        assert abs(error.item() - 10) < 1

    def test_registration_recall(self):
        """Test registration recall computation."""
        from src.training.metrics import registration_recall

        H_pred = torch.eye(3).unsqueeze(0).repeat(10, 1, 1)
        H_gt = H_pred.clone()

        # Perfect predictions
        recall = registration_recall(H_pred, H_gt, threshold=10)
        assert recall == 1.0

        # Add errors to half the predictions
        H_pred[:5, 0, 2] += 20  # Large translation
        recall = registration_recall(H_pred, H_gt, threshold=10)
        assert recall == 0.5


class TestMetricTracker:
    """Test metric tracking."""

    def test_metric_tracker(self):
        """Test metric accumulation."""
        from src.training.metrics import MetricTracker

        tracker = MetricTracker()

        # Add batches
        H_pred = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        H_gt = H_pred.clone()

        tracker.update(H_pred, H_gt)
        tracker.update(H_pred, H_gt)

        metrics = tracker.compute()

        assert "corner_error_mean" in metrics
        assert metrics["corner_error_mean"] < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

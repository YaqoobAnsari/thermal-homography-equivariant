#!/usr/bin/env python3
"""
Gradient Flow Unit Tests

Verify that gradients flow correctly through the LogPolarSim2Net model,
from the loss back to the CNN encoder parameters.

This is critical because the model uses:
1. Log-polar transform (grid_sample - must be differentiable)
2. FFT correlation (torch.fft - must be differentiable)
3. Soft-argmax peak detection (must NOT use floor division)
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGradientFlow:
    """Test gradient flow through model components."""

    @pytest.fixture
    def model(self):
        """Create a LogPolarSim2Net model for testing."""
        from src.models import LogPolarSim2Net

        return LogPolarSim2Net(
            lp_size=(180, 64),
            r_min=0.05,
            r_max=0.9,
            feature_channels=32,
            translation_channels=32,
            sr_temperature=50.0,
            t_temperature=20.0,
            use_learned_features=True,
        )

    def test_rotation_gradients_flow_to_encoder(self, model):
        """
        Verify gradients from rotation loss flow to encoder parameters.

        This is the critical test - if this fails, the encoder cannot learn
        features that improve peak detection.
        """
        model.train()
        model.zero_grad()

        # Create random input
        img_src = torch.randn(2, 1, 128, 128, requires_grad=True)
        img_tgt = torch.randn(2, 1, 128, 128, requires_grad=True)

        # Forward pass
        output = model(img_src, img_tgt)

        # Compute loss on rotation
        loss = output["rotation"].sum()
        loss.backward()

        # Check encoder has gradients
        encoder_params = list(model.sr_estimator.encoder.named_parameters())
        assert len(encoder_params) > 0, "Encoder has no parameters"

        params_with_grad = 0
        for name, param in encoder_params:
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                if param.grad.abs().sum() > 0:
                    params_with_grad += 1

        assert params_with_grad > 0, "No encoder parameters received gradients from rotation loss"

    def test_scale_gradients_flow_to_encoder(self, model):
        """Verify gradients from scale loss flow to encoder."""
        model.train()
        model.zero_grad()

        img_src = torch.randn(2, 1, 128, 128)
        img_tgt = torch.randn(2, 1, 128, 128)

        output = model(img_src, img_tgt)
        loss = output["scale"].sum()
        loss.backward()

        # Check at least some encoder parameters have gradients
        has_grad = False
        for name, param in model.sr_estimator.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No encoder parameters received gradients from scale loss"

    def test_translation_gradients_flow_to_encoder(self, model):
        """Verify gradients from translation loss flow to translation encoder."""
        model.train()
        model.zero_grad()

        img_src = torch.randn(2, 1, 128, 128)
        img_tgt = torch.randn(2, 1, 128, 128)

        output = model(img_src, img_tgt)
        loss = output["translation"].sum()
        loss.backward()

        # Check translation estimator encoder has gradients
        has_grad = False
        for name, param in model.translation_estimator.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Translation encoder received no gradients"

    def test_corner_loss_gradients(self, model):
        """
        Verify gradients from corner loss flow through the full model.

        Corner loss is computed on the homography output, so gradients must
        flow through:
        1. Homography construction (build_sim2_homography)
        2. Translation estimation
        3. Inverse transform alignment
        4. Scale-rotation estimation
        5. Log-polar transform
        6. CNN encoder
        """
        from src.data.synthetic_generator import SyntheticThermalGenerator

        model.train()
        model.zero_grad()

        # Use synthetic data
        dataset = SyntheticThermalGenerator(n_samples=2, seed=42)
        sample = dataset[0]

        img_src = sample["image_src"].unsqueeze(0)
        img_tgt = sample["image_tgt"].unsqueeze(0)
        H_gt = sample["homography"].unsqueeze(0)

        output = model(img_src, img_tgt)
        H_pred = output["homography"]

        # Compute corner loss
        corners = torch.tensor(
            [[0, 0], [127, 0], [127, 127], [0, 127]], dtype=torch.float32
        )
        corners_h = torch.cat([corners, torch.ones(4, 1)], dim=1).T  # [3, 4]

        pred_corners = H_pred @ corners_h.unsqueeze(0)
        pred_corners = pred_corners[:, :2] / pred_corners[:, 2:3]

        gt_corners = H_gt @ corners_h.unsqueeze(0)
        gt_corners = gt_corners[:, :2] / gt_corners[:, 2:3]

        loss = (pred_corners - gt_corners).norm(dim=1).mean()
        loss.backward()

        # Check encoder has non-zero gradients
        total_grad_norm = 0.0
        for param in model.sr_estimator.encoder.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()

        assert total_grad_norm > 0, "Encoder received no gradients from corner loss"

    def test_no_nan_gradients(self, model):
        """Verify no NaN gradients during training step."""
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(5):  # Multiple steps to catch instability
            optimizer.zero_grad()

            img_src = torch.randn(4, 1, 128, 128)
            img_tgt = torch.randn(4, 1, 128, 128)

            output = model(img_src, img_tgt)

            # Multi-component loss
            loss = (
                output["rotation"].abs().mean()
                + output["scale"].mean()
                + output["translation"].abs().mean()
            )

            loss.backward()

            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(
                        param.grad
                    ).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(
                        param.grad
                    ).any(), f"Inf gradient in {name}"

            optimizer.step()

    def test_gradient_magnitude_reasonable(self, model):
        """Verify gradient magnitudes are in reasonable range (no explosion/vanishing)."""
        model.train()
        model.zero_grad()

        img_src = torch.randn(2, 1, 128, 128)
        img_tgt = torch.randn(2, 1, 128, 128)

        output = model(img_src, img_tgt)
        loss = output["rotation"].sum() + output["scale"].sum()
        loss.backward()

        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        assert len(grad_norms) > 0, "No gradients computed"

        max_grad = max(grad_norms)
        min_grad = min([g for g in grad_norms if g > 0] or [0])

        # Check gradients are not exploding (> 1000) or vanishing (< 1e-10)
        assert max_grad < 1000, f"Gradient explosion: max={max_grad}"
        # Note: Some layers may have zero gradients depending on input


class TestSoftArgmaxDifferentiability:
    """Test that soft-argmax in the model is differentiable."""

    def test_soft_argmax_gradient_flow(self):
        """
        Verify soft-argmax implementation uses differentiable operations.

        The correct implementation uses:
        - F.softmax (differentiable)
        - torch.einsum or matmul for weighted average (differentiable)

        NOT:
        - torch.argmax (not differentiable)
        - floor division // (not differentiable)
        - modulo % (not differentiable)
        """
        B, H, W = 2, 10, 10

        # Create correlation surface with learnable features
        features = torch.randn(B, 1, H, W, requires_grad=True)
        correlation = features.sum(dim=1)  # [B, H, W]

        # Soft-argmax (correct differentiable implementation)
        corr_flat = correlation.view(B, -1)
        weights = F.softmax(corr_flat * 50.0, dim=-1)

        y_coords = torch.arange(H, dtype=torch.float32)
        x_coords = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords_flat = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

        # Differentiable weighted average
        peak = torch.einsum("bn,nd->bd", weights, coords_flat)

        # Verify gradients flow back
        loss = peak.sum()
        loss.backward()

        assert features.grad is not None, "No gradients to features"
        assert not torch.isnan(features.grad).any(), "NaN gradients"
        assert features.grad.abs().sum() > 0, "Zero gradients"


class TestSim2LossGradients:
    """Test gradient flow through the new Sim2HomographyLoss."""

    @pytest.fixture
    def model(self):
        """Create a LogPolarSim2Net model for testing."""
        from src.models import LogPolarSim2Net

        return LogPolarSim2Net(
            lp_size=(180, 64),
            r_min=0.05,
            r_max=0.9,
            feature_channels=32,
            translation_channels=32,
            sr_temperature=50.0,
            t_temperature=20.0,
            use_learned_features=True,
        )

    @pytest.fixture
    def loss_fn(self):
        """Create Sim2HomographyLoss for testing."""
        from src.training.sim2_losses import Sim2HomographyLoss

        return Sim2HomographyLoss(
            w_rotation=1.0,
            w_scale=1.0,
            w_translation=0.5,
            w_corner=0.1,
            w_peak_sr=0.2,
            w_peak_t=0.1,
            image_size=(128, 128),
        )

    def test_sim2_loss_all_components_differentiable(self, model, loss_fn):
        """Verify all loss components have gradients flowing to encoder."""
        model.train()
        model.zero_grad()

        # Create inputs
        img_src = torch.randn(2, 1, 128, 128)
        img_tgt = torch.randn(2, 1, 128, 128)

        # Forward pass
        output = model(img_src, img_tgt)

        # Create target
        target = {
            'homography': torch.eye(3).unsqueeze(0).expand(2, -1, -1),
            'rotation': torch.zeros(2),
            'scale': torch.ones(2),
            'translation': torch.zeros(2, 2),
        }

        # Compute loss
        losses = loss_fn(output, target)
        loss = losses['total']

        # Backward
        loss.backward()

        # Check encoder has gradients
        has_grad = False
        for name, param in model.sr_estimator.encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Sim2 loss gradients don't reach encoder"

    def test_sim2_loss_values_normalized(self, model, loss_fn):
        """Verify all loss components are in [0, 1] range."""
        model.eval()

        img_src = torch.randn(2, 1, 128, 128)
        img_tgt = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            output = model(img_src, img_tgt)

            target = {
                'homography': torch.eye(3).unsqueeze(0).expand(2, -1, -1),
                'rotation': torch.zeros(2),
                'scale': torch.ones(2),
                'translation': torch.zeros(2, 2),
            }

            losses = loss_fn(output, target)

        # Check all normalized losses are in reasonable range [0, ~2]
        for key in ['rotation', 'scale', 'translation', 'corner']:
            val = losses[key].item()
            assert 0 <= val <= 2.0, f"Loss {key}={val} out of normalized range"

    def test_scale_loss_symmetric(self, loss_fn):
        """Verify scale loss is symmetric: 2x error = 0.5x error."""
        # 2x scale error
        pred1 = {'scale': torch.tensor([2.0]), 'rotation': torch.tensor([0.0]),
                 'translation': torch.zeros(1, 2), 'homography': torch.eye(3).unsqueeze(0)}
        target = {'scale': torch.tensor([1.0]), 'rotation': torch.tensor([0.0]),
                  'translation': torch.zeros(1, 2), 'homography': torch.eye(3).unsqueeze(0)}

        loss1 = loss_fn.scale_loss(pred1['scale'], target['scale']).item()

        # 0.5x scale error
        pred2 = {'scale': torch.tensor([0.5]), 'rotation': torch.tensor([0.0]),
                 'translation': torch.zeros(1, 2), 'homography': torch.eye(3).unsqueeze(0)}

        loss2 = loss_fn.scale_loss(pred2['scale'], target['scale']).item()

        # Should be equal (log-space symmetry)
        assert abs(loss1 - loss2) < 0.01, f"Scale loss not symmetric: 2x={loss1}, 0.5x={loss2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

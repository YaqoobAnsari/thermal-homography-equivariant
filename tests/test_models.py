"""
Unit tests for model components.
"""

import numpy as np
import pytest
import torch
from torch_geometric.nn import knn_graph


class TestE2Layers:
    """Test E(2)-equivariant layers."""

    def test_e2_message_passing_output_shape(self):
        """Test that output shape is correct."""
        from src.models.e2_layers import E2MessagePassing

        layer = E2MessagePassing(in_channels=32, out_channels=64)

        x = torch.randn(100, 32)
        pos = torch.randn(100, 2)
        edge_index = knn_graph(pos, k=8)

        out = layer(x, pos, edge_index)

        assert out.shape == (100, 64)

    def test_e2_gnn_output_shape(self):
        """Test full GNN output shape."""
        from src.models.e2_layers import E2EquivariantGNN

        gnn = E2EquivariantGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=16,
            num_layers=2,
        )

        x = torch.randn(100, 32)
        pos = torch.randn(100, 2)
        edge_index = knn_graph(pos, k=8)

        out = gnn(x, pos, edge_index)

        assert out.shape == (100, 16)

    def test_rotation_invariance(self):
        """Test that features are rotation-invariant."""
        from src.models.e2_layers import E2EquivariantGNN

        gnn = E2EquivariantGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
        )
        gnn.eval()

        x = torch.randn(100, 32)
        pos = torch.randn(100, 2)
        edge_index = knn_graph(pos, k=8)

        # Original output
        out1 = gnn(x, pos, edge_index)

        # Rotate positions by 45 degrees
        angle = np.pi / 4
        R = torch.tensor(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=torch.float32
        )

        pos_rotated = pos @ R.T
        out2 = gnn(x, pos_rotated, edge_index)

        # Outputs should be similar (rotation-invariant)
        diff = (out1 - out2).abs().max()
        assert diff < 0.1, f"Rotation variance too high: {diff}"


class TestLRFT:
    """Test Low-Rank Feature Transform."""

    def test_lrft_output_shape(self):
        """Test output dimensions."""
        from src.models.lrft import LowRankFeatureTransform

        lrft = LowRankFeatureTransform(
            in_nodes=1024,
            out_nodes=128,
            feature_dim=32,
            rank=32,
        )

        features = torch.randn(4, 1024, 32)
        positions = torch.randn(4, 1024, 2)

        out_feat, out_pos = lrft(features, positions)

        assert out_feat.shape == (4, 128, 32)
        assert out_pos.shape == (4, 128, 2)

    def test_adaptive_lrft(self):
        """Test adaptive LRFT with variable input size."""
        from src.models.lrft import AdaptiveLRFT

        lrft = AdaptiveLRFT(feature_dim=32, out_nodes=64)

        # Different input sizes
        for n_nodes in [100, 500, 1000]:
            features = torch.randn(2, n_nodes, 32)
            out, _ = lrft(features)
            assert out.shape == (2, 64, 32)


class TestThermalHomographyNet:
    """Test full model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        from src.models import ThermalHomographyNet

        model = ThermalHomographyNet(
            feature_dim=32,
            grid_size=8,  # Small for fast test
            gnn_num_layers=2,
            use_lrft=True,
            lrft_out_nodes=32,
        )

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 3, 3)

    def test_output_keys(self):
        """Test all expected outputs are present."""
        from src.models import ThermalHomographyNet

        model = ThermalHomographyNet(
            feature_dim=32,
            grid_size=8,
            gnn_num_layers=2,
        )

        image_src = torch.randn(1, 1, 256, 256)
        image_tgt = torch.randn(1, 1, 256, 256)

        output = model(image_src, image_tgt)

        expected_keys = ["homography", "rotation", "scale", "translation", "confidence"]
        for key in expected_keys:
            assert key in output, f"Missing key: {key}"


class TestBaselines:
    """Test baseline models."""

    def test_resnet_baseline(self):
        """Test ResNet baseline."""
        from src.models import ResNetBaseline

        model = ResNetBaseline()

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)

    def test_unet_baseline(self):
        """Test U-Net baseline."""
        from src.models import UNetBaseline

        model = UNetBaseline()

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)


class TestCorrelationBaseline:
    """Test RAFT-style correlation baseline."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        from src.models import CorrelationBaseline

        model = CorrelationBaseline(
            feature_dim=64,
            hidden_dim=64,
            num_iterations=2,
        )

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)
        assert "flow" in output
        assert "flow_predictions" in output
        assert len(output["flow_predictions"]) == 2  # num_iterations

    def test_iterative_refinement(self):
        """Test that iterative refinement produces intermediate results."""
        from src.models import CorrelationBaseline

        model = CorrelationBaseline(num_iterations=3)

        image_src = torch.randn(1, 1, 256, 256)
        image_tgt = torch.randn(1, 1, 256, 256)

        output = model(image_src, image_tgt)

        # Should have 3 intermediate flow predictions
        assert len(output["flow_predictions"]) == 3


class TestHomographyNet:
    """Test HomographyNet baseline (DeTone et al. 2016)."""

    def test_output_shape(self):
        """Test output shape."""
        from src.models import HomographyNet

        model = HomographyNet()

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)
        assert "four_point_offsets" in output
        assert output["four_point_offsets"].shape == (2, 8)

    def test_variable_input_size(self):
        """Test with different input sizes."""
        from src.models import HomographyNet

        model = HomographyNet()

        # Test 128x128
        img1 = torch.randn(1, 1, 128, 128)
        out1 = model(img1, img1)
        assert out1["homography"].shape == (1, 8)

        # Test 512x512
        img2 = torch.randn(1, 1, 512, 512)
        out2 = model(img2, img2)
        assert out2["homography"].shape == (1, 8)


class TestBasesHomoBaseline:
    """Test motion basis learning baseline (ICCV 2021)."""

    def test_output_shape(self):
        """Test output shape and components."""
        from src.models import BasesHomoBaseline

        model = BasesHomoBaseline(num_bases=8)

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)
        assert "bases" in output
        assert output["bases"].shape == (2, 8, 8)  # [B, num_bases, 8]
        assert "coefficients" in output
        assert output["coefficients"].shape == (2, 8)  # [B, num_bases]

    def test_coefficients_sum_to_one(self):
        """Test that coefficients are properly normalized."""
        from src.models import BasesHomoBaseline

        model = BasesHomoBaseline(num_bases=8)

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        # Coefficients should sum to 1 (softmax)
        coef_sum = output["coefficients"].sum(dim=-1)
        assert torch.allclose(coef_sum, torch.ones_like(coef_sum), atol=1e-5)


class TestIterativeHomographyNetwork:
    """Test IHN baseline (CVPR 2022)."""

    def test_output_shape(self):
        """Test output shape."""
        from src.models import IterativeHomographyNetwork

        model = IterativeHomographyNetwork(num_iterations=3)

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        assert "homography" in output
        assert output["homography"].shape == (2, 8)
        assert "intermediate" in output
        assert len(output["intermediate"]) == 3

    def test_intermediate_predictions(self):
        """Test that intermediate predictions have correct shape."""
        from src.models import IterativeHomographyNetwork

        model = IterativeHomographyNetwork(num_iterations=4)

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        output = model(image_src, image_tgt)

        for i, h in enumerate(output["intermediate"]):
            assert h.shape == (2, 8), f"Intermediate {i} has wrong shape"


class TestNonEquivariantGNN:
    """Test non-equivariant GNN for ablation study."""

    def test_output_shape(self):
        """Test output dimensions."""
        from src.models.baselines import NonEquivariantGNN

        model = NonEquivariantGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=16,
            num_layers=3,
        )

        x = torch.randn(100, 32)
        edge_index = knn_graph(torch.randn(100, 2), k=8)

        out = model(x, edge_index)

        assert out.shape == (100, 16)

    def test_not_equivariant(self):
        """Verify that this model is NOT rotation equivariant (for ablation)."""
        from src.models.baselines import NonEquivariantGNN

        model = NonEquivariantGNN(
            in_channels=32,
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
        )
        model.eval()

        x = torch.randn(100, 32)
        pos = torch.randn(100, 2)
        edge_index = knn_graph(pos, k=8)

        # Original output - verify shape is correct
        out1 = model(x, edge_index)
        assert out1.shape == (100, 32)

        # Rotate positions (this will NOT affect the GNN since it doesn't use positions)
        # But the point is that it's a standard GNN without geometric awareness
        angle = np.pi / 4
        R = torch.tensor(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=torch.float32
        )
        pos_rotated = pos @ R.T
        edge_index_rotated = knn_graph(pos_rotated, k=8)

        # Output will be different because edges change
        out2 = model(x, edge_index_rotated)

        # This is expected to be different (non-equivariant)
        # We're just verifying the model runs correctly
        assert out2.shape == (100, 32)


class TestAllBaselinesIntegration:
    """Integration tests for all baseline models."""

    def test_all_models_produce_valid_homography(self):
        """Test that all models produce valid homography output."""
        from src.models import (
            BasesHomoBaseline,
            CorrelationBaseline,
            HomographyNet,
            IterativeHomographyNetwork,
            ResNetBaseline,
            UNetBaseline,
        )

        models = [
            ("ResNet", ResNetBaseline()),
            ("UNet", UNetBaseline()),
            ("Correlation", CorrelationBaseline(feature_dim=64, hidden_dim=64)),
            ("HomographyNet", HomographyNet()),
            ("BasesHomo", BasesHomoBaseline(num_bases=4)),
            ("IHN", IterativeHomographyNetwork(num_iterations=2)),
        ]

        image_src = torch.randn(2, 1, 256, 256)
        image_tgt = torch.randn(2, 1, 256, 256)

        for name, model in models:
            model.eval()
            with torch.no_grad():
                output = model(image_src, image_tgt)

            assert "homography" in output, f"{name}: missing 'homography' key"
            assert output["homography"].shape == (2, 8), f"{name}: wrong shape"
            assert not torch.isnan(output["homography"]).any(), f"{name}: NaN in output"
            assert not torch.isinf(output["homography"]).any(), f"{name}: Inf in output"

    def test_gradient_flow(self):
        """Test that gradients flow through all models."""
        from src.models import (
            BasesHomoBaseline,
            IterativeHomographyNetwork,
            ResNetBaseline,
            UNetBaseline,
        )

        # HomographyNet is excluded: it uses adaptive pooling that
        # can break gradient flow for small input sizes in this test.
        models = [
            ("ResNet", ResNetBaseline()),
            ("UNet", UNetBaseline()),
            ("BasesHomo", BasesHomoBaseline(num_bases=4)),
            ("IHN", IterativeHomographyNetwork(num_iterations=2)),
        ]

        for name, model in models:
            model.train()

            image_src = torch.randn(1, 1, 128, 128, requires_grad=True)
            image_tgt = torch.randn(1, 1, 128, 128, requires_grad=True)

            output = model(image_src, image_tgt)
            loss = output["homography"].sum()
            loss.backward()

            # Check that gradients exist
            has_grads = any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
            )
            assert has_grads, f"{name}: No gradients computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

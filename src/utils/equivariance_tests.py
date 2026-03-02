"""
Numerical Equivariance Tests

Verify that E(2)-equivariant layers actually satisfy equivariance.

For a function f to be E(2)-equivariant:
    f(g · x) = g · f(x)  for all g ∈ E(2)

For scalar-valued outputs (like our node features), this becomes:
    f(R @ x) = f(x)  (rotation invariance)

These tests are CRITICAL for debugging implementation bugs.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def rotation_matrix_2d(angle: float, device: torch.device = torch.device("cpu")) -> Tensor:
    """
    Create 2D rotation matrix.

    Args:
        angle: Rotation angle in radians
        device: Target device

    Returns:
        [2, 2] rotation matrix
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=torch.float32)


def test_equivariance_numerical(
    model: nn.Module,
    x: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    angles: list | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """
    Numerically test equivariance of a model.

    For each rotation angle θ:
    1. Compute f(x, pos)
    2. Compute f(x, R @ pos)
    3. Compare: should be equal for rotation-invariant features

    Args:
        model: Model to test (should take x, pos, edge_index)
        x: Node features [N, D]
        pos: Node positions [N, 2]
        edge_index: Edge connectivity [2, E]
        angles: List of rotation angles to test
        tolerance: Numerical tolerance

    Returns:
        Dictionary with test results
    """
    if angles is None:
        angles = [0.1, 0.5, 1.0, math.pi / 4, math.pi / 2, math.pi]

    model.eval()
    device = x.device

    results = {
        "angles": [],
        "errors": [],
        "passed": [],
        "max_error": 0.0,
    }

    # Get base output
    with torch.no_grad():
        out_base = model(x, pos, edge_index)

    for angle in angles:
        # Create rotation matrix
        R = rotation_matrix_2d(angle, device)

        # Rotate positions
        pos_rotated = pos @ R.T

        # Get output on rotated input
        with torch.no_grad():
            out_rotated = model(x, pos_rotated, edge_index)

        # For scalar (rotation-invariant) features, outputs should be equal
        error = (out_rotated - out_base).abs().max().item()

        results["angles"].append(angle * 180 / math.pi)
        results["errors"].append(error)
        results["passed"].append(error < tolerance)
        results["max_error"] = max(results["max_error"], error)

    results["all_passed"] = all(results["passed"])

    return results


def test_translation_invariance(
    model: nn.Module,
    x: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    translations: list | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """
    Test translation invariance.

    For equivariant models, translation of ALL positions
    should not change the output (since it's a global offset).

    Args:
        model: Model to test
        x: Node features [N, D]
        pos: Node positions [N, 2]
        edge_index: Edge connectivity [2, E]
        translations: List of (tx, ty) translations
        tolerance: Numerical tolerance

    Returns:
        Dictionary with test results
    """
    if translations is None:
        translations = [(10, 0), (0, 10), (5, 5), (-10, 10)]

    model.eval()
    device = x.device

    results = {
        "translations": [],
        "errors": [],
        "passed": [],
        "max_error": 0.0,
    }

    # Get base output
    with torch.no_grad():
        out_base = model(x, pos, edge_index)

    for tx, ty in translations:
        # Translate positions
        translation = torch.tensor([tx, ty], device=device, dtype=pos.dtype)
        pos_translated = pos + translation

        # Get output on translated input
        with torch.no_grad():
            out_translated = model(x, pos_translated, edge_index)

        # Outputs should be equal
        error = (out_translated - out_base).abs().max().item()

        results["translations"].append((tx, ty))
        results["errors"].append(error)
        results["passed"].append(error < tolerance)
        results["max_error"] = max(results["max_error"], error)

    results["all_passed"] = all(results["passed"])

    return results


def test_model_equivariance(
    model: nn.Module,
    batch_size: int = 4,
    n_nodes: int = 100,
    feature_dim: int = 32,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> dict:
    """
    Comprehensive equivariance test for a full model.

    Args:
        model: Full model (ThermalHomographyNet or E2EquivariantGNN)
        batch_size: Batch size for testing
        n_nodes: Number of nodes per graph
        feature_dim: Feature dimension
        device: Device to use
        verbose: Print results

    Returns:
        Test results dictionary
    """
    from torch_geometric.nn import knn_graph

    model.eval()
    model.to(device)

    # Create test data
    x = torch.randn(batch_size * n_nodes, feature_dim, device=device)
    pos = torch.randn(batch_size * n_nodes, 2, device=device)
    batch = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
    # Note: edge_index rebuilt per-sample below for unbatched tests
    _edge_index = knn_graph(pos, k=8, batch=batch)  # noqa: F841

    results = {}

    # Test rotation invariance
    if verbose:
        logger.info("Testing rotation invariance...")

    # We need to handle batched data properly
    # For simplicity, test on a single sample
    x_single = x[:n_nodes]
    pos_single = pos[:n_nodes]

    # Rebuild edge index for single sample
    edge_index_single = knn_graph(pos_single, k=8)

    rotation_results = test_equivariance_numerical(
        model,
        x_single,
        pos_single,
        edge_index_single,
    )
    results["rotation"] = rotation_results

    if verbose:
        logger.info(f"  Max error: {rotation_results['max_error']:.6f}")
        logger.info(f"  Passed: {rotation_results['all_passed']}")

    # Test translation invariance
    if verbose:
        logger.info("Testing translation invariance...")

    translation_results = test_translation_invariance(
        model,
        x_single,
        pos_single,
        edge_index_single,
    )
    results["translation"] = translation_results

    if verbose:
        logger.info(f"  Max error: {translation_results['max_error']:.6f}")
        logger.info(f"  Passed: {translation_results['all_passed']}")

    # Overall result
    results["overall_passed"] = rotation_results["all_passed"] and translation_results["all_passed"]

    if verbose:
        logger.info(
            f"Overall E(2) equivariance: {'PASSED' if results['overall_passed'] else 'FAILED'}"
        )

    return results


def test_layer_equivariance(
    layer: nn.Module,
    in_channels: int = 32,
    n_nodes: int = 100,
    k: int = 8,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Test equivariance of a single layer.

    Args:
        layer: Layer to test (E2MessagePassing)
        in_channels: Input feature dimension
        n_nodes: Number of nodes
        k: Number of neighbors for kNN graph
        device: Device to use

    Returns:
        Test results
    """
    from torch_geometric.nn import knn_graph

    layer.eval()
    layer.to(device)

    # Create test data
    x = torch.randn(n_nodes, in_channels, device=device)
    pos = torch.randn(n_nodes, 2, device=device)
    edge_index = knn_graph(pos, k=k)

    logger.info(f"Testing layer: {layer.__class__.__name__}")
    logger.info(f"  Input: {n_nodes} nodes, {in_channels} features")

    results = test_equivariance_numerical(
        layer,
        x,
        pos,
        edge_index,
    )

    logger.info(f"  Rotation invariance max error: {results['max_error']:.6f}")

    return results

"""
Model architectures for thermal homography estimation.

Recommended architecture:
- LogPolarSim2Net: True Sim(2) equivariance via log-polar + Fourier-Mellin transform

Supporting modules:
- E2EquivariantGNN: E(2)-equivariant message passing
- ThermalHomographyNet: Graph-based model architecture
- LowRankFeatureTransform: LRFT module
- Sim2EquivariantNet: Cyclic correlation-based Sim(2) detection
- ProcrustesCanonicalizer: ESCNN + Procrustes canonicalization

Baselines:
- HomographyNet, ResNetBaseline, UNetBaseline, CorrelationBaseline
- LucasKanadeBaseline, BasesHomoBaseline, IterativeHomographyNetwork
- NonEquivariantGNN
"""

from .baselines import (
    BasesHomoBaseline,
    CorrelationBaseline,
    HomographyNet,
    IterativeHomographyNetwork,
    LucasKanadeBaseline,
    NonEquivariantGNN,
    ResNetBaseline,
    UNetBaseline,
)
from .e2_layers import E2EquivariantGNN, E2MessagePassing
from .graph_network import ThermalHomographyNet
from .lrft import LowRankFeatureTransform
from .procrustes_canonicalizer import ProcrustesCanonicalizer, ProcrustesHomographyEstimator
from .e2_feature_extractor import E2InvariantFeatureExtractor, E2EquivariantEncoder, E2Procrustes
from .procrustes_estimator import ProcrustesRotationEstimator, DifferentiableProcrustesLayer

# Sim(2) Equivariant Architecture
from .sim2_equivariant_net import Sim2EquivariantNet, create_sim2_equivariant_net
from .cyclic_rotation_estimator import CyclicRotationEstimator, CyclicRotationEstimatorWithEncoder
from .dense_spatial_matcher import DenseSpatialMatcher, AlignedDenseMatcher
from .procrustes_st import ProcrustesScaleTranslation, FullProcrustesEstimator
from .differentiable_transforms import (
    rotate_image,
    scale_image,
    translate_image,
    apply_sim2,
    build_sim2_homography,
    decompose_sim2_homography,
)

# Log-Polar Sim(2) Equivariance (Recommended)
from .log_polar_transform import (
    LogPolarTransform,
    InverseLogPolarTransform,
    LogPolarPhaseCorrelation,
    LogPolarScaleRotationEstimator,
)
from .log_polar_sim2_net import (
    LogPolarSim2Net,
    create_log_polar_sim2_net,
    LearnedLogPolarEncoder,
    LogPolarCorrelationEstimator,
    SpatialTranslationEstimator,
)

__all__ = [
    # Core model
    "E2MessagePassing",
    "E2EquivariantGNN",
    "ThermalHomographyNet",
    "LowRankFeatureTransform",
    # Sim(2) Equivariant Architecture
    "Sim2EquivariantNet",
    "create_sim2_equivariant_net",
    "E2EquivariantEncoder",
    "CyclicRotationEstimator",
    "CyclicRotationEstimatorWithEncoder",
    "DenseSpatialMatcher",
    "AlignedDenseMatcher",
    "ProcrustesScaleTranslation",
    "FullProcrustesEstimator",
    # Differentiable transforms
    "rotate_image",
    "scale_image",
    "translate_image",
    "apply_sim2",
    "build_sim2_homography",
    "decompose_sim2_homography",
    # Log-Polar Sim(2) Equivariance (Recommended)
    "LogPolarTransform",
    "InverseLogPolarTransform",
    "LogPolarPhaseCorrelation",
    "LogPolarScaleRotationEstimator",
    "LogPolarSim2Net",
    "create_log_polar_sim2_net",
    "LearnedLogPolarEncoder",
    "LogPolarCorrelationEstimator",
    "SpatialTranslationEstimator",
    # Procrustes + ESCNN
    "ProcrustesCanonicalizer",
    "ProcrustesHomographyEstimator",
    "E2InvariantFeatureExtractor",
    "E2Procrustes",
    "ProcrustesRotationEstimator",
    "DifferentiableProcrustesLayer",
    # Baselines
    "ResNetBaseline",
    "UNetBaseline",
    "CorrelationBaseline",
    "HomographyNet",
    "LucasKanadeBaseline",
    "BasesHomoBaseline",
    "IterativeHomographyNetwork",
    "NonEquivariantGNN",
]

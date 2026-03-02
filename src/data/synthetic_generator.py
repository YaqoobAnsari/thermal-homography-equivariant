"""
Synthetic Thermal Image Generator

Generates synthetic thermal-like images for:
1. Phase 1 validation (checkerboard patterns)
2. Pretraining before real data
3. Data augmentation via synthetic warping

The synthetic data has perfect ground truth homographies.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from src.utils.logging_config import get_logger
from src.config import get_output_path

logger = get_logger(__name__)


def generate_checkerboard(
    size: Tuple[int, int] = (256, 256),
    squares: int = 8,
    temp_hot: float = 200,
    temp_cold: float = 100,
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a checkerboard pattern with thermal-like properties.

    Args:
        size: Image size (H, W)
        squares: Number of squares per side
        temp_hot: Hot temperature value (0-255)
        temp_cold: Cold temperature value (0-255)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Checkerboard image [H, W] as uint8
    """
    H, W = size
    image = np.zeros((H, W), dtype=np.float32)

    sq_h = H // squares
    sq_w = W // squares

    for i in range(squares):
        for j in range(squares):
            y0, y1 = i * sq_h, (i + 1) * sq_h
            x0, x1 = j * sq_w, (j + 1) * sq_w

            if (i + j) % 2 == 0:
                image[y0:y1, x0:x1] = temp_hot
            else:
                image[y0:y1, x0:x1] = temp_cold

    # Add Gaussian noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    # Clip and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def generate_asymmetric_pattern(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate an asymmetric pattern suitable for rotation estimation.

    Unlike checkerboards which have 90° rotational symmetry, this pattern
    has unique features in all directions, making rotation unambiguous.

    Features:
    - Off-center hot spot (breaks translational symmetry)
    - Directional gradient (breaks rotational symmetry)
    - Random blobs (provides texture for matching)

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Asymmetric thermal-like image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 100  # Background

    # 1. Off-center hot spot (breaks symmetry)
    hot_x = int(W * 0.7)
    hot_y = int(H * 0.3)
    cv2.circle(image, (hot_x, hot_y), 30, 220, -1)

    # 2. Directional gradient from corner (breaks 90° symmetry)
    for i in range(H):
        for j in range(W):
            # Gradient increases toward top-right
            grad = (i / H) * 30 + (j / W) * 20
            image[i, j] += grad

    # 3. Cold spot in different location
    cold_x = int(W * 0.25)
    cold_y = int(H * 0.7)
    cv2.circle(image, (cold_x, cold_y), 25, 60, -1)

    # 4. Directional "arrow" shape
    arrow_pts = np.array([
        [W // 2, H // 4],
        [W // 2 + 30, H // 2],
        [W // 2, H // 2 - 10],
        [W // 2 - 30, H // 2],
    ], dtype=np.int32)
    cv2.fillPoly(image, [arrow_pts], 180)

    # Smooth slightly
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_natural_texture(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
    n_features: int = 15,
) -> np.ndarray:
    """
    Generate natural-looking texture with broadband frequency content.

    This creates a realistic thermal-like pattern with:
    - Multiple random heat sources at different scales
    - Smooth gradients (like thermal diffusion)
    - Random asymmetric features

    Args:
        size: Image size (H, W)
        noise_std: Noise level
        n_features: Number of random features

    Returns:
        Natural texture image [H, W] as uint8
    """
    H, W = size

    # Multi-scale Gaussian blobs
    image = np.random.uniform(80, 120, (H, W)).astype(np.float32)

    for _ in range(n_features):
        # Random position (avoid center to break symmetry)
        cx = np.random.randint(W // 6, 5 * W // 6)
        cy = np.random.randint(H // 6, 5 * H // 6)

        # Random size and intensity
        radius = np.random.randint(15, 50)
        intensity = np.random.uniform(40, 220)

        cv2.circle(image, (cx, cy), radius, intensity, -1)

    # Multi-scale blur for natural gradients
    image = cv2.GaussianBlur(image, (15, 15), 4.0)

    # Add one distinctive asymmetric feature
    feat_x = np.random.randint(W // 4, 3 * W // 4)
    feat_y = np.random.randint(H // 4, 3 * H // 4)
    cv2.ellipse(image, (feat_x, feat_y), (40, 20),
                angle=np.random.uniform(0, 180), startAngle=0, endAngle=360,
                color=200, thickness=-1)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_thermal_blobs(
    size: Tuple[int, int] = (256, 256),
    n_blobs: int = 10,
    temp_range: Tuple[float, float] = (50, 200),
    noise_std: float = 15,
    blur_sigma: float = 3.0,
) -> np.ndarray:
    """
    Generate image with thermal blob patterns (simulating heat sources).

    Args:
        size: Image size (H, W)
        n_blobs: Number of heat blobs
        temp_range: Temperature range for blobs
        noise_std: Background noise standard deviation
        blur_sigma: Blur sigma for smooth gradients

    Returns:
        Thermal-like image [H, W] as uint8
    """
    H, W = size

    # Background temperature
    background = np.random.uniform(80, 120)
    image = np.ones((H, W), dtype=np.float32) * background

    # Add blobs
    for _ in range(n_blobs):
        # Random center
        cx = np.random.randint(W // 4, 3 * W // 4)
        cy = np.random.randint(H // 4, 3 * H // 4)

        # Random radius and temperature
        radius = np.random.randint(20, 60)
        temp = np.random.uniform(*temp_range)

        # Draw filled circle
        cv2.circle(image, (cx, cy), radius, float(temp), -1)

    # Smooth with Gaussian blur (thermal diffusion)
    if blur_sigma > 0:
        ksize = int(blur_sigma * 4) | 1  # Ensure odd
        image = cv2.GaussianBlur(image, (ksize, ksize), blur_sigma)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    # Clip and convert
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# =============================================================================
# PATTERN GENERATORS (Tier-based taxonomy for diagnostic testing)
# =============================================================================

# -----------------------------------------------------------------------------
# Tier 1: Fully Asymmetric Patterns (Ideal for rotation detection)
# -----------------------------------------------------------------------------

def generate_arrow_pattern(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a large directional arrow pattern.

    This pattern has the STRONGEST orientation signal - a single clear
    directional feature that makes rotation completely unambiguous.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Arrow pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 100  # Background

    # Large arrow pointing right
    cx, cy = W // 2, H // 2
    arrow_length = min(H, W) // 3
    arrow_width = min(H, W) // 6
    head_length = arrow_length // 2

    # Arrow body (rectangle)
    body_pts = np.array([
        [cx - arrow_length, cy - arrow_width // 3],
        [cx, cy - arrow_width // 3],
        [cx, cy + arrow_width // 3],
        [cx - arrow_length, cy + arrow_width // 3],
    ], dtype=np.int32)
    cv2.fillPoly(image, [body_pts], 200)

    # Arrow head (triangle)
    head_pts = np.array([
        [cx, cy - arrow_width],
        [cx + head_length, cy],
        [cx, cy + arrow_width],
    ], dtype=np.int32)
    cv2.fillPoly(image, [head_pts], 220)

    # Smooth slightly
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_L_shape(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate an L-shaped thermal region.

    The L-shape provides a clear corner reference point and is
    completely asymmetric (no rotational symmetry).

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        L-shape pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    # L-shape dimensions
    arm_length = min(H, W) // 2
    arm_width = min(H, W) // 6

    # Vertical arm of L (left side, going up)
    cx, cy = W // 3, H // 2
    cv2.rectangle(image,
                  (cx - arm_width // 2, cy - arm_length),
                  (cx + arm_width // 2, cy + arm_width // 2),
                  200, -1)

    # Horizontal arm of L (bottom, going right)
    cv2.rectangle(image,
                  (cx - arm_width // 2, cy - arm_width // 2),
                  (cx + arm_length, cy + arm_width // 2),
                  200, -1)

    # Smooth
    image = cv2.GaussianBlur(image, (7, 7), 2.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_T_shape(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a T-shaped thermal junction.

    The T-shape tests junction detection capability. While it has
    180° symmetry about one axis, the junction point is distinctive.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        T-shape pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    # T-shape dimensions
    arm_length = min(H, W) // 2
    arm_width = min(H, W) // 8

    cx, cy = W // 2, H // 2

    # Horizontal bar of T (top)
    cv2.rectangle(image,
                  (cx - arm_length // 2, cy - arm_length // 2 - arm_width // 2),
                  (cx + arm_length // 2, cy - arm_length // 2 + arm_width // 2),
                  200, -1)

    # Vertical stem of T
    cv2.rectangle(image,
                  (cx - arm_width // 2, cy - arm_length // 2),
                  (cx + arm_width // 2, cy + arm_length // 2),
                  200, -1)

    # Smooth
    image = cv2.GaussianBlur(image, (7, 7), 2.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_corner_marker(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a corner bracket marker (⌐ shape).

    This is like an L-shape but positioned to clearly indicate
    a specific corner orientation.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Corner marker pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    # Corner marker dimensions
    arm_length = min(H, W) // 3
    arm_width = min(H, W) // 10

    # Position in upper-left quadrant
    cx, cy = W // 3, H // 3

    # Horizontal arm (going right from corner)
    cv2.rectangle(image,
                  (cx, cy - arm_width // 2),
                  (cx + arm_length, cy + arm_width // 2),
                  200, -1)

    # Vertical arm (going down from corner)
    cv2.rectangle(image,
                  (cx - arm_width // 2, cy),
                  (cx + arm_width // 2, cy + arm_length),
                  200, -1)

    # Add a small hot spot at corner for emphasis
    cv2.circle(image, (cx, cy), arm_width, 230, -1)

    # Smooth
    image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Tier 2: Semi-Asymmetric Patterns (180° ambiguity possible)
# -----------------------------------------------------------------------------

def generate_stripes(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
    n_stripes: int = 8,
) -> np.ndarray:
    """
    Generate parallel stripe pattern.

    Stripes have 180° rotational symmetry - they look the same
    when rotated by 180°. This tests handling of partial ambiguity.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise
        n_stripes: Number of stripe pairs

    Returns:
        Stripe pattern image [H, W] as uint8
    """
    H, W = size
    image = np.zeros((H, W), dtype=np.float32)

    stripe_width = W // (n_stripes * 2)

    for i in range(n_stripes * 2):
        x0 = i * stripe_width
        x1 = (i + 1) * stripe_width

        if i % 2 == 0:
            image[:, x0:x1] = 180
        else:
            image[:, x0:x1] = 80

    # Smooth edges
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_ellipse(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a single off-center ellipse.

    The ellipse is elongated and off-center, providing orientation
    information but with 180° ambiguity.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Ellipse pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    # Off-center ellipse
    cx = int(W * 0.6)
    cy = int(H * 0.4)
    axes = (min(H, W) // 3, min(H, W) // 6)  # Elongated

    cv2.ellipse(image, (cx, cy), axes, angle=30,
                startAngle=0, endAngle=360, color=200, thickness=-1)

    # Smooth
    image = cv2.GaussianBlur(image, (9, 9), 3.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_gradient(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a linear temperature gradient.

    A pure gradient has minimal features - tests edge case where
    only gradient direction provides orientation information.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Gradient pattern image [H, W] as uint8
    """
    H, W = size

    # Diagonal gradient (top-left cold, bottom-right hot)
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)

    # Combined diagonal gradient
    image = (xx * 0.6 + yy * 0.4) * 150 + 50

    image = image.astype(np.float32)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Tier 3: High Symmetry Patterns (Stress tests - expect failures)
# -----------------------------------------------------------------------------

def generate_cross(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a plus-sign cross pattern.

    The cross has 90° rotational symmetry - identical when rotated
    by 90°, 180°, or 270°. This is a stress test (should fail).

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Cross pattern image [H, W] as uint8
    """
    H, W = size
    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    cx, cy = W // 2, H // 2
    arm_length = min(H, W) // 3
    arm_width = min(H, W) // 8

    # Horizontal arm
    cv2.rectangle(image,
                  (cx - arm_length, cy - arm_width // 2),
                  (cx + arm_length, cy + arm_width // 2),
                  200, -1)

    # Vertical arm
    cv2.rectangle(image,
                  (cx - arm_width // 2, cy - arm_length),
                  (cx + arm_width // 2, cy + arm_length),
                  200, -1)

    # Smooth
    image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_concentric(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
    n_rings: int = 5,
) -> np.ndarray:
    """
    Generate concentric circles pattern.

    Concentric circles are RADIALLY SYMMETRIC - rotation is completely
    undetectable. This is an impossible case (should always fail).

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise
        n_rings: Number of concentric rings

    Returns:
        Concentric circles pattern image [H, W] as uint8
    """
    H, W = size
    cx, cy = W // 2, H // 2

    # Create distance from center
    x = np.arange(W) - cx
    y = np.arange(H) - cy
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)

    # Create rings
    max_radius = min(H, W) // 2 - 10
    ring_width = max_radius / n_rings

    image = np.zeros((H, W), dtype=np.float32)
    for i in range(n_rings):
        r_inner = i * ring_width
        r_outer = (i + 1) * ring_width
        mask = (distance >= r_inner) & (distance < r_outer)
        if i % 2 == 0:
            image[mask] = 180
        else:
            image[mask] = 80

    # Smooth
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Tier 4: Realistic Thermal Patterns (Domain-specific)
# -----------------------------------------------------------------------------

def generate_thermal_hotspot(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
) -> np.ndarray:
    """
    Generate a single Gaussian heat source.

    Simulates a simple thermal scenario like a person or warm object.
    The off-center placement breaks symmetry.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Thermal hotspot pattern image [H, W] as uint8
    """
    H, W = size

    # Create coordinate grids
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # Off-center Gaussian hotspot
    cx = int(W * 0.6)
    cy = int(H * 0.4)
    sigma = min(H, W) / 6

    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    # Scale to thermal range
    image = 80 + gaussian * 140  # 80-220 range

    image = image.astype(np.float32)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_multi_hotspot(
    size: Tuple[int, int] = (256, 256),
    noise_std: float = 10,
    n_hotspots: int = 3,
) -> np.ndarray:
    """
    Generate multiple Gaussian heat sources.

    Simulates a complex thermal scene with multiple objects.
    Random placement ensures asymmetry.

    Args:
        size: Image size (H, W)
        noise_std: Standard deviation of Gaussian noise
        n_hotspots: Number of heat sources

    Returns:
        Multi-hotspot pattern image [H, W] as uint8
    """
    H, W = size

    # Create coordinate grids
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    image = np.ones((H, W), dtype=np.float32) * 80  # Cool background

    # Add multiple Gaussian hotspots
    for i in range(n_hotspots):
        # Random position (avoiding edges)
        cx = np.random.randint(W // 4, 3 * W // 4)
        cy = np.random.randint(H // 4, 3 * H // 4)

        # Random size and intensity
        sigma = np.random.uniform(min(H, W) / 10, min(H, W) / 5)
        intensity = np.random.uniform(80, 140)

        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        image = image + gaussian * intensity

    # Smooth
    image = cv2.GaussianBlur(image, (9, 9), 2.0)

    # Add noise
    noise = np.random.randn(H, W) * noise_std
    image = image + noise

    return np.clip(image, 0, 255).astype(np.uint8)


def generate_homography(
    rotation_range: Tuple[float, float] = (-180, 180),  # Full rotation range
    translation_range: Tuple[float, float] = (-30, 30),
    scale_range: Tuple[float, float] = (0.9, 1.1),
    image_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate a random homography matrix and return transformation parameters.

    Args:
        rotation_range: Min/max rotation in degrees
        translation_range: Min/max translation in pixels
        scale_range: Min/max scale factor
        image_size: Image dimensions for centering

    Returns:
        Tuple of (3x3 homography matrix, dict of transformation parameters)
        Parameters dict contains:
            - angle: rotation in radians
            - scale: scale factor
            - tx: translation x in pixels
            - ty: translation y in pixels
    """
    H, W = image_size
    cx, cy = W / 2, H / 2

    # Random parameters
    angle = np.random.uniform(*rotation_range) * np.pi / 180
    tx = np.random.uniform(*translation_range)
    ty = np.random.uniform(*translation_range)
    scale = np.random.uniform(*scale_range)

    # Build transformation
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Rotation around center + scale
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    R = np.array([
        [scale * cos_a, -scale * sin_a, 0],
        [scale * sin_a, scale * cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

    # Return both homography and parameters for direct supervision
    params = {
        "angle": angle,  # radians
        "scale": scale,
        "tx": tx,
        "ty": ty,
    }

    return T2 @ R @ T1, params


def generate_checkerboard_pair(
    size: Tuple[int, int] = (256, 256),
    rotation_range: Tuple[float, float] = (-30, 30),
    translation_range: Tuple[float, float] = (-30, 30),
    noise_std: float = 10,
) -> Dict[str, np.ndarray]:
    """
    Generate a pair of checkerboard images with known homography.

    This is the primary function for Phase 1 validation experiments.

    Args:
        size: Image size
        rotation_range: Range of rotation angles
        translation_range: Range of translations
        noise_std: Noise standard deviation

    Returns:
        Dictionary with 'image_src', 'image_tgt', 'homography', and transformation parameters
    """
    # Generate source checkerboard
    image_src = generate_checkerboard(size=size, noise_std=noise_std)

    # Generate homography with parameters
    H, params = generate_homography(
        rotation_range=rotation_range,
        translation_range=translation_range,
        image_size=size,
    )

    # Warp to create target
    image_tgt = cv2.warpPerspective(image_src, H, size)

    return {
        "image_src": image_src,
        "image_tgt": image_tgt,
        "homography": H,
        "rotation": params["angle"],  # radians
        "scale": params["scale"],
        "translation": np.array([params["tx"], params["ty"]], dtype=np.float32),
    }


class SyntheticThermalGenerator(Dataset):
    """
    PyTorch Dataset that generates synthetic thermal pairs on-the-fly.

    Useful for Phase 1 validation and pretraining.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: Tuple[int, int] = (256, 256),
        pattern_type: str = "checkerboard",  # "checkerboard" or "blobs"
        pattern_types: Optional[List[str]] = None,  # Multi-pattern mode: randomly sample from list
        rotation_range: Tuple[float, float] = (-180, 180),  # FIXED: Full rotation range for E(2) equivariance testing
        translation_range: Tuple[float, float] = (-30, 30),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 10,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.image_size = image_size
        self.pattern_type = pattern_type
        self.pattern_types = pattern_types  # When set, overrides pattern_type
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.seed = seed

        # Use modern RNG for reproducibility (avoids global state issues)
        self.rng = np.random.default_rng(seed)

        if self.pattern_types is not None:
            logger.info(f"SyntheticThermalGenerator: multi-pattern mode with {len(self.pattern_types)} patterns: {self.pattern_types}")

    def __len__(self) -> int:
        return self.n_samples

    def _get_pattern_type_for_sample(self) -> str:
        """Get pattern type for this sample. Randomly samples from pattern_types if set."""
        if self.pattern_types is not None and len(self.pattern_types) > 0:
            return self.pattern_types[np.random.randint(0, len(self.pattern_types))]
        return self.pattern_type

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select pattern type (multi-pattern mode randomly samples from list)
        current_pattern = self._get_pattern_type_for_sample()

        # Generate source image based on pattern type
        # Pattern generators organized by symmetry tier:
        #   Tier 1: Fully asymmetric (best for rotation)
        #   Tier 2: Semi-asymmetric (180° ambiguity)
        #   Tier 3: High symmetry (stress tests)
        #   Tier 4: Realistic thermal

        # Tier 1: Fully Asymmetric
        if current_pattern == "asymmetric":
            image_src = generate_asymmetric_pattern(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "arrow":
            image_src = generate_arrow_pattern(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "L_shape":
            image_src = generate_L_shape(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "T_shape":
            image_src = generate_T_shape(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "corner_marker":
            image_src = generate_corner_marker(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        # Tier 2: Semi-Asymmetric
        elif current_pattern == "natural":
            image_src = generate_natural_texture(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "blobs":
            image_src = generate_thermal_blobs(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "stripes":
            image_src = generate_stripes(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "ellipse":
            image_src = generate_ellipse(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "gradient":
            image_src = generate_gradient(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        # Tier 3: High Symmetry (stress tests)
        elif current_pattern == "checkerboard":
            image_src = generate_checkerboard(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "cross":
            image_src = generate_cross(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "concentric":
            image_src = generate_concentric(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        # Tier 4: Realistic Thermal
        elif current_pattern == "thermal_hotspot":
            image_src = generate_thermal_hotspot(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        elif current_pattern == "multi_hotspot":
            image_src = generate_multi_hotspot(
                size=self.image_size,
                noise_std=self.noise_std,
            )
        else:
            # Default to asymmetric (safest for rotation estimation)
            logger.warning(f"Unknown pattern type '{current_pattern}', defaulting to 'asymmetric'")
            image_src = generate_asymmetric_pattern(
                size=self.image_size,
                noise_std=self.noise_std,
            )

        # Generate homography with parameters
        H, params = generate_homography(
            rotation_range=self.rotation_range,
            translation_range=self.translation_range,
            scale_range=self.scale_range,
            image_size=self.image_size,
        )

        # Warp to create target
        image_tgt = cv2.warpPerspective(image_src, H, self.image_size)

        # Convert to tensors
        src_tensor = torch.from_numpy(image_src).float().unsqueeze(0) / 255.0
        tgt_tensor = torch.from_numpy(image_tgt).float().unsqueeze(0) / 255.0
        H_tensor = torch.from_numpy(H).float()

        # Convert to 8D vector
        H_normalized = H / (H[2, 2] + 1e-8)
        H_vec = torch.tensor([
            H_normalized[0, 0], H_normalized[0, 1], H_normalized[0, 2],
            H_normalized[1, 0], H_normalized[1, 1], H_normalized[1, 2],
            H_normalized[2, 0], H_normalized[2, 1],
        ], dtype=torch.float32)

        # Get image dimensions for translation normalization
        img_H, img_W = self.image_size

        return {
            "image_src": src_tensor,
            "image_tgt": tgt_tensor,
            "homography": H_tensor,
            "homography_vec": H_vec,
            # Transformation parameters for direct supervision
            "rotation": torch.tensor(params["angle"], dtype=torch.float32),  # radians
            "scale": torch.tensor(params["scale"], dtype=torch.float32),
            # Translation normalized to [-1, 1] range (matching model output convention)
            "translation": torch.tensor([
                params["tx"] / (img_W / 2),
                params["ty"] / (img_H / 2),
            ], dtype=torch.float32),
        }


class RotationEquivarianceDataset(Dataset):
    """
    Dataset for testing rotation equivariance.

    Generates pairs at specific rotation angles to verify
    that model accuracy is independent of input rotation.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        angles: List[float] = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180],
    ):
        self.base_dataset = base_dataset
        self.angles = angles
        self.n_base = len(base_dataset)

    def __len__(self) -> int:
        return self.n_base * len(self.angles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx % self.n_base
        angle_idx = idx // self.n_base

        # Get base sample
        sample = self.base_dataset[base_idx]

        # Apply rotation
        angle = self.angles[angle_idx]
        angle_rad = angle * np.pi / 180

        # Build rotation matrix
        H, W = sample["image_src"].shape[-2:]
        cx, cy = W / 2, H / 2

        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        rotation = T2 @ R @ T1

        # Convert images to numpy for warping
        src_np = (sample["image_src"].squeeze().numpy() * 255).astype(np.uint8)
        tgt_np = (sample["image_tgt"].squeeze().numpy() * 255).astype(np.uint8)

        # Rotate both images
        src_rotated = cv2.warpPerspective(src_np, rotation, (W, H))
        tgt_rotated = cv2.warpPerspective(tgt_np, rotation, (W, H))

        # Update homography
        H_orig = sample["homography"].numpy()
        R_inv = np.linalg.inv(rotation)
        H_new = rotation @ H_orig @ R_inv

        # Convert back to tensors
        src_tensor = torch.from_numpy(src_rotated).float().unsqueeze(0) / 255.0
        tgt_tensor = torch.from_numpy(tgt_rotated).float().unsqueeze(0) / 255.0
        H_tensor = torch.from_numpy(H_new).float()

        # Convert to vec
        H_normalized = H_new / (H_new[2, 2] + 1e-8)
        H_vec = torch.tensor([
            H_normalized[0, 0], H_normalized[0, 1], H_normalized[0, 2],
            H_normalized[1, 0], H_normalized[1, 1], H_normalized[1, 2],
            H_normalized[2, 0], H_normalized[2, 1],
        ], dtype=torch.float32)

        return {
            "image_src": src_tensor,
            "image_tgt": tgt_tensor,
            "homography": H_tensor,
            "homography_vec": H_vec,
            "rotation_angle": torch.tensor(angle),
        }

"""
MPS configuration and debug settings.

This module contains global configuration for Metal Performance Shaders integration,
including debug mode settings and availability checks.
"""

import os

import torch

# ============================================================================
# Debug Mode Configuration
# ============================================================================
# Enable debug mode via environment variable: PIE_METAL_DEBUG=1
# This runs both Metal and PyTorch reference implementations and compares outputs
DEBUG_ENABLED = os.environ.get("PIE_METAL_DEBUG", "0") == "1"
DEBUG_VERBOSITY = int(
    os.environ.get("PIE_METAL_DEBUG_VERBOSITY", "1")
)  # 0=SILENT, 1=SUMMARY, 2=DETAILED, 3=FULL
DEBUG_ATOL = float(os.environ.get("PIE_METAL_DEBUG_ATOL", "1e-5"))  # Absolute tolerance
DEBUG_RTOL = float(os.environ.get("PIE_METAL_DEBUG_RTOL", "1e-3"))  # Relative tolerance

# ============================================================================
# PyTorch Fallback Mode Configuration
# ============================================================================
# Enable PyTorch-only mode via environment variable: PIE_METAL_PYTORCH_MODE=1
# This uses pure PyTorch reference implementations instead of Metal kernels
# Useful for:
# - Testing on systems without Metal support
# - Validating end-to-end functionality without GPU
# - Debugging issues by comparing against known-good PyTorch implementation
PYTORCH_MODE = os.environ.get("PIE_METAL_PYTORCH_MODE", "0") == "1"

# Verbosity levels
VERBOSITY_SILENT = 0
VERBOSITY_SUMMARY = 1
VERBOSITY_DETAILED = 2
VERBOSITY_FULL = 3

# ============================================================================
# MPS Availability Checks
# ============================================================================

# Check if torch.mps.compile_shader is available
try:
    import torch.mps

    # Verify compile_shader attribute exists (sentinel check)
    _ = torch.mps.compile_shader
    MPS_COMPILE_AVAILABLE = True
    print("✅ torch.mps.compile_shader available")
except (ImportError, AttributeError):
    print("❌ torch.mps.compile_shader not available")
    print("   Metal shader compilation requires PyTorch with MPS support")
    print("   Please install PyTorch with MPS: pip install torch>=2.0.0")
    import sys

    sys.exit(1)

# Check if MPS device is available
MPS_DEVICE_AVAILABLE = (
    torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
)

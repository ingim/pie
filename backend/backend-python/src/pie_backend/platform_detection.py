"""
Platform Detection Utilities

Provides functions to detect hardware capabilities and platform features.
"""

import platform


def is_apple_silicon() -> bool:
    """Check if running on macOS with Apple Silicon (M1/M2/M3/M4).

    Returns:
        True if running on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.processor() == "arm"


def is_macos() -> bool:
    """Check if running on macOS.

    Returns:
        True if running on macOS, False otherwise
    """
    return platform.system() == "Darwin"
